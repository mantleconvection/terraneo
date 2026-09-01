#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Python.h insists on being included before any system header.
#include <Python.h>

#include "communication/shell/communication.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/solvers/solver.hpp"
#include "linalg/vector_q1.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "util/logging.hpp"

/// @file
/// @brief A TERRA-NG solver that hands the FE function to an embedded Python model.
///
/// `NeuralSolver` satisfies @ref terra::linalg::solvers::SolverLike, so
///
///     linalg::solvers::solve( neural, K, u, f );
///
/// compiles wherever FGMRES would. `solve_impl` copies the FE right-hand side to
/// the host, calls `terra_infer.call( model, buffers, shapes )` in an embedded
/// CPython interpreter, and writes the returned arrays back into the solution.
///
/// The interpreter lives in this process. Fields cross the boundary as
/// `memoryview`s over our own host buffers, so `np.frombuffer` on the Python side
/// is a view, not a copy; results come back through the buffer protocol, which
/// numpy arrays implement, so nothing here needs the numpy C API.
///
/// \section build Build and run requirements
///
/// Configure with `-DTERRA_ENABLE_PYTHON=ON`; the header is inert otherwise.
/// Two environment facts have to line up on this cluster, both verified:
///
/// - The conda Python stack needs GLIBCXX_3.4.30+, while the system libstdc++
///   that `nvc++` links against stops at 3.4.29. CMake puts conda's newer (and
///   backward-compatible) libstdc++ on the link line when it finds one.
/// - PyTorch ships its own CUDA runtime (cu128) and will not import against the
///   nvhpc 24.11 runtime already loaded in the process. Preload torch's:
///
///       export LD_PRELOAD=<site-packages>/nvidia/cuda_runtime/lib/libcudart.so.12
///
/// The interpreter is initialised once and never finalised: `Py_Finalize` with
/// torch loaded is unreliable, and the process wants the interpreter for its
/// whole lifetime anyway.
namespace terra::ml {

namespace detail {

/// @brief Formats the pending Python exception (with traceback) and clears it.
inline std::string python_error_message()
{
    if ( PyErr_Occurred() == nullptr )
    {
        return "no Python error set";
    }

    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    PyErr_Fetch( &type, &value, &traceback );
    PyErr_NormalizeException( &type, &value, &traceback );

    std::string text;

    PyObject* tb_module = PyImport_ImportModule( "traceback" );
    if ( tb_module != nullptr )
    {
        PyObject* lines = PyObject_CallMethod(
            tb_module, "format_exception", "OOO", type, value ? value : Py_None, traceback ? traceback : Py_None );
        if ( lines != nullptr )
        {
            const Py_ssize_t n = PyList_Size( lines );
            for ( Py_ssize_t i = 0; i < n; ++i )
            {
                PyObject* line = PyList_GetItem( lines, i ); // borrowed
                if ( const char* s = PyUnicode_AsUTF8( line ) )
                {
                    text += s;
                }
            }
            Py_DECREF( lines );
        }
        Py_DECREF( tb_module );
    }

    if ( text.empty() && value != nullptr )
    {
        if ( PyObject* repr = PyObject_Repr( value ) )
        {
            if ( const char* s = PyUnicode_AsUTF8( repr ) )
                text = s;
            Py_DECREF( repr );
        }
    }

    Py_XDECREF( type );
    Py_XDECREF( value );
    Py_XDECREF( traceback );
    PyErr_Clear();

    return text.empty() ? "unknown Python error" : text;
}

/// @brief RAII for a borrowed-or-owned PyObject*, so early returns cannot leak.
class PyRef
{
  public:
    PyRef() = default;
    explicit PyRef( PyObject* obj )
    : obj_( obj )
    {}

    ~PyRef() { Py_XDECREF( obj_ ); }

    PyRef( const PyRef& )            = delete;
    PyRef& operator=( const PyRef& ) = delete;

    PyRef( PyRef&& other ) noexcept
    : obj_( other.obj_ )
    {
        other.obj_ = nullptr;
    }

    [[nodiscard]] PyObject* get() const { return obj_; }
    [[nodiscard]] explicit  operator bool() const { return obj_ != nullptr; }

    PyObject* release()
    {
        PyObject* tmp = obj_;
        obj_          = nullptr;
        return tmp;
    }

  private:
    PyObject* obj_ = nullptr;
};

/// @brief Initialises CPython on first use. Never finalises: see the file comment.
inline void ensure_interpreter()
{
    if ( Py_IsInitialized() == 0 )
    {
        Py_InitializeEx( /*initsigs=*/0 ); // leave signal handlers to MPI/Kokkos
        util::logroot << "NeuralSolver: embedded CPython initialised\n";
    }
}

/// @brief A model predicts each subdomain independently, so nodes shared between
///        subdomains come back disagreeing. Zero every non-owned copy, then SUM
///        exchange: the owner's value ends up on every copy.
template < typename ScalarT >
void make_consistent_scalar( const grid::shell::DistributedDomain& domain, linalg::VectorQ1Scalar< ScalarT >& x )
{
    auto       data = x.grid_data();
    const auto mask = x.mask_data();

    Kokkos::parallel_for(
        "neural_zero_not_owned",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( int s, int i, int j, int k ) {
            if ( !util::has_flag( mask( s, i, j, k ), grid::NodeOwnershipFlag::OWNED ) )
                data( s, i, j, k ) = ScalarT( 0 );
        } );
    Kokkos::fence();

    communication::shell::send_recv( domain, data, communication::CommunicationReduction::SUM );
}

template < typename ScalarT, int VecDim >
void make_consistent_vec( const grid::shell::DistributedDomain& domain, linalg::VectorQ1Vec< ScalarT, VecDim >& x )
{
    const auto mask = x.mask_data();

    for ( int d = 0; d < VecDim; ++d )
    {
        auto data = x.grid_data().comp_[d];

        Kokkos::parallel_for(
            "neural_zero_not_owned",
            grid::shell::local_domain_md_range_policy_nodes( domain ),
            KOKKOS_LAMBDA( int s, int i, int j, int k ) {
                if ( !util::has_flag( mask( s, i, j, k ), grid::NodeOwnershipFlag::OWNED ) )
                    data( s, i, j, k ) = ScalarT( 0 );
            } );
        Kokkos::fence();

        communication::shell::send_recv( domain, data, communication::CommunicationReduction::SUM );
    }
}

} // namespace detail

/// @brief One FE field, flattened to `[n_subdomains, nx, ny, nr, n_components]`.
///
/// The buffer is reused across solves, so a steady-state run does no allocation.
struct HostField
{
    std::string          name;
    int                  n_subdomains = 0;
    int                  nx = 0, ny = 0, nr = 0;
    int                  n_components = 1;
    std::vector< float > values;

    [[nodiscard]] std::size_t num_values() const
    {
        return static_cast< std::size_t >( n_subdomains ) * nx * ny * nr * n_components;
    }
};

struct NeuralSolverOptions
{
    std::string module = "terra_infer"; ///< Python module to import.
    std::string entry  = "call";        ///< Callable in that module: call(model, buffers, shapes) -> dict.
    std::string model  = "zero";        ///< Model name handed to the callable.

    bool log_residual = true; ///< Log ||b - Ax|| before and after. Costs two matvecs.
};

/// @brief Forwards the FE function to an embedded Python model and writes the reply into `x`.
///
/// Handles both vector shapes that reach a solver here: a plain `VectorQ1Vec`
/// (one field, "u") and the block `VectorQ1IsoQ2Q1` used by the Stokes system
/// (two fields, "u" on the velocity domain and "p" on the coarser pressure domain).
template < linalg::OperatorLike OperatorT >
class NeuralSolver
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = linalg::SrcOf< OperatorType >;
    using RHSVectorType      = linalg::DstOf< OperatorType >;
    using ScalarType         = typename SolutionVectorType::ScalarType;

    static constexpr bool is_block = linalg::Block2VectorLike< SolutionVectorType >;

    /// @param options          Module, entry point and model name.
    /// @param velocity_domain  Domain of the (first) block.
    /// @param pressure_domain  Domain of the second block; pass the same domain if unused.
    /// @param residual_tmp     Scratch, only touched when `log_residual` is set.
    NeuralSolver(
        NeuralSolverOptions                   options,
        const grid::shell::DistributedDomain& velocity_domain,
        const grid::shell::DistributedDomain& pressure_domain,
        SolutionVectorType&                   residual_tmp )
    : options_( std::move( options ) )
    , velocity_domain_( velocity_domain )
    , pressure_domain_( pressure_domain )
    , residual_tmp_( residual_tmp )
    {
        detail::ensure_interpreter();

        detail::PyRef module( PyImport_ImportModule( options_.module.c_str() ) );
        if ( !module )
        {
            throw std::runtime_error(
                "NeuralSolver: cannot import '" + options_.module + "':\n" + detail::python_error_message() );
        }

        entry_ = PyObject_GetAttrString( module.get(), options_.entry.c_str() );
        if ( entry_ == nullptr )
        {
            throw std::runtime_error(
                "NeuralSolver: '" + options_.module + "' has no '" + options_.entry + "':\n" +
                detail::python_error_message() );
        }

        // Allocate the host staging buffers once.
        if constexpr ( is_block )
        {
            fields_.push_back( make_field( "u", velocity_domain_, 3 ) );
            fields_.push_back( make_field( "p", pressure_domain_, 1 ) );
        }
        else
        {
            fields_.push_back( make_field( "u", velocity_domain_, components_of< SolutionVectorType >() ) );
        }

        util::logroot << "NeuralSolver: " << options_.module << "." << options_.entry << ", model '" << options_.model
                      << "'\n";
    }

    ~NeuralSolver() { Py_XDECREF( entry_ ); }

    NeuralSolver( const NeuralSolver& )            = delete;
    NeuralSolver& operator=( const NeuralSolver& ) = delete;

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        const double before = options_.log_residual ? residual_norm( A, x, b ) : 0.0;

        // FE right-hand side -> host staging buffers.
        if constexpr ( is_block )
        {
            pack_vec( b.block_1(), fields_[0] );
            pack_scalar( b.block_2(), fields_[1] );
        }
        else
        {
            pack_vec( b, fields_[0] );
        }

        call_python();

        // Python's arrays -> device, then repair the shared-node disagreement.
        if constexpr ( is_block )
        {
            unpack_vec( fields_[0], x.block_1() );
            unpack_scalar( fields_[1], x.block_2() );
            detail::make_consistent_vec( velocity_domain_, x.block_1() );
            detail::make_consistent_scalar( pressure_domain_, x.block_2() );
        }
        else
        {
            unpack_vec( fields_[0], x );
            detail::make_consistent_vec( velocity_domain_, x );
        }

        if ( options_.log_residual )
        {
            const double after = residual_norm( A, x, b );
            util::logroot << "NeuralSolver: ||b - A x|| " << before << " -> " << after << "\n";
        }
    }

  private:
    template < typename VectorT >
    static constexpr int components_of()
    {
        if constexpr ( requires { VectorT::Dim; } )
            return VectorT::Dim;
        else
            return 1;
    }

    // ---- the Python call -------------------------------------------------------------

    /// @brief `entry( model, {name: memoryview}, {name: shape} )`, results copied back
    ///        into the same staging buffers.
    ///
    /// The request memoryviews are read-only views of `fields_[i].values`, so nothing is
    /// copied on the way in. The reply is read through the buffer protocol, which numpy
    /// arrays implement, so no numpy C API is needed here.
    void call_python()
    {
        detail::PyRef buffers( PyDict_New() );
        detail::PyRef shapes( PyDict_New() );
        if ( !buffers || !shapes )
            throw std::runtime_error( "NeuralSolver: out of memory building the request." );

        for ( auto& f : fields_ )
        {
            detail::PyRef view( PyMemoryView_FromMemory(
                reinterpret_cast< char* >( f.values.data() ),
                static_cast< Py_ssize_t >( f.values.size() * sizeof( float ) ),
                PyBUF_READ ) );
            detail::PyRef shape(
                Py_BuildValue( "(iiiii)", f.n_subdomains, f.nx, f.ny, f.nr, f.n_components ) );
            if ( !view || !shape )
                throw std::runtime_error( "NeuralSolver: cannot wrap '" + f.name + "' for Python." );

            PyDict_SetItemString( buffers.get(), f.name.c_str(), view.get() );
            PyDict_SetItemString( shapes.get(), f.name.c_str(), shape.get() );
        }

        detail::PyRef result( PyObject_CallFunction(
            entry_, "sOO", options_.model.c_str(), buffers.get(), shapes.get() ) );
        if ( !result )
        {
            throw std::runtime_error( "NeuralSolver: the model raised:\n" + detail::python_error_message() );
        }
        if ( !PyDict_Check( result.get() ) )
        {
            throw std::runtime_error( "NeuralSolver: the model must return a dict of arrays." );
        }

        for ( auto& f : fields_ )
        {
            PyObject* array = PyDict_GetItemString( result.get(), f.name.c_str() ); // borrowed
            if ( array == nullptr )
            {
                throw std::runtime_error( "NeuralSolver: the model returned no field '" + f.name + "'." );
            }

            Py_buffer view;
            if ( PyObject_GetBuffer( array, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT ) != 0 )
            {
                throw std::runtime_error(
                    "NeuralSolver: '" + f.name + "' is not a C-contiguous buffer:\n" +
                    detail::python_error_message() );
            }

            const bool ok = view.itemsize == static_cast< Py_ssize_t >( sizeof( float ) ) &&
                            view.format != nullptr && std::strcmp( view.format, "f" ) == 0 &&
                            static_cast< std::size_t >( view.len ) == f.values.size() * sizeof( float );
            if ( !ok )
            {
                const auto len    = static_cast< long long >( view.len );
                const auto format = view.format ? std::string( view.format ) : std::string( "?" );
                PyBuffer_Release( &view );
                throw std::runtime_error(
                    "NeuralSolver: '" + f.name + "' must be float32 and " +
                    std::to_string( f.values.size() * sizeof( float ) ) + " bytes, got format '" + format +
                    "' of " + std::to_string( len ) + " bytes." );
            }

            std::memcpy( f.values.data(), view.buf, static_cast< std::size_t >( view.len ) );
            PyBuffer_Release( &view );
        }
    }

    // ---- staging ---------------------------------------------------------------------

    static HostField
        make_field( const std::string& name, const grid::shell::DistributedDomain& domain, const int n_components )
    {
        HostField f;
        f.name         = name;
        f.n_subdomains = static_cast< int >( domain.subdomains().size() );
        f.nx           = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        f.ny           = f.nx;
        f.nr           = domain.domain_info().subdomain_num_nodes_radially();
        f.n_components = n_components;
        f.values.assign( f.num_values(), 0.0f );
        return f;
    }

    template < typename ScalarT, int VecDim >
    static void pack_vec( const linalg::VectorQ1Vec< ScalarT, VecDim >& v, HostField& f )
    {
        for ( int d = 0; d < VecDim; ++d )
        {
            auto host = Kokkos::create_mirror( Kokkos::HostSpace{}, v.grid_data().comp_[d] );
            Kokkos::deep_copy( host, v.grid_data().comp_[d] );

            std::size_t flat = 0;
            for ( int s = 0; s < f.n_subdomains; ++s )
                for ( int i = 0; i < f.nx; ++i )
                    for ( int j = 0; j < f.ny; ++j )
                        for ( int k = 0; k < f.nr; ++k, ++flat )
                            f.values[flat * VecDim + d] = static_cast< float >( host( s, i, j, k ) );
        }
    }

    template < typename ScalarT >
    static void pack_scalar( const linalg::VectorQ1Scalar< ScalarT >& v, HostField& f )
    {
        auto host = Kokkos::create_mirror( Kokkos::HostSpace{}, v.grid_data() );
        Kokkos::deep_copy( host, v.grid_data() );

        std::size_t flat = 0;
        for ( int s = 0; s < f.n_subdomains; ++s )
            for ( int i = 0; i < f.nx; ++i )
                for ( int j = 0; j < f.ny; ++j )
                    for ( int k = 0; k < f.nr; ++k, ++flat )
                        f.values[flat] = static_cast< float >( host( s, i, j, k ) );
    }

    template < typename ScalarT, int VecDim >
    static void unpack_vec( const HostField& f, linalg::VectorQ1Vec< ScalarT, VecDim >& v )
    {
        if ( f.n_components != VecDim )
            throw std::runtime_error( "NeuralSolver: '" + f.name + "' has the wrong component count." );

        for ( int d = 0; d < VecDim; ++d )
        {
            auto host = Kokkos::create_mirror( Kokkos::HostSpace{}, v.grid_data().comp_[d] );
            check_extents( f, host, v.grid_data().comp_[d].label() );

            std::size_t flat = 0;
            for ( int s = 0; s < f.n_subdomains; ++s )
                for ( int i = 0; i < f.nx; ++i )
                    for ( int j = 0; j < f.ny; ++j )
                        for ( int k = 0; k < f.nr; ++k, ++flat )
                            host( s, i, j, k ) = static_cast< ScalarT >( f.values[flat * VecDim + d] );

            Kokkos::deep_copy( v.grid_data().comp_[d], host );
        }
    }

    template < typename ScalarT >
    static void unpack_scalar( const HostField& f, linalg::VectorQ1Scalar< ScalarT >& v )
    {
        auto host = Kokkos::create_mirror( Kokkos::HostSpace{}, v.grid_data() );
        check_extents( f, host, v.grid_data().label() );

        std::size_t flat = 0;
        for ( int s = 0; s < f.n_subdomains; ++s )
            for ( int i = 0; i < f.nx; ++i )
                for ( int j = 0; j < f.ny; ++j )
                    for ( int k = 0; k < f.nr; ++k, ++flat )
                        host( s, i, j, k ) = static_cast< ScalarT >( f.values[flat] );

        Kokkos::deep_copy( v.grid_data(), host );
    }

    template < typename HostView >
    static void check_extents( const HostField& f, const HostView& host, const std::string& label )
    {
        if ( static_cast< std::size_t >( f.n_subdomains ) != host.extent( 0 ) ||
             static_cast< std::size_t >( f.nx ) != host.extent( 1 ) ||
             static_cast< std::size_t >( f.ny ) != host.extent( 2 ) ||
             static_cast< std::size_t >( f.nr ) != host.extent( 3 ) )
        {
            throw std::runtime_error( "NeuralSolver: shape of '" + f.name + "' does not fit '" + label + "'." );
        }
    }

    double residual_norm( OperatorType& A, const SolutionVectorType& x, const RHSVectorType& b )
    {
        linalg::apply( A, const_cast< SolutionVectorType& >( x ), residual_tmp_ );
        linalg::lincomb( residual_tmp_, { ScalarType( 1 ), ScalarType( -1 ) }, { b, residual_tmp_ } );
        return static_cast< double >( linalg::norm_2( residual_tmp_ ) );
    }

    NeuralSolverOptions                   options_;
    const grid::shell::DistributedDomain& velocity_domain_;
    const grid::shell::DistributedDomain& pressure_domain_;
    SolutionVectorType&                   residual_tmp_;

    std::vector< HostField > fields_;
    PyObject*                entry_ = nullptr;
};

} // namespace terra::ml
