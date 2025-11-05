#pragma once

#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::linalg {

template < terra::linalg::OperatorLike OperatorT >
class InvDiagOperator
{
  public:
    using OperatorType  = OperatorT;
    //using SrcVectorType = SrcOf< OperatorT >;
    //using DstVectorType = DstOf< OperatorT >;
    //using ScalarType    = ScalarOf< DstVectorType >;
    using SrcVectorType = linalg::VectorQ1Scalar< double >;
    using DstVectorType = linalg::VectorQ1Scalar< double >;
    using ScalarType    = ScalarOf< DstVectorType >;

  private:
    OperatorT&     op_;
    SrcVectorType& inv_diag_;

  public:
    explicit InvDiagOperator( OperatorT& op, SrcVectorType& inv_diag )
    : op_( op )
    , inv_diag_( inv_diag ) {};

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        apply( op_, src, dst );
        scale_in_place( dst, inv_diag_ );
    }
};

static_assert( linalg::OperatorLike< InvDiagOperator< terra::fe::wedge::operators::shell::LaplaceSimple< double > > > );

} // namespace terra::linalg