
#include <optional>

#include "terra/communication/communication.hpp"
#include "terra/dense/mat.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/vector_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/vtk/vtk.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;

struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        bool                              only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const double                  value  = coords( 0 ) * Kokkos::sin( coords( 1 ) ) * Kokkos::cos( coords( 2 ) );
        // const double value = coords( 0 );

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

struct SetOnBoundary
{
    Grid4DDataScalar< double > src_;
    Grid4DDataScalar< double > dst_;
    int                        num_shells_;

    SetOnBoundary( const Grid4DDataScalar< double >& src, const Grid4DDataScalar< double >& dst, const int num_shells )
    : src_( src )
    , dst_( dst )
    , num_shells_( num_shells )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_idx, const int x, const int y, const int r ) const
    {
        if ( ( r == 0 || r == num_shells_ - 1 ) )
        {
            dst_( local_subdomain_idx, x, y, r ) = src_( local_subdomain_idx, x, y, r );
        }
    }
};

struct LaplaceOperator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;

    Grid4DDataScalar< double > src_;
    Grid4DDataScalar< double > dst_;
    bool                       treat_boundary_;
    bool                       diagonal_;

    LaplaceOperator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& src,
        const Grid4DDataScalar< double >& dst,
        const bool                        treat_boundary,
        const bool                        diagonal )
    : grid_( grid )
    , radii_( radii )
    , src_( src )
    , dst_( dst )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    {}

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Extract vertex positions
        dense::Vec< double, 3 > coords[2][2][2];

        for ( int x = x_cell; x <= x_cell + 1; x++ )
        {
            for ( int y = y_cell; y <= y_cell + 1; y++ )
            {
                for ( int r = r_cell; r <= r_cell + 1; r++ )
                {
                    coords[x - x_cell][y - y_cell][r - r_cell] =
                        grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
                }
            }
        }

        const double x1 = coords[0][0][0]( 0 );
        const double x2 = coords[1][0][0]( 0 );
        const double x3 = coords[0][1][0]( 0 );
        const double x4 = coords[1][1][0]( 0 );
        const double x5 = coords[0][0][1]( 0 );
        const double x6 = coords[1][0][1]( 0 );
        const double x7 = coords[0][1][1]( 0 );
        const double x8 = coords[1][1][1]( 0 );

        const double y1 = coords[0][0][0]( 1 );
        const double y2 = coords[1][0][0]( 1 );
        const double y3 = coords[0][1][0]( 1 );
        const double y4 = coords[1][1][0]( 1 );
        const double y5 = coords[0][0][1]( 1 );
        const double y6 = coords[1][0][1]( 1 );
        const double y7 = coords[0][1][1]( 1 );
        const double y8 = coords[1][1][1]( 1 );

        const double z1 = coords[0][0][0]( 2 );
        const double z2 = coords[1][0][0]( 2 );
        const double z3 = coords[0][1][0]( 2 );
        const double z4 = coords[1][1][0]( 2 );
        const double z5 = coords[0][0][1]( 2 );
        const double z6 = coords[1][0][1]( 2 );
        const double z7 = coords[0][1][1]( 2 );
        const double z8 = coords[1][1][1]( 2 );

        // Gauss-Lobatto quadrature
        constexpr int nq    = 3;
        const double  q[nq] = { -1.0, 0.0, 1.0 };
        const double  w[nq] = { 1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0 };

        dense::Mat< double, 8, 8 > A;
        dense::Vec< double, 8 >    src;

        for ( int r = r_cell; r <= r_cell + 1; r++ )
        {
            for ( int y = y_cell; y <= y_cell + 1; y++ )
            {
                for ( int x = x_cell; x <= x_cell + 1; x++ )
                {
                    const int x_local                          = x - x_cell;
                    const int y_local                          = y - y_cell;
                    const int r_local                          = r - r_cell;
                    src( 4 * r_local + 2 * y_local + x_local ) = src_( local_subdomain_id, x, y, r );
                }
            }
        }

        for ( int qx = 0; qx < nq; qx++ )
        {
            for ( int qy = 0; qy < nq; qy++ )
            {
                for ( int qr = 0; qr < nq; qr++ )
                {
                    dense::Mat< double, 8, 8 > dNdx;

                    const double www = w[qx] * w[qy] * w[qr];

                    const double qp0 = q[qx];
                    const double qp1 = q[qy];
                    const double qp2 = q[qr];

                    // --- CSE temporaries ---
                    double tmp_0  = qp1 - 1;
                    double tmp_1  = -tmp_0;
                    double tmp_2  = qp2 - 1;
                    double tmp_3  = -tmp_2;
                    double tmp_4  = qp1 + 1;
                    double tmp_5  = qp2 + 1;
                    double tmp_6  = ( 1.0 / 8.0 ) * tmp_1;
                    double tmp_7  = tmp_3 * tmp_6;
                    double tmp_8  = ( 1.0 / 8.0 ) * tmp_4;
                    double tmp_9  = tmp_3 * tmp_8;
                    double tmp_10 = tmp_5 * tmp_6;
                    double tmp_11 = tmp_5 * tmp_8;
                    double tmp_12 = tmp_11 * x7 - tmp_11 * x8;
                    double tmp_13 = ( 1.0 / 8.0 ) * tmp_1 * tmp_3 * x2 + ( 1.0 / 8.0 ) * tmp_1 * tmp_5 * x6 -
                                    tmp_10 * x5 - tmp_12 + ( 1.0 / 8.0 ) * tmp_3 * tmp_4 * x4 - tmp_7 * x1 - tmp_9 * x3;
                    double tmp_14 = qp0 - 1;
                    double tmp_15 = -tmp_14;
                    double tmp_16 = qp0 + 1;
                    double tmp_17 = ( 1.0 / 8.0 ) * tmp_3;
                    double tmp_18 = tmp_15 * tmp_17;
                    double tmp_19 = tmp_16 * tmp_17;
                    double tmp_20 = ( 1.0 / 8.0 ) * tmp_5;
                    double tmp_21 = tmp_15 * tmp_20;
                    double tmp_22 = tmp_16 * tmp_20;
                    double tmp_23 = tmp_22 * y6 - tmp_22 * y8;
                    double tmp_24 = ( 1.0 / 8.0 ) * tmp_15 * tmp_3 * y3 + ( 1.0 / 8.0 ) * tmp_15 * tmp_5 * y7 +
                                    ( 1.0 / 8.0 ) * tmp_16 * tmp_3 * y4 - tmp_18 * y1 - tmp_19 * y2 - tmp_21 * y5 -
                                    tmp_23;
                    double tmp_25 = tmp_13 * tmp_24;
                    double tmp_26 = tmp_22 * x6 - tmp_22 * x8;
                    double tmp_27 = ( 1.0 / 8.0 ) * tmp_15 * tmp_3 * x3 + ( 1.0 / 8.0 ) * tmp_15 * tmp_5 * x7 +
                                    ( 1.0 / 8.0 ) * tmp_16 * tmp_3 * x4 - tmp_18 * x1 - tmp_19 * x2 - tmp_21 * x5 -
                                    tmp_26;
                    double tmp_28 = tmp_11 * y7 - tmp_11 * y8;
                    double tmp_29 = ( 1.0 / 8.0 ) * tmp_1 * tmp_3 * y2 + ( 1.0 / 8.0 ) * tmp_1 * tmp_5 * y6 -
                                    tmp_10 * y5 - tmp_28 + ( 1.0 / 8.0 ) * tmp_3 * tmp_4 * y4 - tmp_7 * y1 - tmp_9 * y3;
                    double tmp_30 = tmp_27 * tmp_29;
                    double tmp_31 = tmp_25 - tmp_30;
                    double tmp_32 = tmp_22 * z6 - tmp_22 * z8;
                    double tmp_33 = ( 1.0 / 8.0 ) * tmp_15 * tmp_3 * z3 + ( 1.0 / 8.0 ) * tmp_15 * tmp_5 * z7 +
                                    ( 1.0 / 8.0 ) * tmp_16 * tmp_3 * z4 - tmp_18 * z1 - tmp_19 * z2 - tmp_21 * z5 -
                                    tmp_32;
                    double tmp_34 = tmp_15 * tmp_6;
                    double tmp_35 = tmp_16 * tmp_6;
                    double tmp_36 = tmp_15 * tmp_8;
                    double tmp_37 = tmp_16 * tmp_8;
                    double tmp_38 = tmp_37 * x4 - tmp_37 * x8;
                    double tmp_39 = ( 1.0 / 8.0 ) * tmp_1 * tmp_15 * x5 + ( 1.0 / 8.0 ) * tmp_1 * tmp_16 * x6 +
                                    ( 1.0 / 8.0 ) * tmp_15 * tmp_4 * x7 - tmp_34 * x1 - tmp_35 * x2 - tmp_36 * x3 -
                                    tmp_38;
                    double tmp_40 = tmp_29 * tmp_39;
                    double tmp_41 = tmp_37 * z4 - tmp_37 * z8;
                    double tmp_42 = ( 1.0 / 8.0 ) * tmp_1 * tmp_15 * z5 + ( 1.0 / 8.0 ) * tmp_1 * tmp_16 * z6 +
                                    ( 1.0 / 8.0 ) * tmp_15 * tmp_4 * z7 - tmp_34 * z1 - tmp_35 * z2 - tmp_36 * z3 -
                                    tmp_41;
                    double tmp_43 = tmp_11 * z7 - tmp_11 * z8;
                    double tmp_44 = ( 1.0 / 8.0 ) * tmp_1 * tmp_3 * z2 + ( 1.0 / 8.0 ) * tmp_1 * tmp_5 * z6 -
                                    tmp_10 * z5 + ( 1.0 / 8.0 ) * tmp_3 * tmp_4 * z4 - tmp_43 - tmp_7 * z1 - tmp_9 * z3;
                    double tmp_45 = tmp_37 * y4 - tmp_37 * y8;
                    double tmp_46 = ( 1.0 / 8.0 ) * tmp_1 * tmp_15 * y5 + ( 1.0 / 8.0 ) * tmp_1 * tmp_16 * y6 +
                                    ( 1.0 / 8.0 ) * tmp_15 * tmp_4 * y7 - tmp_34 * y1 - tmp_35 * y2 - tmp_36 * y3 -
                                    tmp_45;
                    double tmp_47 = tmp_24 * tmp_39;
                    double tmp_48 = tmp_13 * tmp_46;
                    double tmp_49 = 1.0 / ( tmp_25 * tmp_42 + tmp_27 * tmp_44 * tmp_46 - tmp_30 * tmp_42 +
                                            tmp_33 * tmp_40 - tmp_33 * tmp_48 - tmp_44 * tmp_47 );
                    double tmp_50 = tmp_34 * tmp_49;
                    double tmp_51 = tmp_31 * tmp_50;
                    double tmp_52 = tmp_27 * tmp_46 - tmp_47;
                    double tmp_53 = tmp_49 * tmp_7;
                    double tmp_54 = tmp_52 * tmp_53;
                    double tmp_55 = tmp_40 - tmp_48;
                    double tmp_56 = tmp_18 * tmp_49;
                    double tmp_57 = tmp_55 * tmp_56;
                    double tmp_58 = -tmp_51 - tmp_54 - tmp_57;
                    double tmp_59 = -tmp_13 * tmp_33 + tmp_27 * tmp_44;
                    double tmp_60 = tmp_50 * tmp_59;
                    double tmp_61 = -tmp_27 * tmp_42 + tmp_33 * tmp_39;
                    double tmp_62 = tmp_53 * tmp_61;
                    double tmp_63 = tmp_13 * tmp_42 - tmp_39 * tmp_44;
                    double tmp_64 = tmp_56 * tmp_63;
                    double tmp_65 = -tmp_60 - tmp_62 - tmp_64;
                    double tmp_66 = -tmp_24 * tmp_44 + tmp_29 * tmp_33;
                    double tmp_67 = tmp_50 * tmp_66;
                    double tmp_68 = tmp_24 * tmp_42 - tmp_33 * tmp_46;
                    double tmp_69 = tmp_53 * tmp_68;
                    double tmp_70 = -tmp_29 * tmp_42 + tmp_44 * tmp_46;
                    double tmp_71 = tmp_56 * tmp_70;
                    double tmp_72 = -tmp_67 - tmp_69 - tmp_71;
                    double tmp_73 = ( 1.0 / 8.0 ) * tmp_0;
                    double tmp_74 = tmp_16 * tmp_73;
                    double tmp_75 = tmp_14 * tmp_8;
                    double tmp_76 = tmp_14 * tmp_73;
                    double tmp_77 =
                        tmp_38 - tmp_74 * x2 + tmp_74 * x6 - tmp_75 * x3 + tmp_75 * x7 + tmp_76 * x1 - tmp_76 * x5;
                    double tmp_78 = tmp_2 * tmp_73;
                    double tmp_79 = tmp_2 * tmp_8;
                    double tmp_80 = tmp_0 * tmp_20;
                    double tmp_81 =
                        tmp_28 + tmp_78 * y1 - tmp_78 * y2 - tmp_79 * y3 + tmp_79 * y4 - tmp_80 * y5 + tmp_80 * y6;
                    double tmp_82 = ( 1.0 / 8.0 ) * tmp_2;
                    double tmp_83 = tmp_16 * tmp_82;
                    double tmp_84 = tmp_14 * tmp_82;
                    double tmp_85 = tmp_14 * tmp_20;
                    double tmp_86 =
                        tmp_32 - tmp_83 * z2 + tmp_83 * z4 + tmp_84 * z1 - tmp_84 * z3 - tmp_85 * z5 + tmp_85 * z7;
                    double tmp_87 =
                        tmp_12 + tmp_78 * x1 - tmp_78 * x2 - tmp_79 * x3 + tmp_79 * x4 - tmp_80 * x5 + tmp_80 * x6;
                    double tmp_88 =
                        tmp_23 - tmp_83 * y2 + tmp_83 * y4 + tmp_84 * y1 - tmp_84 * y3 - tmp_85 * y5 + tmp_85 * y7;
                    double tmp_89 =
                        tmp_41 - tmp_74 * z2 + tmp_74 * z6 - tmp_75 * z3 + tmp_75 * z7 + tmp_76 * z1 - tmp_76 * z5;
                    double tmp_90 =
                        tmp_26 - tmp_83 * x2 + tmp_83 * x4 + tmp_84 * x1 - tmp_84 * x3 - tmp_85 * x5 + tmp_85 * x7;
                    double tmp_91 =
                        tmp_45 - tmp_74 * y2 + tmp_74 * y6 - tmp_75 * y3 + tmp_75 * y7 + tmp_76 * y1 - tmp_76 * y5;
                    double tmp_92 =
                        tmp_43 + tmp_78 * z1 - tmp_78 * z2 - tmp_79 * z3 + tmp_79 * z4 - tmp_80 * z5 + tmp_80 * z6;
                    double tmp_93 =
                        www * fabs(
                                  tmp_77 * tmp_81 * tmp_86 - tmp_77 * tmp_88 * tmp_92 - tmp_81 * tmp_89 * tmp_90 -
                                  tmp_86 * tmp_87 * tmp_91 + tmp_87 * tmp_88 * tmp_89 + tmp_90 * tmp_91 * tmp_92 );
                    double tmp_94  = tmp_35 * tmp_49;
                    double tmp_95  = tmp_31 * tmp_94;
                    double tmp_96  = tmp_19 * tmp_49;
                    double tmp_97  = tmp_55 * tmp_96;
                    double tmp_98  = tmp_54 - tmp_95 - tmp_97;
                    double tmp_99  = tmp_59 * tmp_94;
                    double tmp_100 = tmp_63 * tmp_96;
                    double tmp_101 = -tmp_100 + tmp_62 - tmp_99;
                    double tmp_102 = tmp_66 * tmp_94;
                    double tmp_103 = tmp_70 * tmp_96;
                    double tmp_104 = -tmp_102 - tmp_103 + tmp_69;
                    double tmp_105 = tmp_93 * ( tmp_101 * tmp_65 + tmp_104 * tmp_72 + tmp_58 * tmp_98 );
                    double tmp_106 = tmp_36 * tmp_49;
                    double tmp_107 = tmp_106 * tmp_31;
                    double tmp_108 = tmp_49 * tmp_9;
                    double tmp_109 = tmp_108 * tmp_52;
                    double tmp_110 = -tmp_107 - tmp_109 + tmp_57;
                    double tmp_111 = tmp_106 * tmp_59;
                    double tmp_112 = tmp_108 * tmp_61;
                    double tmp_113 = -tmp_111 - tmp_112 + tmp_64;
                    double tmp_114 = tmp_106 * tmp_66;
                    double tmp_115 = tmp_108 * tmp_68;
                    double tmp_116 = -tmp_114 - tmp_115 + tmp_71;
                    double tmp_117 = tmp_93 * ( tmp_110 * tmp_58 + tmp_113 * tmp_65 + tmp_116 * tmp_72 );
                    double tmp_118 = tmp_37 * tmp_49;
                    double tmp_119 = tmp_118 * tmp_31;
                    double tmp_120 = tmp_109 - tmp_119 + tmp_97;
                    double tmp_121 = tmp_118 * tmp_59;
                    double tmp_122 = tmp_100 + tmp_112 - tmp_121;
                    double tmp_123 = tmp_118 * tmp_66;
                    double tmp_124 = tmp_103 + tmp_115 - tmp_123;
                    double tmp_125 = tmp_93 * ( tmp_120 * tmp_58 + tmp_122 * tmp_65 + tmp_124 * tmp_72 );
                    double tmp_126 = tmp_10 * tmp_49;
                    double tmp_127 = tmp_126 * tmp_52;
                    double tmp_128 = tmp_21 * tmp_49;
                    double tmp_129 = tmp_128 * tmp_55;
                    double tmp_130 = -tmp_127 - tmp_129 + tmp_51;
                    double tmp_131 = tmp_126 * tmp_61;
                    double tmp_132 = tmp_128 * tmp_63;
                    double tmp_133 = -tmp_131 - tmp_132 + tmp_60;
                    double tmp_134 = tmp_126 * tmp_68;
                    double tmp_135 = tmp_128 * tmp_70;
                    double tmp_136 = -tmp_134 - tmp_135 + tmp_67;
                    double tmp_137 = tmp_93 * ( tmp_130 * tmp_58 + tmp_133 * tmp_65 + tmp_136 * tmp_72 );
                    double tmp_138 = tmp_22 * tmp_49;
                    double tmp_139 = tmp_138 * tmp_55;
                    double tmp_140 = tmp_127 - tmp_139 + tmp_95;
                    double tmp_141 = tmp_138 * tmp_63;
                    double tmp_142 = tmp_131 - tmp_141 + tmp_99;
                    double tmp_143 = tmp_138 * tmp_70;
                    double tmp_144 = tmp_102 + tmp_134 - tmp_143;
                    double tmp_145 = tmp_93 * ( tmp_140 * tmp_58 + tmp_142 * tmp_65 + tmp_144 * tmp_72 );
                    double tmp_146 = tmp_11 * tmp_49;
                    double tmp_147 = tmp_146 * tmp_52;
                    double tmp_148 = tmp_107 + tmp_129 - tmp_147;
                    double tmp_149 = tmp_146 * tmp_61;
                    double tmp_150 = tmp_111 + tmp_132 - tmp_149;
                    double tmp_151 = tmp_146 * tmp_68;
                    double tmp_152 = tmp_114 + tmp_135 - tmp_151;
                    double tmp_153 = tmp_93 * ( tmp_148 * tmp_58 + tmp_150 * tmp_65 + tmp_152 * tmp_72 );
                    double tmp_154 = tmp_119 + tmp_139 + tmp_147;
                    double tmp_155 = tmp_121 + tmp_141 + tmp_149;
                    double tmp_156 = tmp_123 + tmp_143 + tmp_151;
                    double tmp_157 = tmp_93 * ( tmp_154 * tmp_58 + tmp_155 * tmp_65 + tmp_156 * tmp_72 );
                    double tmp_158 = tmp_93 * ( tmp_101 * tmp_113 + tmp_104 * tmp_116 + tmp_110 * tmp_98 );
                    double tmp_159 = tmp_93 * ( tmp_101 * tmp_122 + tmp_104 * tmp_124 + tmp_120 * tmp_98 );
                    double tmp_160 = tmp_93 * ( tmp_101 * tmp_133 + tmp_104 * tmp_136 + tmp_130 * tmp_98 );
                    double tmp_161 = tmp_93 * ( tmp_101 * tmp_142 + tmp_104 * tmp_144 + tmp_140 * tmp_98 );
                    double tmp_162 = tmp_93 * ( tmp_101 * tmp_150 + tmp_104 * tmp_152 + tmp_148 * tmp_98 );
                    double tmp_163 = tmp_93 * ( tmp_101 * tmp_155 + tmp_104 * tmp_156 + tmp_154 * tmp_98 );
                    double tmp_164 = tmp_93 * ( tmp_110 * tmp_120 + tmp_113 * tmp_122 + tmp_116 * tmp_124 );
                    double tmp_165 = tmp_93 * ( tmp_110 * tmp_130 + tmp_113 * tmp_133 + tmp_116 * tmp_136 );
                    double tmp_166 = tmp_93 * ( tmp_110 * tmp_140 + tmp_113 * tmp_142 + tmp_116 * tmp_144 );
                    double tmp_167 = tmp_93 * ( tmp_110 * tmp_148 + tmp_113 * tmp_150 + tmp_116 * tmp_152 );
                    double tmp_168 = tmp_93 * ( tmp_110 * tmp_154 + tmp_113 * tmp_155 + tmp_116 * tmp_156 );
                    double tmp_169 = tmp_93 * ( tmp_120 * tmp_130 + tmp_122 * tmp_133 + tmp_124 * tmp_136 );
                    double tmp_170 = tmp_93 * ( tmp_120 * tmp_140 + tmp_122 * tmp_142 + tmp_124 * tmp_144 );
                    double tmp_171 = tmp_93 * ( tmp_120 * tmp_148 + tmp_122 * tmp_150 + tmp_124 * tmp_152 );
                    double tmp_172 = tmp_93 * ( tmp_120 * tmp_154 + tmp_122 * tmp_155 + tmp_124 * tmp_156 );
                    double tmp_173 = tmp_93 * ( tmp_130 * tmp_140 + tmp_133 * tmp_142 + tmp_136 * tmp_144 );
                    double tmp_174 = tmp_93 * ( tmp_130 * tmp_148 + tmp_133 * tmp_150 + tmp_136 * tmp_152 );
                    double tmp_175 = tmp_93 * ( tmp_130 * tmp_154 + tmp_133 * tmp_155 + tmp_136 * tmp_156 );
                    double tmp_176 = tmp_93 * ( tmp_140 * tmp_148 + tmp_142 * tmp_150 + tmp_144 * tmp_152 );
                    double tmp_177 = tmp_93 * ( tmp_140 * tmp_154 + tmp_142 * tmp_155 + tmp_144 * tmp_156 );
                    double tmp_178 = tmp_93 * ( tmp_148 * tmp_154 + tmp_150 * tmp_155 + tmp_152 * tmp_156 );

                    // --- Shape function gradients ---
                    dNdx( 0, 0 ) = tmp_93 * ( pow( tmp_58, 2 ) + pow( tmp_65, 2 ) + pow( tmp_72, 2 ) );
                    dNdx( 0, 1 ) = tmp_105;
                    dNdx( 0, 2 ) = tmp_117;
                    dNdx( 0, 3 ) = tmp_125;
                    dNdx( 0, 4 ) = tmp_137;
                    dNdx( 0, 5 ) = tmp_145;
                    dNdx( 0, 6 ) = tmp_153;
                    dNdx( 0, 7 ) = tmp_157;
                    dNdx( 1, 0 ) = tmp_105;
                    dNdx( 1, 1 ) = tmp_93 * ( pow( tmp_101, 2 ) + pow( tmp_104, 2 ) + pow( tmp_98, 2 ) );
                    dNdx( 1, 2 ) = tmp_158;
                    dNdx( 1, 3 ) = tmp_159;
                    dNdx( 1, 4 ) = tmp_160;
                    dNdx( 1, 5 ) = tmp_161;
                    dNdx( 1, 6 ) = tmp_162;
                    dNdx( 1, 7 ) = tmp_163;
                    dNdx( 2, 0 ) = tmp_117;
                    dNdx( 2, 1 ) = tmp_158;
                    dNdx( 2, 2 ) = tmp_93 * ( pow( tmp_110, 2 ) + pow( tmp_113, 2 ) + pow( tmp_116, 2 ) );
                    dNdx( 2, 3 ) = tmp_164;
                    dNdx( 2, 4 ) = tmp_165;
                    dNdx( 2, 5 ) = tmp_166;
                    dNdx( 2, 6 ) = tmp_167;
                    dNdx( 2, 7 ) = tmp_168;
                    dNdx( 3, 0 ) = tmp_125;
                    dNdx( 3, 1 ) = tmp_159;
                    dNdx( 3, 2 ) = tmp_164;
                    dNdx( 3, 3 ) = tmp_93 * ( pow( tmp_120, 2 ) + pow( tmp_122, 2 ) + pow( tmp_124, 2 ) );
                    dNdx( 3, 4 ) = tmp_169;
                    dNdx( 3, 5 ) = tmp_170;
                    dNdx( 3, 6 ) = tmp_171;
                    dNdx( 3, 7 ) = tmp_172;
                    dNdx( 4, 0 ) = tmp_137;
                    dNdx( 4, 1 ) = tmp_160;
                    dNdx( 4, 2 ) = tmp_165;
                    dNdx( 4, 3 ) = tmp_169;
                    dNdx( 4, 4 ) = tmp_93 * ( pow( tmp_130, 2 ) + pow( tmp_133, 2 ) + pow( tmp_136, 2 ) );
                    dNdx( 4, 5 ) = tmp_173;
                    dNdx( 4, 6 ) = tmp_174;
                    dNdx( 4, 7 ) = tmp_175;
                    dNdx( 5, 0 ) = tmp_145;
                    dNdx( 5, 1 ) = tmp_161;
                    dNdx( 5, 2 ) = tmp_166;
                    dNdx( 5, 3 ) = tmp_170;
                    dNdx( 5, 4 ) = tmp_173;
                    dNdx( 5, 5 ) = tmp_93 * ( pow( tmp_140, 2 ) + pow( tmp_142, 2 ) + pow( tmp_144, 2 ) );
                    dNdx( 5, 6 ) = tmp_176;
                    dNdx( 5, 7 ) = tmp_177;
                    dNdx( 6, 0 ) = tmp_153;
                    dNdx( 6, 1 ) = tmp_162;
                    dNdx( 6, 2 ) = tmp_167;
                    dNdx( 6, 3 ) = tmp_171;
                    dNdx( 6, 4 ) = tmp_174;
                    dNdx( 6, 5 ) = tmp_176;
                    dNdx( 6, 6 ) = tmp_93 * ( pow( tmp_148, 2 ) + pow( tmp_150, 2 ) + pow( tmp_152, 2 ) );
                    dNdx( 6, 7 ) = tmp_178;
                    dNdx( 7, 0 ) = tmp_157;
                    dNdx( 7, 1 ) = tmp_163;
                    dNdx( 7, 2 ) = tmp_168;
                    dNdx( 7, 3 ) = tmp_172;
                    dNdx( 7, 4 ) = tmp_175;
                    dNdx( 7, 5 ) = tmp_177;
                    dNdx( 7, 6 ) = tmp_178;
                    dNdx( 7, 7 ) = tmp_93 * ( pow( tmp_154, 2 ) + pow( tmp_155, 2 ) + pow( tmp_156, 2 ) );

                    A += dNdx;
                }
            }
        }

        if ( treat_boundary_ )
        {
            // Later we will multiply all non-diagonal entries with row or column with that value.
            dense::Mat< double, 8, 8 > boundary_mask;
            boundary_mask.fill( 1.0 );

            for ( int r = r_cell; r <= r_cell + 1; r++ )
            {
                for ( int y = y_cell; y <= y_cell + 1; y++ )
                {
                    for ( int x = x_cell; x <= x_cell + 1; x++ )
                    {
                        const double factor = ( r == 0 || r == radii_.extent( 1 ) - 1 ) ? 0.0 : 1.0;

                        const int x_local = x - x_cell;
                        const int y_local = y - y_cell;
                        const int r_local = r - r_cell;

                        const int diag_entry = 4 * r_local + 2 * y_local + x_local;

                        for ( int i = 0; i < 8; i++ )
                        {
                            if ( i != diag_entry )
                            {
                                boundary_mask( i, diag_entry ) *= factor;
                                boundary_mask( diag_entry, i ) *= factor;
                            }
                        }
                    }
                }
            }

            A.hadamard_product( boundary_mask );
        }

        if ( diagonal_ )
        {
            for ( int i = 0; i < 8; i++ )
            {
                for ( int j = 0; j < 8; j++ )
                {
                    if ( i != j )
                    {
                        A( i, j ) = 0.0;
                    }
                }
            }
        }

        dense::Vec< double, 8 > dst = A * src;

        for ( int r = r_cell; r <= r_cell + 1; r++ )
        {
            for ( int y = y_cell; y <= y_cell + 1; y++ )
            {
                for ( int x = x_cell; x <= x_cell + 1; x++ )
                {
                    const int x_local = x - x_cell;
                    const int y_local = y - y_cell;
                    const int r_local = r - r_cell;

                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x, y, r ), dst( 4 * r_local + 2 * y_local + x_local ) );
                }
            }
        }
    }
};

// x = x + wI * ( b - Ax )
void richardson_step(
    const DistributedDomain&          domain,
    const Grid3DDataVec< double, 3 >& grid,
    const Grid2DDataScalar< double >& radii,
    const Grid4DDataScalar< double >& x,
    const Grid4DDataScalar< double >& b,
    const Grid4DDataScalar< double >& tmp,
    const double                      omega )
{
    // We need that in matvec - maybe resolved when stencils?
    kernels::common::set_constant( tmp, 0.0 );

    Kokkos::parallel_for(
        "matvec",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        LaplaceOperator( grid, radii, x, tmp, true, false ) );

    kernels::common::lincomb( x, 1.0, x, omega, b, -omega, tmp );
}

void single_apply()
{
    const auto domain = grid::shell::DistributedDomain::create_uniform_single_subdomain( 4, 3, 0.5, 1.0 );

    const auto src = grid::shell::allocate_scalar_grid( "src", domain );
    const auto dst = grid::shell::allocate_scalar_grid( "dst", domain );

    communication::SubdomainNeighborhoodSendBuffer send_buffers( domain );
    communication::SubdomainNeighborhoodRecvBuffer recv_buffers( domain );

    std::vector< std::array< int, 11 > > expected_recvs_metadata;
    std::vector< MPI_Request >           expected_recvs_requests;

    const auto subdomain_shell_coords = terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain );
    const auto subdomain_radii        = terra::grid::shell::subdomain_shell_radii( domain );

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( subdomain_shell_coords, subdomain_radii, src, false ) );

    // kernels::common::set_constant( dst, 1.0 );

#if 1
    Kokkos::parallel_for(
        "matvec",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        LaplaceOperator( subdomain_shell_coords, subdomain_radii, src, dst, false, false ) );
#endif

#if 1
    communication::pack_and_send_local_subdomain_boundaries(
        domain, dst, send_buffers, expected_recvs_requests, expected_recvs_metadata );

    MPI_Barrier( MPI_COMM_WORLD );
    Kokkos::fence();

    communication::recv_unpack_and_add_local_subdomain_boundaries(
        domain, dst, recv_buffers, expected_recvs_requests, expected_recvs_metadata );
#endif

    terra::vtk::VTKOutput vtk_after( subdomain_shell_coords, subdomain_radii, "laplace_apply.vtu", false );
    vtk_after.add_scalar_field( src.label(), src );
    vtk_after.add_scalar_field( dst.label(), dst );
    vtk_after.write();
}

#if 1

void all_diamonds()
{
    /**

    Boundary handling notes.

    Using inhom boundary conditions we approach the elimination as follows (for the moment).

    Let A be the "Neumann" operator, i.e., we do not treat the boundaries any differently.

    1. Interpolate Dirichlet boundary conditions into g.
    2. Compute g_A <- A       * g.
    3. Compute g_D <- diag(A) * g.
    4. Set the rhs to b = f - g_A.
    5. Set the rhs at the boundary nodes to g_D.
    6. Solve
            A_elim x = b
       where A_elim is A but with all off-diagonal entries in the same row/col as a boundary node set to zero.
       In a matrix-free context, we have to adapt the element matrix A_local accordingly by (symmetrically ) zeroing
       out all the off-diagonals (row and col) that correspond to a boundary node. But we keep the diagonal intact.
       We still have diag(A) == diag(A_elim).
    7. x is the solution of the original problem. No boundary correction should be necessary.

    **/

    const auto domain = grid::shell::DistributedDomain::create_uniform_single_subdomain( 3, 3, 0.5, 1.0 );

    const auto u        = grid::shell::allocate_scalar_grid( "u", domain );
    const auto g        = grid::shell::allocate_scalar_grid( "g", domain );
    const auto Adiagg   = grid::shell::allocate_scalar_grid( "Adiagg", domain );
    const auto tmp      = grid::shell::allocate_scalar_grid( "tmp", domain );
    const auto solution = grid::shell::allocate_scalar_grid( "solution", domain );
    const auto error    = grid::shell::allocate_scalar_grid( "error", domain );
    const auto b        = grid::shell::allocate_scalar_grid( "b", domain );

    communication::SubdomainNeighborhoodSendBuffer send_buffers( domain );
    communication::SubdomainNeighborhoodRecvBuffer recv_buffers( domain );

    std::vector< std::array< int, 11 > > expected_recvs_metadata;
    std::vector< MPI_Request >           expected_recvs_requests;

    const auto subdomain_shell_coords = terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain );
    const auto subdomain_radii        = terra::grid::shell::subdomain_shell_radii( domain );

    // Set up solution data.
    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( subdomain_shell_coords, subdomain_radii, solution, false ) );

    // Set up boundary data.
    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( subdomain_shell_coords, subdomain_radii, g, true ) );

    Kokkos::parallel_for(
        "matvec",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        LaplaceOperator( subdomain_shell_coords, subdomain_radii, g, Adiagg, false, true ) );

    communication::pack_and_send_local_subdomain_boundaries(
        domain, Adiagg, send_buffers, expected_recvs_requests, expected_recvs_metadata );
    communication::recv_unpack_and_add_local_subdomain_boundaries(
        domain, Adiagg, recv_buffers, expected_recvs_requests, expected_recvs_metadata );

    Kokkos::parallel_for(
        "matvec",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        LaplaceOperator( subdomain_shell_coords, subdomain_radii, g, b, false, false ) );

    communication::pack_and_send_local_subdomain_boundaries(
        domain, b, send_buffers, expected_recvs_requests, expected_recvs_metadata );
    communication::recv_unpack_and_add_local_subdomain_boundaries(
        domain, b, recv_buffers, expected_recvs_requests, expected_recvs_metadata );

    terra::kernels::common::scale( b, -1.0 );

    Kokkos::parallel_for(
        "set on boundary",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        SetOnBoundary( Adiagg, b, domain.domain_info().subdomain_num_nodes_radially() ) );

    // Solve.

    const double omega = 0.3;

    for ( int iter = 0; iter < 1000; iter++ )
    {
        if ( iter % 100 == 0 )
        {
            std::cout << "iter = " << iter;
            const bool comp_norm = true;
            if ( comp_norm )
            {
                kernels::common::lincomb( error, 1.0, u, -1.0, solution );
                const auto error_inf_norm = kernels::common::max_magnitude( error );
                std::cout << ", error inf norm: " << error_inf_norm;
            }
            std::cout << std::endl;
        }

        // We need that in matvec - maybe resolved when stencils?
        kernels::common::set_constant( tmp, 0.0 );

        Kokkos::parallel_for(
            "matvec",
            grid::shell::local_domain_md_range_policy_cells( domain ),
            LaplaceOperator( subdomain_shell_coords, subdomain_radii, u, tmp, true, false ) );

        communication::pack_and_send_local_subdomain_boundaries(
            domain, tmp, send_buffers, expected_recvs_requests, expected_recvs_metadata );
        communication::recv_unpack_and_add_local_subdomain_boundaries(
            domain, tmp, recv_buffers, expected_recvs_requests, expected_recvs_metadata );

        kernels::common::lincomb( u, 1.0, u, omega, b, -omega, tmp );
    }

    kernels::common::lincomb( error, 1.0, u, -1.0, solution );

    terra::vtk::VTKOutput vtk_after( subdomain_shell_coords, subdomain_radii, "laplace.vtu", false );
    vtk_after.add_scalar_field( g.label(), g );
    vtk_after.add_scalar_field( u.label(), u );
    vtk_after.add_scalar_field( solution.label(), solution );
    vtk_after.add_scalar_field( error.label(), error );
    vtk_after.write();
}
#endif

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    {
        // single_apply();
        all_diamonds();
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}