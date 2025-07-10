
#include <optional>

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "terra/dense/mat.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
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
        const double                  value  = coords( 0 ) * Kokkos::sin( coords( 1 ) ) * Kokkos::sinh( coords( 2 ) );
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

struct LaplaceOperator2
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;

    Grid4DDataScalar< double > src_;
    Grid4DDataScalar< double > dst_;
    bool                       treat_boundary_;
    bool                       diagonal_;

    LaplaceOperator2(
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
        dense::Vec< double, 3 > coords[2][2];

        for ( int x = x_cell; x <= x_cell + 1; x++ )
        {
            for ( int y = y_cell; y <= y_cell + 1; y++ )
            {
                coords[x - x_cell][y - y_cell]( 0 ) = grid_( local_subdomain_id, x, y, 0 );
                coords[x - x_cell][y - y_cell]( 1 ) = grid_( local_subdomain_id, x, y, 1 );
                coords[x - x_cell][y - y_cell]( 2 ) = grid_( local_subdomain_id, x, y, 2 );
            }
        }

        const double x_unit_1 = coords[0][0]( 0 );
        const double x_unit_2 = coords[1][0]( 0 );
        const double x_unit_3 = coords[0][1]( 0 );
        const double x_unit_4 = coords[1][1]( 0 );

        const double y_unit_1 = coords[0][0]( 1 );
        const double y_unit_2 = coords[1][0]( 1 );
        const double y_unit_3 = coords[0][1]( 1 );
        const double y_unit_4 = coords[1][1]( 1 );

        const double z_unit_1 = coords[0][0]( 2 );
        const double z_unit_2 = coords[1][0]( 2 );
        const double z_unit_3 = coords[0][1]( 2 );
        const double z_unit_4 = coords[1][1]( 2 );

        // Gauss-Lobatto quadrature
#if 0
        constexpr int nq    = 3;
        const double  q[nq] = { -1.0, 0.0, 1.0 };
        const double  w[nq] = { 1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0 };
#else
        // Gauss-Legendre quadrature
        constexpr int nq    = 2;
        const double  q[nq] = { -0.57735026919, 0.57735026919 };
        const double  w[nq] = { 1.0, 1.0 };
#endif

        dense::Vec< double, 8 > src;

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

        // Later we will multiply all non-diagonal entries with row or column with that value.
        dense::Mat< double, 8, 8 > boundary_mask;
        boundary_mask.fill( 1.0 );

        if ( treat_boundary_ )
        {
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
        }

        if ( diagonal_ )
        {
            for ( int i = 0; i < 8; i++ )
            {
                for ( int j = 0; j < 8; j++ )
                {
                    if ( i != j )
                    {
                        boundary_mask( i, j ) = 0.0;
                    }
                }
            }
        }

        const auto src_0 = src( 0 );
        const auto src_1 = src( 1 );
        const auto src_2 = src( 2 );
        const auto src_3 = src( 3 );
        const auto src_4 = src( 4 );
        const auto src_5 = src( 5 );
        const auto src_6 = src( 6 );
        const auto src_7 = src( 7 );

        const auto r_inner = radii_( local_subdomain_id, r_cell );
        const auto r_outer = radii_( local_subdomain_id, r_cell + 1 );

        dense::Vec< double, 8 > dst;

        for ( int qx = 0; qx < nq; qx++ )
        {
            for ( int qy = 0; qy < nq; qy++ )
            {
                for ( int qr = 0; qr < nq; qr++ )
                {
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
                    double tmp_6  = ( 1.0 / 8.0 ) * r_inner;
                    double tmp_7  = tmp_1 * tmp_6;
                    double tmp_8  = tmp_3 * tmp_7;
                    double tmp_9  = tmp_4 * tmp_6;
                    double tmp_10 = tmp_3 * tmp_9;
                    double tmp_11 = ( 1.0 / 8.0 ) * r_outer;
                    double tmp_12 = tmp_1 * tmp_11 * tmp_5;
                    double tmp_13 = tmp_11 * tmp_4;
                    double tmp_14 = tmp_13 * tmp_5;
                    double tmp_15 = tmp_14 * x_unit_3 - tmp_14 * x_unit_4;
                    double tmp_16 = ( 1.0 / 8.0 ) * r_inner * tmp_1 * tmp_3 * x_unit_2 +
                                    ( 1.0 / 8.0 ) * r_inner * tmp_3 * tmp_4 * x_unit_4 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_1 * tmp_5 * x_unit_2 - tmp_10 * x_unit_3 -
                                    tmp_12 * x_unit_1 - tmp_15 - tmp_8 * x_unit_1;
                    double tmp_17 = qp0 - 1;
                    double tmp_18 = -tmp_17;
                    double tmp_19 = qp0 + 1;
                    double tmp_20 = tmp_3 * tmp_6;
                    double tmp_21 = tmp_18 * y_unit_1;
                    double tmp_22 = tmp_19 * y_unit_2;
                    double tmp_23 = tmp_11 * tmp_5;
                    double tmp_24 = tmp_19 * y_unit_4;
                    double tmp_25 = tmp_22 * tmp_23 - tmp_23 * tmp_24;
                    double tmp_26 = ( 1.0 / 8.0 ) * r_inner * tmp_18 * tmp_3 * y_unit_3 +
                                    ( 1.0 / 8.0 ) * r_inner * tmp_19 * tmp_3 * y_unit_4 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_18 * tmp_5 * y_unit_3 - tmp_20 * tmp_21 -
                                    tmp_20 * tmp_22 - tmp_21 * tmp_23 - tmp_25;
                    double tmp_27 = tmp_16 * tmp_26;
                    double tmp_28 = tmp_18 * x_unit_1;
                    double tmp_29 = tmp_19 * x_unit_2;
                    double tmp_30 = tmp_19 * x_unit_4;
                    double tmp_31 = tmp_23 * tmp_29 - tmp_23 * tmp_30;
                    double tmp_32 = ( 1.0 / 8.0 ) * r_inner * tmp_18 * tmp_3 * x_unit_3 +
                                    ( 1.0 / 8.0 ) * r_inner * tmp_19 * tmp_3 * x_unit_4 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_18 * tmp_5 * x_unit_3 - tmp_20 * tmp_28 -
                                    tmp_20 * tmp_29 - tmp_23 * tmp_28 - tmp_31;
                    double tmp_33 = tmp_14 * y_unit_3 - tmp_14 * y_unit_4;
                    double tmp_34 = ( 1.0 / 8.0 ) * r_inner * tmp_1 * tmp_3 * y_unit_2 +
                                    ( 1.0 / 8.0 ) * r_inner * tmp_3 * tmp_4 * y_unit_4 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_1 * tmp_5 * y_unit_2 - tmp_10 * y_unit_3 -
                                    tmp_12 * y_unit_1 - tmp_33 - tmp_8 * y_unit_1;
                    double tmp_35 = tmp_32 * tmp_34;
                    double tmp_36 = tmp_27 - tmp_35;
                    double tmp_37 = tmp_18 * z_unit_1;
                    double tmp_38 = tmp_19 * z_unit_2;
                    double tmp_39 = tmp_19 * z_unit_4;
                    double tmp_40 = tmp_23 * tmp_38 - tmp_23 * tmp_39;
                    double tmp_41 = ( 1.0 / 8.0 ) * r_inner * tmp_18 * tmp_3 * z_unit_3 +
                                    ( 1.0 / 8.0 ) * r_inner * tmp_19 * tmp_3 * z_unit_4 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_18 * tmp_5 * z_unit_3 - tmp_20 * tmp_37 -
                                    tmp_20 * tmp_38 - tmp_23 * tmp_37 - tmp_40;
                    double tmp_42 = -tmp_13 * tmp_30 + tmp_30 * tmp_9;
                    double tmp_43 = ( 1.0 / 8.0 ) * r_outer * tmp_1 * tmp_18 * x_unit_1 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_1 * tmp_19 * x_unit_2 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_18 * tmp_4 * x_unit_3 - tmp_18 * tmp_9 * x_unit_3 -
                                    tmp_28 * tmp_7 - tmp_29 * tmp_7 - tmp_42;
                    double tmp_44 = tmp_34 * tmp_43;
                    double tmp_45 = -tmp_13 * tmp_39 + tmp_39 * tmp_9;
                    double tmp_46 = ( 1.0 / 8.0 ) * r_outer * tmp_1 * tmp_18 * z_unit_1 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_1 * tmp_19 * z_unit_2 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_18 * tmp_4 * z_unit_3 - tmp_18 * tmp_9 * z_unit_3 -
                                    tmp_37 * tmp_7 - tmp_38 * tmp_7 - tmp_45;
                    double tmp_47 = tmp_14 * z_unit_3 - tmp_14 * z_unit_4;
                    double tmp_48 = ( 1.0 / 8.0 ) * r_inner * tmp_1 * tmp_3 * z_unit_2 +
                                    ( 1.0 / 8.0 ) * r_inner * tmp_3 * tmp_4 * z_unit_4 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_1 * tmp_5 * z_unit_2 - tmp_10 * z_unit_3 -
                                    tmp_12 * z_unit_1 - tmp_47 - tmp_8 * z_unit_1;
                    double tmp_49 = -tmp_13 * tmp_24 + tmp_24 * tmp_9;
                    double tmp_50 = ( 1.0 / 8.0 ) * r_outer * tmp_1 * tmp_18 * y_unit_1 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_1 * tmp_19 * y_unit_2 +
                                    ( 1.0 / 8.0 ) * r_outer * tmp_18 * tmp_4 * y_unit_3 - tmp_18 * tmp_9 * y_unit_3 -
                                    tmp_21 * tmp_7 - tmp_22 * tmp_7 - tmp_49;
                    double tmp_51 = tmp_26 * tmp_43;
                    double tmp_52 = tmp_16 * tmp_50;
                    double tmp_53 = ( 1.0 / 8.0 ) / ( tmp_27 * tmp_46 + tmp_32 * tmp_48 * tmp_50 - tmp_35 * tmp_46 +
                                                      tmp_41 * tmp_44 - tmp_41 * tmp_52 - tmp_48 * tmp_51 );
                    double tmp_54 = tmp_1 * tmp_53;
                    double tmp_55 = tmp_18 * tmp_54;
                    double tmp_56 = tmp_36 * tmp_55;
                    double tmp_57 = tmp_32 * tmp_50 - tmp_51;
                    double tmp_58 = tmp_3 * tmp_54;
                    double tmp_59 = tmp_57 * tmp_58;
                    double tmp_60 = tmp_44 - tmp_52;
                    double tmp_61 = tmp_3 * tmp_53;
                    double tmp_62 = tmp_18 * tmp_61;
                    double tmp_63 = tmp_60 * tmp_62;
                    double tmp_64 = -tmp_56 - tmp_59 - tmp_63;
                    double tmp_65 = -tmp_16 * tmp_41 + tmp_32 * tmp_48;
                    double tmp_66 = tmp_55 * tmp_65;
                    double tmp_67 = -tmp_32 * tmp_46 + tmp_41 * tmp_43;
                    double tmp_68 = tmp_58 * tmp_67;
                    double tmp_69 = tmp_16 * tmp_46 - tmp_43 * tmp_48;
                    double tmp_70 = tmp_62 * tmp_69;
                    double tmp_71 = -tmp_66 - tmp_68 - tmp_70;
                    double tmp_72 = -tmp_26 * tmp_48 + tmp_34 * tmp_41;
                    double tmp_73 = tmp_55 * tmp_72;
                    double tmp_74 = tmp_26 * tmp_46 - tmp_41 * tmp_50;
                    double tmp_75 = tmp_58 * tmp_74;
                    double tmp_76 = -tmp_34 * tmp_46 + tmp_48 * tmp_50;
                    double tmp_77 = tmp_62 * tmp_76;
                    double tmp_78 = -tmp_73 - tmp_75 - tmp_77;
                    double tmp_79 = tmp_0 * tmp_6;
                    double tmp_80 = tmp_17 * x_unit_3;
                    double tmp_81 = tmp_17 * x_unit_1;
                    double tmp_82 = tmp_0 * tmp_11;
                    double tmp_83 = tmp_13 * tmp_80 - tmp_29 * tmp_79 + tmp_29 * tmp_82 + tmp_42 + tmp_79 * tmp_81 -
                                    tmp_80 * tmp_9 - tmp_81 * tmp_82;
                    double tmp_84 = tmp_2 * tmp_79;
                    double tmp_85 = tmp_2 * tmp_9;
                    double tmp_86 = tmp_0 * tmp_23;
                    double tmp_87 = tmp_33 + tmp_84 * y_unit_1 - tmp_84 * y_unit_2 - tmp_85 * y_unit_3 +
                                    tmp_85 * y_unit_4 - tmp_86 * y_unit_1 + tmp_86 * y_unit_2;
                    double tmp_88 = tmp_2 * tmp_6;
                    double tmp_89 = tmp_17 * tmp_88;
                    double tmp_90 = tmp_17 * tmp_23;
                    double tmp_91 = -tmp_38 * tmp_88 + tmp_39 * tmp_88 + tmp_40 + tmp_89 * z_unit_1 -
                                    tmp_89 * z_unit_3 - tmp_90 * z_unit_1 + tmp_90 * z_unit_3;
                    double tmp_92 = tmp_15 + tmp_84 * x_unit_1 - tmp_84 * x_unit_2 - tmp_85 * x_unit_3 +
                                    tmp_85 * x_unit_4 - tmp_86 * x_unit_1 + tmp_86 * x_unit_2;
                    double tmp_93 = -tmp_22 * tmp_88 + tmp_24 * tmp_88 + tmp_25 + tmp_89 * y_unit_1 -
                                    tmp_89 * y_unit_3 - tmp_90 * y_unit_1 + tmp_90 * y_unit_3;
                    double tmp_94 = tmp_17 * z_unit_3;
                    double tmp_95 = tmp_17 * z_unit_1;
                    double tmp_96 = tmp_13 * tmp_94 - tmp_38 * tmp_79 + tmp_38 * tmp_82 + tmp_45 + tmp_79 * tmp_95 -
                                    tmp_82 * tmp_95 - tmp_9 * tmp_94;
                    double tmp_97 = tmp_23 * tmp_80 - tmp_23 * tmp_81 - tmp_29 * tmp_88 + tmp_30 * tmp_88 + tmp_31 -
                                    tmp_80 * tmp_88 + tmp_81 * tmp_88;
                    double tmp_98  = tmp_17 * y_unit_3;
                    double tmp_99  = tmp_17 * y_unit_1;
                    double tmp_100 = tmp_13 * tmp_98 - tmp_22 * tmp_79 + tmp_22 * tmp_82 + tmp_49 + tmp_79 * tmp_99 -
                                     tmp_82 * tmp_99 - tmp_9 * tmp_98;
                    double tmp_101 = tmp_47 + tmp_84 * z_unit_1 - tmp_84 * z_unit_2 - tmp_85 * z_unit_3 +
                                     tmp_85 * z_unit_4 - tmp_86 * z_unit_1 + tmp_86 * z_unit_2;
                    double tmp_102 =
                        www * fabs(
                                  tmp_100 * tmp_101 * tmp_97 - tmp_100 * tmp_91 * tmp_92 - tmp_101 * tmp_83 * tmp_93 +
                                  tmp_83 * tmp_87 * tmp_91 - tmp_87 * tmp_96 * tmp_97 + tmp_92 * tmp_93 * tmp_96 );
                    double tmp_103 = src_0 * tmp_102;
                    double tmp_104 = src_7 * tmp_102;
                    double tmp_105 = tmp_4 * tmp_53;
                    double tmp_106 = tmp_105 * tmp_19;
                    double tmp_107 = tmp_106 * tmp_36;
                    double tmp_108 = tmp_105 * tmp_5;
                    double tmp_109 = tmp_108 * tmp_57;
                    double tmp_110 = tmp_5 * tmp_53;
                    double tmp_111 = tmp_110 * tmp_19;
                    double tmp_112 = tmp_111 * tmp_60;
                    double tmp_113 = tmp_107 + tmp_109 + tmp_112;
                    double tmp_114 = tmp_106 * tmp_65;
                    double tmp_115 = tmp_108 * tmp_67;
                    double tmp_116 = tmp_111 * tmp_69;
                    double tmp_117 = tmp_114 + tmp_115 + tmp_116;
                    double tmp_118 = tmp_106 * tmp_72;
                    double tmp_119 = tmp_108 * tmp_74;
                    double tmp_120 = tmp_111 * tmp_76;
                    double tmp_121 = tmp_118 + tmp_119 + tmp_120;
                    double tmp_122 = boundary_mask( 0, 7 ) * ( tmp_113 * tmp_64 + tmp_117 * tmp_71 + tmp_121 * tmp_78 );
                    double tmp_123 = src_3 * tmp_102;
                    double tmp_124 = tmp_4 * tmp_61;
                    double tmp_125 = tmp_124 * tmp_57;
                    double tmp_126 = tmp_19 * tmp_61;
                    double tmp_127 = tmp_126 * tmp_60;
                    double tmp_128 = -tmp_107 + tmp_125 + tmp_127;
                    double tmp_129 = tmp_124 * tmp_67;
                    double tmp_130 = tmp_126 * tmp_69;
                    double tmp_131 = -tmp_114 + tmp_129 + tmp_130;
                    double tmp_132 = tmp_124 * tmp_74;
                    double tmp_133 = tmp_126 * tmp_76;
                    double tmp_134 = -tmp_118 + tmp_132 + tmp_133;
                    double tmp_135 = boundary_mask( 0, 3 ) * ( tmp_128 * tmp_64 + tmp_131 * tmp_71 + tmp_134 * tmp_78 );
                    double tmp_136 = src_5 * tmp_102;
                    double tmp_137 = tmp_19 * tmp_54;
                    double tmp_138 = tmp_137 * tmp_36;
                    double tmp_139 = tmp_5 * tmp_54;
                    double tmp_140 = tmp_139 * tmp_57;
                    double tmp_141 = -tmp_112 + tmp_138 + tmp_140;
                    double tmp_142 = tmp_137 * tmp_65;
                    double tmp_143 = tmp_139 * tmp_67;
                    double tmp_144 = -tmp_116 + tmp_142 + tmp_143;
                    double tmp_145 = tmp_137 * tmp_72;
                    double tmp_146 = tmp_139 * tmp_74;
                    double tmp_147 = -tmp_120 + tmp_145 + tmp_146;
                    double tmp_148 = boundary_mask( 0, 5 ) * ( tmp_141 * tmp_64 + tmp_144 * tmp_71 + tmp_147 * tmp_78 );
                    double tmp_149 = src_6 * tmp_102;
                    double tmp_150 = tmp_105 * tmp_18;
                    double tmp_151 = tmp_150 * tmp_36;
                    double tmp_152 = tmp_110 * tmp_18;
                    double tmp_153 = tmp_152 * tmp_60;
                    double tmp_154 = -tmp_109 + tmp_151 + tmp_153;
                    double tmp_155 = tmp_150 * tmp_65;
                    double tmp_156 = tmp_152 * tmp_69;
                    double tmp_157 = -tmp_115 + tmp_155 + tmp_156;
                    double tmp_158 = tmp_150 * tmp_72;
                    double tmp_159 = tmp_152 * tmp_76;
                    double tmp_160 = -tmp_119 + tmp_158 + tmp_159;
                    double tmp_161 = boundary_mask( 0, 6 ) * ( tmp_154 * tmp_64 + tmp_157 * tmp_71 + tmp_160 * tmp_78 );
                    double tmp_162 = -tmp_127 - tmp_138 + tmp_59;
                    double tmp_163 = -tmp_130 - tmp_142 + tmp_68;
                    double tmp_164 = -tmp_133 - tmp_145 + tmp_75;
                    double tmp_165 = boundary_mask( 0, 1 ) * ( tmp_162 * tmp_64 + tmp_163 * tmp_71 + tmp_164 * tmp_78 );
                    double tmp_166 = src_1 * tmp_102;
                    double tmp_167 = src_2 * tmp_102;
                    double tmp_168 = -tmp_125 - tmp_151 + tmp_63;
                    double tmp_169 = -tmp_129 - tmp_155 + tmp_70;
                    double tmp_170 = -tmp_132 - tmp_158 + tmp_77;
                    double tmp_171 = boundary_mask( 0, 2 ) * ( tmp_168 * tmp_64 + tmp_169 * tmp_71 + tmp_170 * tmp_78 );
                    double tmp_172 = src_4 * tmp_102;
                    double tmp_173 = -tmp_140 - tmp_153 + tmp_56;
                    double tmp_174 = -tmp_143 - tmp_156 + tmp_66;
                    double tmp_175 = -tmp_146 - tmp_159 + tmp_73;
                    double tmp_176 = boundary_mask( 0, 4 ) * ( tmp_173 * tmp_64 + tmp_174 * tmp_71 + tmp_175 * tmp_78 );
                    double tmp_177 =
                        boundary_mask( 1, 7 ) * ( tmp_113 * tmp_162 + tmp_117 * tmp_163 + tmp_121 * tmp_164 );
                    double tmp_178 =
                        boundary_mask( 1, 3 ) * ( tmp_128 * tmp_162 + tmp_131 * tmp_163 + tmp_134 * tmp_164 );
                    double tmp_179 =
                        boundary_mask( 1, 5 ) * ( tmp_141 * tmp_162 + tmp_144 * tmp_163 + tmp_147 * tmp_164 );
                    double tmp_180 =
                        boundary_mask( 1, 6 ) * ( tmp_154 * tmp_162 + tmp_157 * tmp_163 + tmp_160 * tmp_164 );
                    double tmp_181 =
                        boundary_mask( 1, 2 ) * ( tmp_162 * tmp_168 + tmp_163 * tmp_169 + tmp_164 * tmp_170 );
                    double tmp_182 =
                        boundary_mask( 1, 4 ) * ( tmp_162 * tmp_173 + tmp_163 * tmp_174 + tmp_164 * tmp_175 );
                    double tmp_183 =
                        boundary_mask( 2, 7 ) * ( tmp_113 * tmp_168 + tmp_117 * tmp_169 + tmp_121 * tmp_170 );
                    double tmp_184 =
                        boundary_mask( 2, 3 ) * ( tmp_128 * tmp_168 + tmp_131 * tmp_169 + tmp_134 * tmp_170 );
                    double tmp_185 =
                        boundary_mask( 2, 5 ) * ( tmp_141 * tmp_168 + tmp_144 * tmp_169 + tmp_147 * tmp_170 );
                    double tmp_186 =
                        boundary_mask( 2, 6 ) * ( tmp_154 * tmp_168 + tmp_157 * tmp_169 + tmp_160 * tmp_170 );
                    double tmp_187 =
                        boundary_mask( 2, 4 ) * ( tmp_168 * tmp_173 + tmp_169 * tmp_174 + tmp_170 * tmp_175 );
                    double tmp_188 =
                        boundary_mask( 3, 7 ) * ( tmp_113 * tmp_128 + tmp_117 * tmp_131 + tmp_121 * tmp_134 );
                    double tmp_189 =
                        boundary_mask( 3, 5 ) * ( tmp_128 * tmp_141 + tmp_131 * tmp_144 + tmp_134 * tmp_147 );
                    double tmp_190 =
                        boundary_mask( 3, 6 ) * ( tmp_128 * tmp_154 + tmp_131 * tmp_157 + tmp_134 * tmp_160 );
                    double tmp_191 =
                        boundary_mask( 3, 4 ) * ( tmp_128 * tmp_173 + tmp_131 * tmp_174 + tmp_134 * tmp_175 );
                    double tmp_192 =
                        boundary_mask( 4, 7 ) * ( tmp_113 * tmp_173 + tmp_117 * tmp_174 + tmp_121 * tmp_175 );
                    double tmp_193 =
                        boundary_mask( 4, 5 ) * ( tmp_141 * tmp_173 + tmp_144 * tmp_174 + tmp_147 * tmp_175 );
                    double tmp_194 =
                        boundary_mask( 4, 6 ) * ( tmp_154 * tmp_173 + tmp_157 * tmp_174 + tmp_160 * tmp_175 );
                    double tmp_195 =
                        boundary_mask( 5, 7 ) * ( tmp_113 * tmp_141 + tmp_117 * tmp_144 + tmp_121 * tmp_147 );
                    double tmp_196 =
                        boundary_mask( 5, 6 ) * ( tmp_141 * tmp_154 + tmp_144 * tmp_157 + tmp_147 * tmp_160 );
                    double tmp_197 =
                        boundary_mask( 6, 7 ) * ( tmp_113 * tmp_154 + tmp_117 * tmp_157 + tmp_121 * tmp_160 );
                    dst( 0 ) = boundary_mask( 0, 0 ) * tmp_103 *
                                   ( ( tmp_64 * tmp_64 ) + ( tmp_71 * tmp_71 ) + ( tmp_78 * tmp_78 ) ) +
                               tmp_104 * tmp_122 + tmp_123 * tmp_135 + tmp_136 * tmp_148 + tmp_149 * tmp_161 +
                               tmp_165 * tmp_166 + tmp_167 * tmp_171 + tmp_172 * tmp_176;
                    dst( 1 ) = boundary_mask( 1, 1 ) * tmp_166 *
                                   ( ( tmp_162 * tmp_162 ) + ( tmp_163 * tmp_163 ) + ( tmp_164 * tmp_164 ) ) +
                               tmp_103 * tmp_165 + tmp_104 * tmp_177 + tmp_123 * tmp_178 + tmp_136 * tmp_179 +
                               tmp_149 * tmp_180 + tmp_167 * tmp_181 + tmp_172 * tmp_182;
                    dst( 2 ) = boundary_mask( 2, 2 ) * tmp_167 *
                                   ( ( tmp_168 * tmp_168 ) + ( tmp_169 * tmp_169 ) + ( tmp_170 * tmp_170 ) ) +
                               tmp_103 * tmp_171 + tmp_104 * tmp_183 + tmp_123 * tmp_184 + tmp_136 * tmp_185 +
                               tmp_149 * tmp_186 + tmp_166 * tmp_181 + tmp_172 * tmp_187;
                    dst( 3 ) = boundary_mask( 3, 3 ) * tmp_123 *
                                   ( ( tmp_128 * tmp_128 ) + ( tmp_131 * tmp_131 ) + ( tmp_134 * tmp_134 ) ) +
                               tmp_103 * tmp_135 + tmp_104 * tmp_188 + tmp_136 * tmp_189 + tmp_149 * tmp_190 +
                               tmp_166 * tmp_178 + tmp_167 * tmp_184 + tmp_172 * tmp_191;
                    dst( 4 ) = boundary_mask( 4, 4 ) * tmp_172 *
                                   ( ( tmp_173 * tmp_173 ) + ( tmp_174 * tmp_174 ) + ( tmp_175 * tmp_175 ) ) +
                               tmp_103 * tmp_176 + tmp_104 * tmp_192 + tmp_123 * tmp_191 + tmp_136 * tmp_193 +
                               tmp_149 * tmp_194 + tmp_166 * tmp_182 + tmp_167 * tmp_187;
                    dst( 5 ) = boundary_mask( 5, 5 ) * tmp_136 *
                                   ( ( tmp_141 * tmp_141 ) + ( tmp_144 * tmp_144 ) + ( tmp_147 * tmp_147 ) ) +
                               tmp_103 * tmp_148 + tmp_104 * tmp_195 + tmp_123 * tmp_189 + tmp_149 * tmp_196 +
                               tmp_166 * tmp_179 + tmp_167 * tmp_185 + tmp_172 * tmp_193;
                    dst( 6 ) = boundary_mask( 6, 6 ) * tmp_149 *
                                   ( ( tmp_154 * tmp_154 ) + ( tmp_157 * tmp_157 ) + ( tmp_160 * tmp_160 ) ) +
                               tmp_103 * tmp_161 + tmp_104 * tmp_197 + tmp_123 * tmp_190 + tmp_136 * tmp_196 +
                               tmp_166 * tmp_180 + tmp_167 * tmp_186 + tmp_172 * tmp_194;
                    dst( 7 ) = boundary_mask( 7, 7 ) * tmp_104 *
                                   ( ( tmp_113 * tmp_113 ) + ( tmp_117 * tmp_117 ) + ( tmp_121 * tmp_121 ) ) +
                               tmp_103 * tmp_122 + tmp_123 * tmp_188 + tmp_136 * tmp_195 + tmp_149 * tmp_197 +
                               tmp_166 * tmp_177 + tmp_167 * tmp_183 + tmp_172 * tmp_192;

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
            }
        }
    }
};

struct LaplaceOperatorWedge
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;

    Grid4DDataScalar< double > src_;
    Grid4DDataScalar< double > dst_;
    bool                       treat_boundary_;
    bool                       diagonal_;

    LaplaceOperatorWedge(
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
        // First all the r-independent stuff.
        // Gather surface points for each wedge.
        constexpr int num_wedges = 2;

        // Extract vertex positions of quad
        // (0, 0), (1, 0), (0, 1), (1, 1).
        dense::Vec< double, 3 > quad_surface_coords[2][2];

        for ( int x = x_cell; x <= x_cell + 1; x++ )
        {
            for ( int y = y_cell; y <= y_cell + 1; y++ )
            {
                for ( int d = 0; d < 3; d++ )
                {
                    quad_surface_coords[x - x_cell][y - y_cell]( d ) = grid_( local_subdomain_id, x, y, d );
                }
            }
        }

        // Sort coords for the two wedge surfaces.
        dense::Vec< double, 3 > wedge_phy_surf[num_wedges][3] = {};

        wedge_phy_surf[0][0] = quad_surface_coords[0][0];
        wedge_phy_surf[0][1] = quad_surface_coords[1][0];
        wedge_phy_surf[0][2] = quad_surface_coords[0][1];

        wedge_phy_surf[1][0] = quad_surface_coords[1][1];
        wedge_phy_surf[1][1] = quad_surface_coords[0][1];
        wedge_phy_surf[1][2] = quad_surface_coords[1][0];

        // Compute lateral part of Jacobian.
        // For now, we only do this at a single quad point.

#if 0
        constexpr int    nq              = 2;
        constexpr double one_over_sqrt_3 = 0.57735026918962576450914878050195745564760175127012687601860232648397767230;
        constexpr dense::Vec< double, 3 > qp[nq] = {
            { 1.0 / 3.0, 1.0 / 3.0, -one_over_sqrt_3 }, { 1.0 / 3.0, 1.0 / 3.0, one_over_sqrt_3 } };
        constexpr double qw[nq] = { 1.0, 1.0 };
#endif

#if 1
        constexpr int                     nq     = 1;
        constexpr dense::Vec< double, 3 > qp[nq] = { { 1.0 / 3.0, 1.0 / 3.0, 0.0 } };
        constexpr double                  qw[nq] = { 1.0 };
#endif

#if 0

        constexpr int                     nq     = 6;
        constexpr dense::Vec< double, 3 > qp[nq] = {
            { { 0.6666666666666666, 0.1666666666666667, -0.5773502691896257 } },
            { { 0.1666666666666667, 0.6666666666666666, -0.5773502691896257 } },
            { { 0.1666666666666667, 0.1666666666666667, -0.5773502691896257 } },
            { { 0.6666666666666666, 0.1666666666666667, 0.5773502691896257 } },
            { { 0.1666666666666667, 0.6666666666666666, 0.5773502691896257 } },
            { { 0.1666666666666667, 0.1666666666666667, 0.5773502691896257 } } };
        constexpr double qw[nq] = {
            0.1666666666666667,
            0.1666666666666667,
            0.1666666666666667,
            0.1666666666666667,
            0.1666666666666667,
            0.1666666666666667 };
#endif

        dense::Mat< double, 3, 3 > jac_lat_inv_t[num_wedges][nq] = {};
        double                     det_jac_lat[num_wedges][nq]   = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int q = 0; q < nq; q++ )
            {
                const auto jac_lat = fe::wedge::jac_lat(
                    wedge_phy_surf[wedge][0],
                    wedge_phy_surf[wedge][1],
                    wedge_phy_surf[wedge][2],
                    qp[q]( 0 ),
                    qp[q]( 1 ) );

                det_jac_lat[wedge][q] = Kokkos::abs( jac_lat.det() );

                jac_lat_inv_t[wedge][q] = jac_lat.inv().transposed();
            }
        }

        constexpr int num_nodes_per_wedge = 6;

        // Let's now gather all the shape functions and gradients we need.
        double shape_lat[num_wedges][num_nodes_per_wedge][nq] = {};
        double shape_rad[num_wedges][num_nodes_per_wedge][nq] = {};

        double grad_shape_lat_xi[num_wedges][num_nodes_per_wedge]  = {};
        double grad_shape_lat_eta[num_wedges][num_nodes_per_wedge] = {};
        double grad_shape_rad[num_wedges][num_nodes_per_wedge]     = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                for ( int q = 0; q < nq; q++ )
                {
                    shape_lat[wedge][node_idx][q] = fe::wedge::shape_lat( qp[q]( 0 ), qp[q]( 1 ) )( node_idx % 3 );
                    shape_rad[wedge][node_idx][q] = fe::wedge::shape_rad( qp[q]( 2 ) )( node_idx / 3 );
                }

                grad_shape_lat_xi[wedge][node_idx]  = fe::wedge::grad_shape_lat_xi()( node_idx % 3 );
                grad_shape_lat_eta[wedge][node_idx] = fe::wedge::grad_shape_lat_eta()( node_idx % 3 );
                grad_shape_rad[wedge][node_idx]     = fe::wedge::grad_shape_rad()( node_idx / 3 );
            }
        }

        dense::Vec< double, 3 > g_rad[num_wedges][num_nodes_per_wedge][nq] = {};
        dense::Vec< double, 3 > g_lat[num_wedges][num_nodes_per_wedge][nq] = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                for ( int q = 0; q < nq; q++ )
                {
                    g_rad[wedge][node_idx][q] = jac_lat_inv_t[wedge][q] *
                                                dense::Vec< double, 3 >{
                                                    grad_shape_lat_xi[wedge][node_idx] * shape_rad[wedge][node_idx][q],
                                                    grad_shape_lat_eta[wedge][node_idx] * shape_rad[wedge][node_idx][q],
                                                    0.0 };

                    g_lat[wedge][node_idx][q] =
                        jac_lat_inv_t[wedge][q] * dense::Vec< double, 3 >{ 0.0, 0.0, shape_lat[wedge][node_idx][q] };
                }
            }
        }

        // Only now we introduce radially dependent terms.
        const double r_1 = radii_( local_subdomain_id, r_cell );
        const double r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // For now, compute the local element matrix. We'll improve that later.
        dense::Mat< double, 6, 6 > A[num_wedges] = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int q = 0; q < nq; q++ )
            {
                const double r = fe::wedge::forward_map_rad( r_1, r_2, qp[q]( 2 ) );
                // TODO: we can precompute that per quadrature point to avoid the division.
                const double r_inv = 1.0 / r;

                const double grad_r = fe::wedge::grad_forward_map_rad( r_1, r_2 );
                // TODO: we can precompute that per quadrature point to avoid the division.
                const double grad_r_inv = 1.0 / grad_r;

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const dense::Vec< double, 3 > grad_i =
                            r_inv * g_rad[wedge][i][q] + grad_shape_rad[wedge][i] * grad_r_inv * g_lat[wedge][i][q];
                        const dense::Vec< double, 3 > grad_j =
                            r_inv * g_rad[wedge][j][q] + grad_shape_rad[wedge][j] * grad_r_inv * g_lat[wedge][j][q];

                        A[wedge]( i, j ) += qw[q] * ( grad_i.dot( grad_j ) * r * r * grad_r * det_jac_lat[wedge][q] );
                    }
                }
            }
        }

        if ( treat_boundary_ )
        {
            for ( int wedge = 0; wedge < num_wedges; wedge++ )
            {
                dense::Mat< double, 6, 6 > boundary_mask;
                boundary_mask.fill( 1.0 );
                if ( r_cell == 0 )
                {
                    // Inner boundary (CMB).
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j && ( i < 3 || j < 3 ) )
                            {
                                boundary_mask( i, j ) = 0.0;
                            }
                        }
                    }
                }

                if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
                {
                    // Outer boundary (surface).
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j && ( i >= 3 || j >= 3 ) )
                            {
                                boundary_mask( i, j ) = 0.0;
                            }
                        }
                    }
                }

                A[wedge].hadamard_product( boundary_mask );
            }
        }

        if ( diagonal_ )
        {
            for ( int wedge = 0; wedge < num_wedges; wedge++ )
            {
                for ( int i = 0; i < 6; i++ )
                {
                    for ( int j = 0; j < 6; j++ )
                    {
                        if ( i != j )
                        {
                            A[wedge]( i, j ) = 0.0;
                        }
                    }
                }
            }
        }

        dense::Vec< double, 6 > src[num_wedges];

        src[0]( 0 ) = src_( local_subdomain_id, x_cell, y_cell, r_cell );
        src[0]( 1 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
        src[0]( 2 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
        src[0]( 3 ) = src_( local_subdomain_id, x_cell, y_cell, r_cell + 1 );
        src[0]( 4 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
        src[0]( 5 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );

        src[1]( 0 ) = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell );
        src[1]( 1 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
        src[1]( 2 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
        src[1]( 3 ) = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 );
        src[1]( 4 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );
        src[1]( 5 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );

        dense::Vec< double, 6 > dst[num_wedges];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        // std::cout << A[0] << std::endl;
        // std::cout << A[1] << std::endl;

        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell ), dst[0]( 0 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell ), dst[0]( 1 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell ), dst[0]( 2 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), dst[0]( 3 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ), dst[0]( 4 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ), dst[0]( 5 ) );

        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell ), dst[1]( 0 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell ), dst[1]( 1 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell ), dst[1]( 2 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 ), dst[1]( 3 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ), dst[1]( 4 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ), dst[1]( 5 ) );
    }
};

struct LaplaceOperatorWedgeOptimized
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;

    Grid4DDataScalar< double > src_;
    Grid4DDataScalar< double > dst_;
    bool                       treat_boundary_;
    bool                       diagonal_;

    constexpr static int num_wedges          = 2;
    constexpr static int num_nodes_per_wedge = 6;

#if 0
    constexpr int    nq              = 2;
    constexpr double one_over_sqrt_3 = 0.57735026918962576450914878050195745564760175127012687601860232648397767230;
    constexpr dense::Vec< double, 3 > qp[nq] = {
        { 1.0 / 3.0, 1.0 / 3.0, -one_over_sqrt_3 }, { 1.0 / 3.0, 1.0 / 3.0, one_over_sqrt_3 } };
    constexpr double qw[nq] = { 1.0, 1.0 };
#endif

#if 1
    constexpr static int                     nq     = 1;
    constexpr static dense::Vec< double, 3 > qp[nq] = { { 1.0 / 3.0, 1.0 / 3.0, 0.0 } };
    constexpr static double                  qw[nq] = { 1.0 };
#endif

#if 0

    constexpr int                     nq     = 6;
    constexpr dense::Vec< double, 3 > qp[nq] = {
        { { 0.6666666666666666, 0.1666666666666667, -0.5773502691896257 } },
        { { 0.1666666666666667, 0.6666666666666666, -0.5773502691896257 } },
        { { 0.1666666666666667, 0.1666666666666667, -0.5773502691896257 } },
        { { 0.6666666666666666, 0.1666666666666667, 0.5773502691896257 } },
        { { 0.1666666666666667, 0.6666666666666666, 0.5773502691896257 } },
        { { 0.1666666666666667, 0.1666666666666667, 0.5773502691896257 } } };
    constexpr double qw[nq] = {
        0.1666666666666667,
        0.1666666666666667,
        0.1666666666666667,
        0.1666666666666667,
        0.1666666666666667,
        0.1666666666666667 };
#endif

    LaplaceOperatorWedgeOptimized(
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
        operator()( Kokkos::TeamPolicy< Kokkos::DefaultExecutionSpace >::member_type team_member ) const
    {
        const int num_subdomains = static_cast< int >( src_.extent( 0 ) );
        const int num_cells_x    = static_cast< int >( src_.extent( 1 ) - 1 );
        const int num_cells_y    = static_cast< int >( src_.extent( 2 ) - 1 );
        const int num_cells_r    = static_cast< int >( src_.extent( 3 ) - 1 );

        // --- Recover Outer Loop Indices (subdomain, x, y) ---

        // Get the unique ID for this team (0 to league_size-1)
        const int league_rank = team_member.league_rank();

        // Un-flatten the league_rank to get the 3D outer indices.
        // This is the C-style (row-major) un-flattening logic.
        // flat_index = sub*(N_x*N_y) + x*(N_y) + y
#if 1
        const int y_cell             = league_rank % num_cells_y;
        const int temp_index         = league_rank / num_cells_y;
        const int x_cell             = temp_index % num_cells_x;
        const int local_subdomain_id = temp_index / num_cells_x;
#else
        const int local_subdomain_id = league_rank % num_subdomains;
        const int temp_index         = league_rank / num_subdomains;
        const int x_cell             = temp_index % num_cells_x;
        const int y_cell             = temp_index / num_cells_x;
#endif

        // First all the r-independent stuff.
        // Gather surface points for each wedge.
        constexpr int num_wedges = 2;

        // Extract vertex positions of quad
        // (0, 0), (1, 0), (0, 1), (1, 1).
        dense::Vec< double, 3 > quad_surface_coords[2][2];

        for ( int x = x_cell; x <= x_cell + 1; x++ )
        {
            for ( int y = y_cell; y <= y_cell + 1; y++ )
            {
                for ( int d = 0; d < 3; d++ )
                {
                    quad_surface_coords[x - x_cell][y - y_cell]( d ) = grid_( local_subdomain_id, x, y, d );
                }
            }
        }

        // Sort coords for the two wedge surfaces.
        dense::Vec< double, 3 > wedge_phy_surf[num_wedges][3] = {};

        wedge_phy_surf[0][0] = quad_surface_coords[0][0];
        wedge_phy_surf[0][1] = quad_surface_coords[1][0];
        wedge_phy_surf[0][2] = quad_surface_coords[0][1];

        wedge_phy_surf[1][0] = quad_surface_coords[1][1];
        wedge_phy_surf[1][1] = quad_surface_coords[0][1];
        wedge_phy_surf[1][2] = quad_surface_coords[1][0];

        // Compute lateral part of Jacobian.
        // For now, we only do this at a single quad point.

        dense::Mat< double, 3, 3 > jac_lat_inv_t[num_wedges][nq] = {};
        double                     det_jac_lat[num_wedges][nq]   = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int q = 0; q < nq; q++ )
            {
                const auto jac_lat = fe::wedge::jac_lat(
                    wedge_phy_surf[wedge][0],
                    wedge_phy_surf[wedge][1],
                    wedge_phy_surf[wedge][2],
                    qp[q]( 0 ),
                    qp[q]( 1 ) );

                det_jac_lat[wedge][q] = Kokkos::abs( jac_lat.det() );

                jac_lat_inv_t[wedge][q] = jac_lat.inv().transposed();
            }
        }

        // Let's now gather all the shape functions and gradients we need.
        double shape_lat[num_wedges][num_nodes_per_wedge][nq] = {};
        double shape_rad[num_wedges][num_nodes_per_wedge][nq] = {};

        double grad_shape_lat_xi[num_wedges][num_nodes_per_wedge]  = {};
        double grad_shape_lat_eta[num_wedges][num_nodes_per_wedge] = {};
        double grad_shape_rad[num_wedges][num_nodes_per_wedge]     = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                for ( int q = 0; q < nq; q++ )
                {
                    shape_lat[wedge][node_idx][q] = fe::wedge::shape_lat( qp[q]( 0 ), qp[q]( 1 ) )( node_idx % 3 );
                    shape_rad[wedge][node_idx][q] = fe::wedge::shape_rad( qp[q]( 2 ) )( node_idx / 3 );
                }

                grad_shape_lat_xi[wedge][node_idx]  = fe::wedge::grad_shape_lat_xi()( node_idx % 3 );
                grad_shape_lat_eta[wedge][node_idx] = fe::wedge::grad_shape_lat_eta()( node_idx % 3 );
                grad_shape_rad[wedge][node_idx]     = fe::wedge::grad_shape_rad()( node_idx / 3 );
            }
        }

        dense::Vec< double, 3 > g_rad[num_wedges][num_nodes_per_wedge][nq] = {};
        dense::Vec< double, 3 > g_lat[num_wedges][num_nodes_per_wedge][nq] = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                for ( int q = 0; q < nq; q++ )
                {
                    g_rad[wedge][node_idx][q] = jac_lat_inv_t[wedge][q] *
                                                dense::Vec< double, 3 >{
                                                    grad_shape_lat_xi[wedge][node_idx] * shape_rad[wedge][node_idx][q],
                                                    grad_shape_lat_eta[wedge][node_idx] * shape_rad[wedge][node_idx][q],
                                                    0.0 };

                    g_lat[wedge][node_idx][q] =
                        jac_lat_inv_t[wedge][q] * dense::Vec< double, 3 >{ 0.0, 0.0, shape_lat[wedge][node_idx][q] };
                }
            }
        }

        // Only now we introduce radially dependent terms.

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team_member, num_cells_r ), [&]( const int& r_cell ) {
            // Inside this lambda, we have all four indices!
            // (local_subdomain_id, x_cell, y_cell, r_cell)

            const double r_1 = radii_( local_subdomain_id, r_cell );
            const double r_2 = radii_( local_subdomain_id, r_cell + 1 );

            // For now, compute the local element matrix. We'll improve that later.
            dense::Mat< double, 6, 6 > A[num_wedges] = {};

            for ( int wedge = 0; wedge < num_wedges; wedge++ )
            {
                for ( int q = 0; q < nq; q++ )
                {
                    const double r = fe::wedge::forward_map_rad( r_1, r_2, qp[q]( 2 ) );
                    // TODO: we can precompute that per quadrature point to avoid the division.
                    const double r_inv = 1.0 / r;

                    const double grad_r = fe::wedge::grad_forward_map_rad( r_1, r_2 );
                    // TODO: we can precompute that per quadrature point to avoid the division.
                    const double grad_r_inv = 1.0 / grad_r;

                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            const dense::Vec< double, 3 > grad_i =
                                r_inv * g_rad[wedge][i][q] + grad_shape_rad[wedge][i] * grad_r_inv * g_lat[wedge][i][q];
                            const dense::Vec< double, 3 > grad_j =
                                r_inv * g_rad[wedge][j][q] + grad_shape_rad[wedge][j] * grad_r_inv * g_lat[wedge][j][q];

                            A[wedge]( i, j ) +=
                                qw[q] * ( grad_i.dot( grad_j ) * r * r * grad_r * det_jac_lat[wedge][q] );
                        }
                    }
                }
            }

            if ( treat_boundary_ )
            {
                for ( int wedge = 0; wedge < num_wedges; wedge++ )
                {
                    dense::Mat< double, 6, 6 > boundary_mask;
                    boundary_mask.fill( 1.0 );
                    if ( r_cell == 0 )
                    {
                        // Inner boundary (CMB).
                        for ( int i = 0; i < 6; i++ )
                        {
                            for ( int j = 0; j < 6; j++ )
                            {
                                if ( i != j && ( i < 3 || j < 3 ) )
                                {
                                    boundary_mask( i, j ) = 0.0;
                                }
                            }
                        }
                    }

                    if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
                    {
                        // Outer boundary (surface).
                        for ( int i = 0; i < 6; i++ )
                        {
                            for ( int j = 0; j < 6; j++ )
                            {
                                if ( i != j && ( i >= 3 || j >= 3 ) )
                                {
                                    boundary_mask( i, j ) = 0.0;
                                }
                            }
                        }
                    }

                    A[wedge].hadamard_product( boundary_mask );
                }
            }

            if ( diagonal_ )
            {
                for ( int wedge = 0; wedge < num_wedges; wedge++ )
                {
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j )
                            {
                                A[wedge]( i, j ) = 0.0;
                            }
                        }
                    }
                }
            }

            dense::Vec< double, 6 > src[num_wedges];

            src[0]( 0 ) = src_( local_subdomain_id, x_cell, y_cell, r_cell );
            src[0]( 1 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
            src[0]( 2 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
            src[0]( 3 ) = src_( local_subdomain_id, x_cell, y_cell, r_cell + 1 );
            src[0]( 4 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
            src[0]( 5 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );

            src[1]( 0 ) = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell );
            src[1]( 1 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
            src[1]( 2 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
            src[1]( 3 ) = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 );
            src[1]( 4 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );
            src[1]( 5 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );

            dense::Vec< double, 6 > dst[num_wedges];

            dst[0] = A[0] * src[0];
            dst[1] = A[1] * src[1];

            // std::cout << A[0] << std::endl;
            // std::cout << A[1] << std::endl;

            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell ), dst[0]( 0 ) );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell ), dst[0]( 1 ) );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell ), dst[0]( 2 ) );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), dst[0]( 3 ) );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ), dst[0]( 4 ) );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ), dst[0]( 5 ) );

            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell ), dst[1]( 0 ) );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell ), dst[1]( 1 ) );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell ), dst[1]( 2 ) );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 ), dst[1]( 3 ) );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ), dst[1]( 4 ) );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ), dst[1]( 5 ) );
        } );
    }
};

struct LaplaceOperatorWedgePrecomp
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;

    Grid4DDataScalar< double > src_;
    Grid4DDataScalar< double > dst_;
    bool                       treat_boundary_;
    bool                       diagonal_;

    constexpr static int num_wedges          = 2;
    constexpr static int num_nodes_per_wedge = 6;

#if 0
    constexpr int    nq              = 2;
    constexpr double one_over_sqrt_3 = 0.57735026918962576450914878050195745564760175127012687601860232648397767230;
    constexpr dense::Vec< double, 3 > qp[nq] = {
        { 1.0 / 3.0, 1.0 / 3.0, -one_over_sqrt_3 }, { 1.0 / 3.0, 1.0 / 3.0, one_over_sqrt_3 } };
    constexpr double qw[nq] = { 1.0, 1.0 };
#endif

#if 1
    constexpr static int                     nq     = 1;
    constexpr static dense::Vec< double, 3 > qp[nq] = { { 1.0 / 3.0, 1.0 / 3.0, 0.0 } };
    constexpr static double                  qw[nq] = { 1.0 };
#endif

#if 0

    constexpr int                     nq     = 6;
    constexpr dense::Vec< double, 3 > qp[nq] = {
        { { 0.6666666666666666, 0.1666666666666667, -0.5773502691896257 } },
        { { 0.1666666666666667, 0.6666666666666666, -0.5773502691896257 } },
        { { 0.1666666666666667, 0.1666666666666667, -0.5773502691896257 } },
        { { 0.6666666666666666, 0.1666666666666667, 0.5773502691896257 } },
        { { 0.1666666666666667, 0.6666666666666666, 0.5773502691896257 } },
        { { 0.1666666666666667, 0.1666666666666667, 0.5773502691896257 } } };
    constexpr double qw[nq] = {
        0.1666666666666667,
        0.1666666666666667,
        0.1666666666666667,
        0.1666666666666667,
        0.1666666666666667,
        0.1666666666666667 };
#endif

    // [subdomain][x_cell][y_cell][wedge][node][q][component]
    Kokkos::View< double*** [num_wedges][num_nodes_per_wedge][nq][3] > g_lat_;
    Kokkos::View< double*** [num_wedges][num_nodes_per_wedge][nq][3] > g_rad_;
    Kokkos::View< double*** [num_wedges][nq] >                         det_jac_lat_;

    LaplaceOperatorWedgePrecomp(
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
    {
        g_lat_ = Kokkos::View< double*** [num_wedges][num_nodes_per_wedge][nq][3] >(
            "g_lat", src.extent( 0 ), src.extent( 1 ), src.extent( 2 ) );
        g_rad_ = Kokkos::View< double*** [num_wedges][num_nodes_per_wedge][nq][3] >(
            "g_rad", src.extent( 0 ), src.extent( 1 ), src.extent( 2 ) );
        det_jac_lat_ = Kokkos::View< double*** [num_wedges][nq] >(
            "det_jac_lat", src.extent( 0 ), src.extent( 1 ), src.extent( 2 ) );
    }

    KOKKOS_INLINE_FUNCTION void operator()( const int local_subdomain_id, const int x_cell, const int y_cell ) const
    {
        // Extract vertex positions of quad
        // (0, 0), (1, 0), (0, 1), (1, 1).
        dense::Vec< double, 3 > quad_surface_coords[2][2];

        for ( int x = x_cell; x <= x_cell + 1; x++ )
        {
            for ( int y = y_cell; y <= y_cell + 1; y++ )
            {
                for ( int d = 0; d < 3; d++ )
                {
                    quad_surface_coords[x - x_cell][y - y_cell]( d ) = grid_( local_subdomain_id, x, y, d );
                }
            }
        }

        // Sort coords for the two wedge surfaces.
        dense::Vec< double, 3 > wedge_phy_surf[num_wedges][3] = {};

        wedge_phy_surf[0][0] = quad_surface_coords[0][0];
        wedge_phy_surf[0][1] = quad_surface_coords[1][0];
        wedge_phy_surf[0][2] = quad_surface_coords[0][1];

        wedge_phy_surf[1][0] = quad_surface_coords[1][1];
        wedge_phy_surf[1][1] = quad_surface_coords[0][1];
        wedge_phy_surf[1][2] = quad_surface_coords[1][0];

        // Compute lateral part of Jacobian.
        // For now, we only do this at a single quad point.

        dense::Mat< double, 3, 3 > jac_lat_inv_t[num_wedges][nq] = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int q = 0; q < nq; q++ )
            {
                const auto jac_lat = fe::wedge::jac_lat(
                    wedge_phy_surf[wedge][0],
                    wedge_phy_surf[wedge][1],
                    wedge_phy_surf[wedge][2],
                    qp[q]( 0 ),
                    qp[q]( 1 ) );

                det_jac_lat_( local_subdomain_id, x_cell, y_cell, wedge, q ) = Kokkos::abs( jac_lat.det() );

                jac_lat_inv_t[wedge][q] = jac_lat.inv().transposed();
            }
        }

        // Let's now gather all the shape functions and gradients we need.
        double shape_lat[num_wedges][num_nodes_per_wedge][nq] = {};
        double shape_rad[num_wedges][num_nodes_per_wedge][nq] = {};

        double grad_shape_lat_xi[num_wedges][num_nodes_per_wedge]  = {};
        double grad_shape_lat_eta[num_wedges][num_nodes_per_wedge] = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                for ( int q = 0; q < nq; q++ )
                {
                    shape_lat[wedge][node_idx][q] = fe::wedge::shape_lat( qp[q]( 0 ), qp[q]( 1 ) )( node_idx % 3 );
                    shape_rad[wedge][node_idx][q] = fe::wedge::shape_rad( qp[q]( 2 ) )( node_idx / 3 );
                }

                grad_shape_lat_xi[wedge][node_idx]  = fe::wedge::grad_shape_lat_xi()( node_idx % 3 );
                grad_shape_lat_eta[wedge][node_idx] = fe::wedge::grad_shape_lat_eta()( node_idx % 3 );
            }
        }

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                for ( int q = 0; q < nq; q++ )
                {
                    const dense::Vec< double, 3 > g_rad_local =
                        jac_lat_inv_t[wedge][q] *
                        dense::Vec< double, 3 >{
                            grad_shape_lat_xi[wedge][node_idx] * shape_rad[wedge][node_idx][q],
                            grad_shape_lat_eta[wedge][node_idx] * shape_rad[wedge][node_idx][q],
                            0.0 };
                    g_rad_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 0 ) = g_rad_local( 0 );
                    g_rad_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 1 ) = g_rad_local( 1 );
                    g_rad_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 2 ) = g_rad_local( 2 );

                    const dense::Vec< double, 3 > g_lat_local =
                        jac_lat_inv_t[wedge][q] * dense::Vec< double, 3 >{ 0.0, 0.0, shape_lat[wedge][node_idx][q] };
                    g_lat_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 0 ) = g_lat_local( 0 );
                    g_lat_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 1 ) = g_lat_local( 1 );
                    g_lat_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 2 ) = g_lat_local( 2 );
                }
            }
        }
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Only now we introduce radially dependent terms.

        // Inside this lambda, we have all four indices!
        // (local_subdomain_id, x_cell, y_cell, r_cell)

        const double r_1 = radii_( local_subdomain_id, r_cell );
        const double r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // For now, compute the local element matrix. We'll improve that later.
        dense::Mat< double, 6, 6 > A[num_wedges] = {};

        dense::Vec< double, 3 > g_rad[num_wedges][num_nodes_per_wedge][nq];
        dense::Vec< double, 3 > g_lat[num_wedges][num_nodes_per_wedge][nq];

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                for ( int q = 0; q < nq; q++ )
                {
                    g_rad[wedge][node_idx][q] = {
                        g_rad_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 0 ),
                        g_rad_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 1 ),
                        g_rad_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 2 ) };

                    g_lat[wedge][node_idx][q] = {
                        g_lat_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 0 ),
                        g_lat_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 1 ),
                        g_lat_( local_subdomain_id, x_cell, y_cell, wedge, node_idx, q, 2 ) };
                }
            }
        }

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int q = 0; q < nq; q++ )
            {
                const double r = fe::wedge::forward_map_rad( r_1, r_2, qp[q]( 2 ) );
                // TODO: we can precompute that per quadrature point to avoid the division.
                const double r_inv = 1.0 / r;

                const double grad_r = fe::wedge::grad_forward_map_rad( r_1, r_2 );
                // TODO: we can precompute that per quadrature point to avoid the division.
                const double grad_r_inv = 1.0 / grad_r;

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const double grad_shape_rad_i = fe::wedge::grad_shape_rad()( i / 3 );
                        const double grad_shape_rad_j = fe::wedge::grad_shape_rad()( j / 3 );

                        const dense::Vec< double, 3 > grad_i =
                            r_inv * g_rad[wedge][i][q] + grad_shape_rad_i * grad_r_inv * g_lat[wedge][i][q];
                        const dense::Vec< double, 3 > grad_j =
                            r_inv * g_rad[wedge][j][q] + grad_shape_rad_j * grad_r_inv * g_lat[wedge][j][q];

                        A[wedge]( i, j ) += qw[q] * ( grad_i.dot( grad_j ) * r * r * grad_r *
                                                      det_jac_lat_( local_subdomain_id, x_cell, y_cell, wedge, q ) );
                    }
                }
            }
        }

        if ( treat_boundary_ )
        {
            for ( int wedge = 0; wedge < num_wedges; wedge++ )
            {
                dense::Mat< double, 6, 6 > boundary_mask;
                boundary_mask.fill( 1.0 );
                if ( r_cell == 0 )
                {
                    // Inner boundary (CMB).
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j && ( i < 3 || j < 3 ) )
                            {
                                boundary_mask( i, j ) = 0.0;
                            }
                        }
                    }
                }

                if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
                {
                    // Outer boundary (surface).
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j && ( i >= 3 || j >= 3 ) )
                            {
                                boundary_mask( i, j ) = 0.0;
                            }
                        }
                    }
                }

                A[wedge].hadamard_product( boundary_mask );
            }
        }

        if ( diagonal_ )
        {
            for ( int wedge = 0; wedge < num_wedges; wedge++ )
            {
                for ( int i = 0; i < 6; i++ )
                {
                    for ( int j = 0; j < 6; j++ )
                    {
                        if ( i != j )
                        {
                            A[wedge]( i, j ) = 0.0;
                        }
                    }
                }
            }
        }

        dense::Vec< double, 6 > src[num_wedges];

        src[0]( 0 ) = src_( local_subdomain_id, x_cell, y_cell, r_cell );
        src[0]( 1 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
        src[0]( 2 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
        src[0]( 3 ) = src_( local_subdomain_id, x_cell, y_cell, r_cell + 1 );
        src[0]( 4 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
        src[0]( 5 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );

        src[1]( 0 ) = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell );
        src[1]( 1 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
        src[1]( 2 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
        src[1]( 3 ) = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 );
        src[1]( 4 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );
        src[1]( 5 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );

        dense::Vec< double, 6 > dst[num_wedges];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        // std::cout << A[0] << std::endl;
        // std::cout << A[1] << std::endl;

        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell ), dst[0]( 0 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell ), dst[0]( 1 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell ), dst[0]( 2 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), dst[0]( 3 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ), dst[0]( 4 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ), dst[0]( 5 ) );

        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell ), dst[1]( 0 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell ), dst[1]( 1 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell ), dst[1]( 2 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 ), dst[1]( 3 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ), dst[1]( 4 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ), dst[1]( 5 ) );
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

    kernels::common::lincomb( x, 0.0, 1.0, x, omega, b, -omega, tmp );
}

void single_apply()
{
    const auto domain = grid::shell::DistributedDomain::create_uniform_single_subdomain( 3, 3, 0.5, 1.0 );

    const auto src = grid::shell::allocate_scalar_grid< double >( "src", domain );
    const auto dst = grid::shell::allocate_scalar_grid< double >( "dst", domain );

    communication::shell::SubdomainNeighborhoodSendBuffer< double > send_buffers( domain );
    communication::shell::SubdomainNeighborhoodRecvBuffer< double > recv_buffers( domain );

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
        LaplaceOperatorWedge( subdomain_shell_coords, subdomain_radii, src, dst, false, false ) );
#endif

#if 1
    communication::shell::pack_and_send_local_subdomain_boundaries(
        domain, dst, send_buffers, expected_recvs_requests, expected_recvs_metadata );

    MPI_Barrier( MPI_COMM_WORLD );
    Kokkos::fence();

    communication::shell::recv_unpack_and_add_local_subdomain_boundaries(
        domain, dst, recv_buffers, expected_recvs_requests, expected_recvs_metadata );
#endif

    terra::vtk::VTKOutput vtk_after( subdomain_shell_coords, subdomain_radii, "laplace_apply.vtu", false );
    vtk_after.add_scalar_field( src.label(), src );
    vtk_after.add_scalar_field( dst.label(), dst );
    vtk_after.write();
}

#if 1

void apply_laplacian(
    const auto& domain,
    const auto& subdomain_shell_coords,
    const auto& subdomain_radii,
    const auto& src,
    const auto& dst,
    const bool  treat_boundary,
    const bool  diagonal )
{
#if 0
    Kokkos::parallel_for(
        "matvec",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        LaplaceOperatorWedge( subdomain_shell_coords, subdomain_radii, src, dst, treat_boundary, diagonal ) );
#else

    // The league size is the total number of work items for the outer loops
    const int num_subdomains = src.extent( 0 );
    const int num_cells_x    = src.extent( 1 ) - 1;
    const int num_cells_y    = src.extent( 2 ) - 1;
    const int num_cells_r    = src.extent( 3 ) - 1;

    const int league_size = num_subdomains * num_cells_x * num_cells_y;

    // The number of threads in each team. This should be large enough
    // to cover the inner loop dimension N_r.
    // const int team_size = num_cells_r; // Or Kokkos::AUTO
    const auto team_size = Kokkos::AUTO;

    // Create the policy
    using team_policy_t = Kokkos::TeamPolicy<>;
    team_policy_t policy( league_size, team_size );

    Kokkos::parallel_for(
        "matvec",
        policy,
        LaplaceOperatorWedgeOptimized( subdomain_shell_coords, subdomain_radii, src, dst, treat_boundary, diagonal ) );

#endif
}

void all_diamonds()
{
    using LaplaceOperator = LaplaceOperatorWedge;

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

    const auto domain = grid::shell::DistributedDomain::create_uniform_single_subdomain( 4, 4, 0.5, 1.0 );

    const auto u        = grid::shell::allocate_scalar_grid< double >( "u", domain );
    const auto g        = grid::shell::allocate_scalar_grid< double >( "g", domain );
    const auto Adiagg   = grid::shell::allocate_scalar_grid< double >( "Adiagg", domain );
    const auto tmp      = grid::shell::allocate_scalar_grid< double >( "tmp", domain );
    const auto solution = grid::shell::allocate_scalar_grid< double >( "solution", domain );
    const auto error    = grid::shell::allocate_scalar_grid< double >( "error", domain );
    const auto b        = grid::shell::allocate_scalar_grid< double >( "b", domain );
    const auto r        = grid::shell::allocate_scalar_grid< double >( "r", domain );

    const int num_dofs = u.span();
    std::cout << "Num DoFs: " << num_dofs << std::endl;

    communication::shell::SubdomainNeighborhoodSendBuffer< double > send_buffers( domain );
    communication::shell::SubdomainNeighborhoodRecvBuffer< double > recv_buffers( domain );

    std::vector< std::array< int, 11 > > expected_recvs_metadata;
    std::vector< MPI_Request >           expected_recvs_requests;

    const auto subdomain_shell_coords = terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain );
    const auto subdomain_radii        = terra::grid::shell::subdomain_shell_radii( domain );

    LaplaceOperator lapl( subdomain_shell_coords, subdomain_radii, tmp, tmp, false, true );

    if constexpr ( std::is_same_v< LaplaceOperator, LaplaceOperatorWedgePrecomp > )
    {
        // Precompute
        Kokkos::parallel_for(
            "matvec",
            Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
                { 0, 0, 0 },
                { static_cast< long long >( domain.subdomains().size() ),
                  domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                  domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1 } ),
            lapl );
    }

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

    lapl.src_            = g;
    lapl.dst_            = Adiagg;
    lapl.treat_boundary_ = false;
    lapl.diagonal_       = true;
    Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain ), lapl );

    communication::shell::pack_and_send_local_subdomain_boundaries(
        domain, Adiagg, send_buffers, expected_recvs_requests, expected_recvs_metadata );
    communication::shell::recv_unpack_and_add_local_subdomain_boundaries(
        domain, Adiagg, recv_buffers, expected_recvs_requests, expected_recvs_metadata );

    lapl.src_            = g;
    lapl.dst_            = b;
    lapl.treat_boundary_ = false;
    lapl.diagonal_       = false;
    Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain ), lapl );

    communication::shell::pack_and_send_local_subdomain_boundaries(
        domain, b, send_buffers, expected_recvs_requests, expected_recvs_metadata );
    communication::shell::recv_unpack_and_add_local_subdomain_boundaries(
        domain, b, recv_buffers, expected_recvs_requests, expected_recvs_metadata );

    terra::kernels::common::scale( b, -1.0 );

    Kokkos::parallel_for(
        "set on boundary",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        SetOnBoundary( Adiagg, b, domain.domain_info().subdomain_num_nodes_radially() ) );

    // Solve.

    const double omega = 0.2;

    double duration_matvec_sum = 0;
    double duration_iter_sum   = 0;

    const int iterations = 10;

    for ( int iter = 0; iter < iterations; iter++ )
    {
        if ( iter % 1 == 0 )
        {
            std::cout << "iter = " << iter;
            const bool comp_norm = true;
            if ( comp_norm )
            {
                kernels::common::lincomb( error, 0.0, 1.0, u, -1.0, solution );
                const auto error_inf_norm = kernels::common::max_magnitude( error );
                std::cout << ", error inf norm: " << error_inf_norm;

                kernels::common::set_constant( tmp, 0.0 );

                lapl.src_            = u;
                lapl.dst_            = tmp;
                lapl.treat_boundary_ = true;
                lapl.diagonal_       = false;
                Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain ), lapl );

                communication::shell::pack_and_send_local_subdomain_boundaries(
                    domain, tmp, send_buffers, expected_recvs_requests, expected_recvs_metadata );
                communication::shell::recv_unpack_and_add_local_subdomain_boundaries(
                    domain, tmp, recv_buffers, expected_recvs_requests, expected_recvs_metadata );

                kernels::common::lincomb( r, 0.0, 1.0, b, -1.0, tmp );

                const auto residual_inf_norm = kernels::common::max_magnitude( r );
                std::cout << ", residual inf norm: " << residual_inf_norm;
            }
            std::cout << std::endl;
        }

        Kokkos::Timer timer;

        timer.reset();

        // We need that in matvec - maybe resolved when stencils?
        kernels::common::set_constant( tmp, 0.0 );

        Kokkos::Timer timer_matvec;
        lapl.src_            = u;
        lapl.dst_            = tmp;
        lapl.treat_boundary_ = true;
        lapl.diagonal_       = false;
        timer_matvec.reset();
        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain ), lapl );
        Kokkos::fence();
        duration_matvec_sum += timer_matvec.seconds();

        communication::shell::pack_and_send_local_subdomain_boundaries(
            domain, tmp, send_buffers, expected_recvs_requests, expected_recvs_metadata );
        communication::shell::recv_unpack_and_add_local_subdomain_boundaries(
            domain, tmp, recv_buffers, expected_recvs_requests, expected_recvs_metadata );

        kernels::common::lincomb( u, 0.0, 1.0, u, omega, b, -omega, tmp );

        Kokkos::fence();
        duration_iter_sum += timer.seconds();
    }

    const double avg_iteration_duration     = duration_iter_sum / iterations;
    const double avg_matvec_duration        = duration_matvec_sum / iterations;
    const double dofs_per_second_per_iter   = num_dofs / avg_iteration_duration;
    const double dofs_per_second_per_matvec = num_dofs / avg_matvec_duration;
    std::cout << "Average iteration duration: " << avg_iteration_duration << " seconds" << std::endl;
    std::cout << "Average matvec duration:    " << avg_matvec_duration << " seconds" << std::endl;
    std::cout << "Dofs per second per iteration: " << dofs_per_second_per_iter << std::endl;
    std::cout << "Dofs per second per matvec:    " << dofs_per_second_per_matvec << std::endl;

    kernels::common::lincomb( error, 0.0, 1.0, u, -1.0, solution );

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
    Kokkos::ScopeGuard scope_guard( argc, argv );

    single_apply();
    all_diamonds();

    MPI_Finalize();
    return 0;
}