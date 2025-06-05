#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace terra::util {

class TablePrinter
{
  public:
    using Cell = std::variant< std::string, int, float, double >;

    TablePrinter( std::ostream& out = std::cout, int padding = 1, int floatPrecision = 2 )
    : out( out )
    , padding( padding )
    , floatPrecision( floatPrecision )
    {}

    void addRow( const std::vector< Cell >& row )
    {
        rows.push_back( row );
        if ( row.size() > columnWidths.size() )
            columnWidths.resize( row.size(), 0 );
        for ( size_t i = 0; i < row.size(); ++i )
        {
            std::string cellStr = formatCell( row[i] );
            columnWidths[i]     = std::max( columnWidths[i], static_cast< int >( cellStr.length() ) );
        }
    }

    void print( bool headerBorder = true, bool rowBorders = false ) const
    {
        for ( size_t rowIdx = 0; rowIdx < rows.size(); ++rowIdx )
        {
            const auto& row = rows[rowIdx];

            // Print row
            for ( size_t i = 0; i < row.size(); ++i )
            {
                bool isNumeric = std::holds_alternative< int >( row[i] ) || std::holds_alternative< float >( row[i] ) ||
                                 std::holds_alternative< double >( row[i] );

                std::string cellStr = formatCell( row[i] );
                if ( isNumeric )
                    out << std::setw( columnWidths[i] + padding ) << std::right << cellStr;
                else
                    out << std::setw( columnWidths[i] + padding ) << std::left << cellStr;
            }
            out << '\n';

            // Print horizontal border
            bool isHeaderLine = ( rowIdx == 0 && headerBorder );
            bool isRowLine    = ( rowIdx < rows.size() - 1 && rowBorders );

            if ( isHeaderLine || isRowLine )
                printSeparator();
        }
    }

  private:
    std::ostream&                      out;
    int                                padding;
    int                                floatPrecision;
    std::vector< std::vector< Cell > > rows;
    std::vector< int >                 columnWidths;

    std::string formatCell( const Cell& cell ) const
    {
        std::ostringstream oss;
        std::visit(
            [&]( auto&& value ) {
                using T = std::decay_t< decltype( value ) >;
                if constexpr ( std::is_floating_point_v< T > )
                    oss << std::scientific << std::setprecision( floatPrecision ) << value;
                else
                    oss << value;
            },
            cell );
        return oss.str();
    }

    void printSeparator() const
    {
        for ( size_t i = 0; i < columnWidths.size(); ++i )
        {
            out << std::string( columnWidths[i] + padding, '-' );
        }
        out << '\n';
    }
};

} // namespace terra::util