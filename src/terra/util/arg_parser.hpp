
#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace terra::util {

/// @brief Simple argument parser.
///
/// Parses all arguments starting with one or two dashes followed by and integer, double, string or nothing (bool)
/// or a list of integers, doubles, or strings.
///
/// Optionally, the first argument can be a text file with space-separated arguments.
/// Inside that text file, in each line everything including and after a '#' character is skipped.
///
/// This parser does not implement any logic about missing/required arguments. It just parses what's there.
/// However, you can check that later via has() and get< T >(). The latter will throw if no argument was found
/// (also you can supply default values).
///
/// Some examples:
///
/// @code
/// int main( int argc, char** argv )
/// {
///     terra::util::terra_initialize( &argc, &argv );
///     const terra::util::ArgParser args( argc, argv );
///     args.print();
///     return 0;
/// }
/// @endcode
///
/// with
///
/// @code
/// ./test_arg_parser -a -b 42 -c 42.1 -d hi --aaa --bbb 43 --ccc 43.1 --ddd hello
/// @endcode
///
/// will print
///
/// @code
/// a = true
/// aaa = true
/// b = 42
/// bbb = 43
/// c = 42.1
/// ccc = 43.1
/// d = hi
/// ddd = hello
/// @endcode
///
/// You can also add a text file, lists (converted to std::vector), and longer strings, like this:
///
/// @note The text file must be the first argument!
/// @note Command-line arguments overwrite textfile arguments.
///
/// @code
/// # some_argument_file.txt (lines are ignored after '#')
/// -e
/// --eee I am a vector of strings       # some comment
/// --eee-long-string "I am a long string"
/// --bbb "I will be overwritten by CLI"
/// @endcode
///
/// with
///
/// @code
/// ./test_arg_parser test.txt -a --bbb 43 --asdf 4 5 6
/// @endcode
///
/// gives
///
/// @code
/// asdf = [4, 5, 6]
/// bbb = 43
/// a = true
/// eee = [I, am, a, vector, of, strings]
/// eee-long-string = I am a long string
/// e = true
/// @endcode
///
class [[deprecated("Use CLI11 - see test_arg_parser.cpp")]] ArgParser
{
  public:
    using Scalar = std::variant< int, double, std::string, bool >;
    using Value  = std::variant< Scalar, std::vector< Scalar > >;

    ArgParser( int argc, char** argv )
    {
        if ( argc > 1 && !is_flag( argv[1] ) )
        {
            auto tokens = load_file( argv[1] );
            parse_tokens( tokens, /*overwrite=*/false );
            start_index = 2;
        }
        else
        {
            start_index = 1;
        }
        std::vector< std::string > cli_tokens( argv + start_index, argv + argc );
        parse_tokens( cli_tokens, /*overwrite=*/true ); // CLI overwrites
    }

    template < typename T >
    T get( const std::string& key ) const
    {
        auto it = args.find( key );
        if ( it == args.end() )
        {
            throw std::runtime_error( "ArgParser::get: key not found" );
        }

        // vector<T>
        if constexpr (
            std::is_same_v< T, std::vector< int > > || std::is_same_v< T, std::vector< double > > ||
            std::is_same_v< T, std::vector< std::string > > || std::is_same_v< T, std::vector< bool > > )
        {
            if ( auto vec = std::get_if< std::vector< Scalar > >( &it->second ) )
            {
                std::vector< typename T::value_type > out;
                for ( auto& s : *vec )
                {
                    if ( auto v = std::get_if< T::value_type >( &s ) )
                    {
                        out.push_back( *v );
                    }
                    else
                    {
                        throw std::runtime_error( "Type mismatch in vector for key: " + key );
                    }
                }
                return out;
            }
            // scalar promoted to vector
            if ( auto sc = std::get_if< Scalar >( &it->second ) )
            {
                if ( auto v = std::get_if< T::value_type >( sc ) )
                {
                    return std::vector< typename T::value_type >{ *v };
                }
            }
            throw std::runtime_error( "Type mismatch for key: " + key );
        }
        else
        {
            // scalar
            if ( auto sc = std::get_if< Scalar >( &it->second ) )
            {
                if ( auto v = std::get_if< T >( sc ) )
                    return *v;
            }
            throw std::runtime_error( "Type mismatch for key: " + key );
        }
    }

    template < typename T >
    T get( const std::string& key, const T& default_val ) const
    {
        auto it = args.find( key );
        if ( it == args.end() )
        {
            return default_val;
        }

        return get< T >( key );
    }

    [[nodiscard]] bool has( const std::string& key ) const { return args.find( key ) != args.end(); }

    template < typename T >
    [[nodiscard]] bool has_typed( const std::string& key ) const
    {
        auto it = args.find( key );
        if ( it == args.end() )
        {
            return false;
        }

        return std::holds_alternative< T >( it->second );
    }

    [[nodiscard]] auto begin() const { return args.begin(); }
    [[nodiscard]] auto end() const { return args.end(); }

    void print( std::ostream& os = std::cout ) const
    {
        for ( auto& [k, v] : args )
        {
            os << k << " = ";
            std::visit(
                [&]( auto&& val ) {
                    using U = std::decay_t< decltype( val ) >;
                    if constexpr ( std::is_same_v< U, Scalar > )
                    {
                        std::visit( [&]( auto&& s ) { os << s; }, val );
                    }
                    else if constexpr ( std::is_same_v< U, std::vector< Scalar > > )
                    {
                        os << "[";
                        for ( size_t i = 0; i < val.size(); ++i )
                        {
                            std::visit( [&]( auto&& s ) { os << s; }, val[i] );
                            if ( i + 1 < val.size() )
                                os << ", ";
                        }
                        os << "]";
                    }
                },
                v );
            os << "\n";
        }
    }

  private:
    std::unordered_map< std::string, Value > args;
    int                                      start_index = 1;

    bool is_flag( const std::string& s ) const { return !s.empty() && s[0] == '-'; }

    // simple tokenizer with quotes + comments
    std::vector< std::string > load_file( const std::string& filename )
    {
        std::ifstream in( filename );
        if ( !in )
            throw std::runtime_error( "Could not open file: " + filename );

        std::vector< std::string > tokens;
        std::string                line;
        while ( std::getline( in, line ) )
        {
            std::string token;
            bool        in_quote = false;
            for ( size_t i = 0; i < line.size(); ++i )
            {
                char c = line[i];
                if ( !in_quote && c == '#' )
                    break; // comment
                if ( c == '"' )
                {
                    in_quote = !in_quote;
                    if ( !in_quote )
                    { // closing quote
                        tokens.push_back( token );
                        token.clear();
                    }
                }
                else if ( std::isspace( (unsigned char) c ) && !in_quote )
                {
                    if ( !token.empty() )
                    {
                        tokens.push_back( token );
                        token.clear();
                    }
                }
                else
                {
                    token.push_back( c );
                }
            }
            if ( !token.empty() )
                tokens.push_back( token );
        }
        return tokens;
    }

    void parse_tokens( const std::vector< std::string >& tokens, bool overwrite )
    {
        for ( size_t i = 0; i < tokens.size(); )
        {
            std::string arg = tokens[i];
            if ( arg.rfind( "--", 0 ) == 0 )
            {
                auto [k, v, consumed] = split_arg( arg.substr( 2 ), tokens, i );
                store( k, v, overwrite );
                i += 1 + consumed; // 1 for the flag itself + N values
            }
            else if ( arg.rfind( "-", 0 ) == 0 )
            {
                auto [k, v, consumed] = split_arg( arg.substr( 1 ), tokens, i );
                store( k, v, overwrite );
                i += 1 + consumed;
            }
            else
            {
                ++i; // skip non-flag token
            }
        }
    }

    struct SplitResult
    {
        std::string                key;
        std::vector< std::string > values;
        size_t                     consumed;
    };

    SplitResult split_arg( const std::string& token, const std::vector< std::string >& tokens, size_t i ) const
    {
        SplitResult out{ token, {}, 0 };
        auto        eq = token.find( '=' );
        if ( eq != std::string::npos )
        {
            out.key = token.substr( 0, eq );
            out.values.push_back( strip_quotes( token.substr( eq + 1 ) ) );
            return out;
        }
        size_t j = i + 1;
        while ( j < tokens.size() && !is_flag( tokens[j] ) )
        {
            out.values.push_back( strip_quotes( tokens[j] ) );
            ++j;
        }
        out.consumed = out.values.size(); // number of values after the flag
        if ( out.values.empty() )
        {
            out.values.push_back( "true" ); // bare flag
        }
        return out;
    }

    void store( const std::string& key, const std::vector< std::string >& vals, bool overwrite )
    {
        std::vector< Scalar > converted;
        for ( auto& v : vals )
            converted.push_back( convert( v ) );

        if ( overwrite || args.find( key ) == args.end() )
        {
            if ( converted.size() == 1 )
            {
                args[key] = converted[0];
            }
            else
            {
                args[key] = converted;
            }
        }
        else
        {
            // only for config file merge mode
            auto& existing = args[key];
            if ( auto vec = std::get_if< std::vector< Scalar > >( &existing ) )
            {
                vec->insert( vec->end(), converted.begin(), converted.end() );
            }
            else if ( auto sc = std::get_if< Scalar >( &existing ) )
            {
                std::vector< Scalar > merged{ *sc };
                merged.insert( merged.end(), converted.begin(), converted.end() );
                existing = merged;
            }
        }
    }

    static std::string strip_quotes( const std::string& s )
    {
        if ( s.size() >= 2 && s.front() == '"' && s.back() == '"' )
            return s.substr( 1, s.size() - 2 );
        return s;
    }

    Scalar convert( const std::string& s ) const
    {
        // try int
        try
        {
            size_t idx;
            int    val = std::stoi( s, &idx );
            if ( idx == s.size() )
                return val;
        }
        catch ( ... )
        {}
        // try double
        try
        {
            size_t idx;
            double val = std::stod( s, &idx );
            if ( idx == s.size() )
                return val;
        }
        catch ( ... )
        {}
        // fallback string
        return s;
    }
};

} // namespace terra::util