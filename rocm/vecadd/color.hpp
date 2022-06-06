#include <ostream>
namespace Color {
    enum Code {
        FG_RED      = 31,
        FG_GREEN    = 32,
        FG_BLUE     = 34,
        FG_DEFAULT  = 39
    };
    class Modifier {
        Code code;
    public :
        Modifier (Code pCode) : code(pCode) {}
        friend std::ostream&
        operator << (std::ostream& os, const Modifier& mod) {
            return os << "\033[" << mod.code << "m";
        }
    };
}
