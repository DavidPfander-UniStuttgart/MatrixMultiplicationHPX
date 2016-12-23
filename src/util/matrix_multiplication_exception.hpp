#include <exception>
#include <string>

namespace util {

class matrix_multiplication_exception: public std::exception {
private:
  std::string message;
public:
  matrix_multiplication_exception(const std::string &message): message(message) {
  }

  virtual const char* what() const throw() {
    return (std::string("matrix_multiplication_exception: ") + message).c_str();
  }
};

}
