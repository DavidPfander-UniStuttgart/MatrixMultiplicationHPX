#include <exception>
#include <string>

class memory_layout_exception: public std::exception {
private:
  std::string message;
public:
  memory_layout_exception(const std::string &message): message(message) {
  }
  
  virtual const char* what() const throw() {
    return (std::string("memory_layout_exception: ") + message).c_str();
  }
};
