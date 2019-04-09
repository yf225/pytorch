#include <c10/core/AutogradMetaFactory.h>
#include <iostream>

namespace c10 {

std::unique_ptr<c10::AutogradMetaInterface> AutogradMetaFactory::create_something() {
  std::cout << "we are here!" << std::endl;
  return nullptr;
}

}