#include <c10/util/flat_hash_map.h>

#include <string>

namespace torch {

/// yf225 TODO: add doc here
class SerializableOptions {
 public:
  ska::flat_hash_map<std::string, std::reference_wrapper<at::IValue>> arg_map;  // yf225 TODO: make this private and add some nice checks in accessors
};

/// yf225 TODO: provide a free function that can serialize SerializableOptions
}
} // namespace torch