#include <string>
#include <ATen/core/ivalue.h>

// yf225 TODO: add extensive tests for the intricacy in this macro
// yf225 TODO: it is not worth it to have dual storage. Let's just have one (i.e. only the dict).
#define TORCH_OPTIONS_ARG(T, name)                                       \
 public:                                                         \
  inline auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    this->args.insert(#name, new_##name); \
    return *this;                                                \
  }                                                              \
  inline auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    this->args.insert(#name, std::move(new_##name)); \
    return *this;                                                \
  }                                                              \
  inline const T& name() const noexcept { /* NOLINT */                  \
    if (!this->args.contains(#name)) { \
      this->args.insert(#name, name##_default_value_); \
    } \
    return this->args.at(#name).template to<T>(); \
  } \
 private: \
  T name##_default_value_ /* NOLINT */


namespace torch {

struct TORCH_API SerializableOptions {
 protected:
  c10::Dict<std::string, at::IValue> args;
};

} // namespace torch
