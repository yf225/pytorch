#pragma once

#include <utility>

// yf225 TODO: we won't be able to support the old usage of `options.some_value_ > 5` with the new implementation
// although we can allow people to continue using this if they want, for BC reasons.
// Use TORCH_WARN_ONCE to tell people the existence of TORCH_OPTIONS_ARG
#define TORCH_ARG(T, name)                                       \
  auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name;                                  \
    return *this;                                                \
  }                                                              \
  auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                       \
    return *this;                                                \
  }                                                              \
  const T& name() const noexcept { /* NOLINT */                  \
    return this->name##_;                                        \
  }                                                              \
  T name##_ /* NOLINT */

#define TORCH_OPTIONS_ARG(T, name)                            \
 public: \
  auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name; \
    this->arg_map.emplace(#name, std::ref(this->name##_));              \
    return *this;                                                \
  }                                                              \
  auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                       \
    this->arg_map.emplace(#name, std::ref(this->name##_)); \
    return *this;                                                \
  }                                                              \
  T name() const noexcept { /* NOLINT */                  \
    return name##_.to<T>();                       \
  } \
 private: \
  at::IValue name##_ /* NOLINT */
