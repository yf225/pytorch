#pragma once

#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>

#include <utility>

namespace torch {

/// Serializes the given `value`.
/// There must be an overload of `operator<<` between `serialize::OutputArchive`
/// and `Value` for this method to be well-formed. Currently, such an overload
/// is provided for (subclasses of):
///
/// - `torch::nn::Module`,
/// - `torch::optim::Optimizer`
/// - `torch::Tensor`
///
/// To perform the serialization, a `serialize::OutputArchive` is constructed,
/// and all arguments after the `value` are forwarded to its `save_to` method.
/// For example, you can pass a filename, or an `ostream`.
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::Linear model(3, 4);
///   torch::save(model, "model.pt");
///
///   torch::optim::SGD sgd(/*lr=*/0.9);
///   std::ostringstream stream;
///   torch::save(sgd, stream);
///
///   auto tensor = torch::ones({3, 4});
///   torch::save(tensor, "my_tensor.pt");
/// \endrst
template <typename Value, typename... SaveToArgs>
void save(const Value& value, SaveToArgs&&... args) {
  serialize::OutputArchive archive;
  archive << value;
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

/// Deserializes the given `value`.
/// There must be an overload of `operator>>` between `serialize::InputArchive`
/// and `Value` for this method to be well-formed. Currently, such an overload
/// is provided for (subclasses of):
///
/// - `torch::nn::Module`,
/// - `torch::optim::Optimizer`
/// - `torch::Tensor`
///
/// To perform the serialization, a `serialize::InputArchive` is constructed,
/// and all arguments after the `value` are forwarded to its `load_from` method.
/// For example, you can pass a filename, or an `istream`.
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::Linear model(3, 4);
///   torch::load(model, "model.pt");
///
///   torch::optim::SGD sgd(/*lr=*/0.9);
///   std::istringstream stream("...");
///   torch::load(sgd, stream);
///
///   auto tensor = torch::ones({3, 4});
///   torch::load(tensor, "my_tensor.pt");
/// \endrst

/*
template <typename First, typename Second, typename... Rest,
    typename = torch::disable_if_t<std::is_same<First, std::string>::value ||
      std::is_same<typename std::decay<First>::type, std::decay<const char (&)[]>::type>::value>>
  void push_back(First&& first, Second&& second, Rest&&... rest) {
    push_back(std::forward<First>(first));
    // Recursively calls this method, until the parameter pack only thas this
    // entry left. Then calls `push_back()` a final time (above).
    push_back(std::forward<Second>(second), std::forward<Rest>(rest)...);
  }
*/

template <typename Value, typename... LoadFromArgs>
void load(Value& value, bool strict, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);

  load_module_from_archive(archive, value, strict);
}

template <typename Value, typename FirstLoadFromArg, typename... RestLoadFromArgs,
  typename = torch::disable_if_t<std::is_same<FirstLoadFromArg, bool>::value>>
void load(Value& value, FirstLoadFromArg&& first, RestLoadFromArgs&&... rest) {
  load(value, false, std::forward<FirstLoadFromArg>(first), std::forward<RestLoadFromArgs>(rest)...);
}
} // namespace torch
