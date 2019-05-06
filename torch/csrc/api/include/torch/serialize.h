#pragma once

#include <torch/detail/static.h>
#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>

#include <utility>

// yf225 TODO: move this somewhere else
// Helper to determine whether there's a const_iterator for T.

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
///   // Note that the same stream cannot be used in multiple torch::save(...)
///   // invocations, otherwise the header will be corrupted.
///   torch::save(sgd, stream);
///
///   auto tensor = torch::ones({3, 4});
///   torch::save(tensor, "my_tensor.pt");
/// \endrst
template <typename Value, typename... SaveToArgs>
torch::disable_if_t<torch::detail::has_const_iterator<Value>::value, void>
save(const Value& value, SaveToArgs&&... args) {
  serialize::OutputArchive archive;
  archive << value;
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

/// Serializes the given `container` of type `ContainerType`
/// (e.g. `std::vector<torch::Tensor>`).
///
/// To perform the serialization, a `serialize::OutputArchive` is constructed,
/// and all arguments after the `tensor_vec` are forwarded to its `save_to`
/// method. For example, you can pass a filename, or an `ostream`.
///
/// \rst
/// .. code-block:: cpp
///
///   std::vector<torch::Tensor> tensor_vec = { torch::randn({1, 2}), torch::randn({3, 4}) };
///   torch::save(tensor_vec, "my_tensor_vec.pt");
///
///   std::vector<torch::Tensor> tensor_vec = { torch::randn({5, 6}), torch::randn({7, 8}) };
///   std::ostringstream stream;
///   // Note that the same stream cannot be used in multiple torch::save(...)
///   // invocations, otherwise the header will be corrupted.
///   torch::save(tensor_vec, stream);
/// \endrst
template <typename ContainerType, typename... SaveToArgs>
torch::enable_if_t<torch::detail::has_const_iterator<ContainerType>::value, void>
save(const ContainerType& container, SaveToArgs&&... args) {
  serialize::OutputArchive archive;
  archive.write("size", torch::tensor(static_cast<int64_t>(container.size())));
  for (size_t index = 0; index < container.size(); ++index) {
    archive.write(std::to_string(index), container[index]);
  }
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
template <typename Value, typename... LoadFromArgs>
torch::disable_if_t<torch::detail::has_const_iterator<Value>::value, void>
load(Value& value, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);
  archive >> value;
}

/// Deserializes the given `container` of type `ContainerType`
/// (e.g. `std::vector<torch::Tensor>`).
///
/// To perform the serialization, a `serialize::InputArchive` is constructed,
/// and all arguments after the `value` are forwarded to its `load_from` method.
/// For example, you can pass a filename, or an `istream`.
///
/// \rst
/// .. code-block:: cpp
///
///   std::vector<torch::Tensor> tensor_vec;
///   torch::load(tensor_vec, "my_tensor_vec.pt");
///
///   std::vector<torch::Tensor> tensor_vec;
///   std::istringstream stream("...");
///   torch::load(tensor_vec, stream);
/// \endrst
template <typename ContainerType, typename... LoadFromArgs>
torch::enable_if_t<torch::detail::has_const_iterator<ContainerType>::value, void>
load(ContainerType& container, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);

  torch::Tensor size_tensor;
  if (archive.try_read("size", size_tensor)) {
    const size_t size = size_tensor.item<int64_t>();
    for (size_t index = 0; index < size; ++index) {
      container.emplace_back();
      archive.read(std::to_string(index), container.back());
    }
  } else {
    // NOTE: In the old deprecated serialization format (introduced in PyTorch 1.1,
    // and deprecated in PyTorch 1.2), the number of elements in the serialized
    // `ContainerType` is not known ahead of time, so we need a while-loop
    // to increment the index, and use `archive.try_read(...)` to check whether
    // we have reached the end of the serialized `ContainerType`.
    size_t index = 0;
    torch::Tensor value;
    while (archive.try_read(std::to_string(index), value)) {
      container.push_back(std::move(value));
      value = torch::Tensor();
      index++;
    }
  }
}

} // namespace torch
