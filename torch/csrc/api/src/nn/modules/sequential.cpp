#include <torch/nn/modules/sequential.h>

namespace torch {
namespace nn {

/// Adds a new (boxed) `Module` to the `Sequential` container.
template <typename ModuleType>
void SequentialImpl::push_back(std::shared_ptr<ModuleType> module_ptr, optional<std::string> name) {
  // Nesting Sequential doesn't work because `forward()`'s return type is
  // templatized, so it'll give a nasty compiler error.
  static_assert(
      !std::is_same<SequentialImpl, ModuleType>::value,
      "Sequential is not nestable");
  static_assert(
      torch::detail::is_module<ModuleType>::value,
      "Can only add objects derived from nn::Module to Sequential");
  static_assert(
      torch::detail::has_forward<ModuleType>::value,
      "Can only add modules with a forward() method to Sequential");
  push_back(AnyModule(std::move(module_ptr)), name);
}

/// Adds a new `Module` to the `Sequential` container, moving or copying it
/// into a `shared_ptr` internally. This method allows passing value types,
/// and letting the container deal with the boxing. This means you can write
/// `Sequential(Module(3, 4))` instead of
/// `Sequential(std::make_shared<Module>(3, 4))`.
template <typename M, typename = torch::detail::enable_if_module_t<M>>
void SequentialImpl::push_back(M&& module, optional<std::string> name) {
  // Need to get rid of any reference components for make_unique.
  using Type = typename std::remove_reference<M>::type;
  // Here we move (or copy) the module into a new shared_ptr.
  push_back(std::make_shared<Type>(std::forward<M>(module)), name);
}

/// Unwraps the contained module of a `ModuleHolder` and adds it to the
/// `Sequential`.
template <typename M>
void SequentialImpl::push_back(const ModuleHolder<M>& module_holder, optional<std::string> name) {
  push_back(module_holder.ptr(), name);
}
