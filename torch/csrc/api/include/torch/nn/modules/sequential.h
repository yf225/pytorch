#pragma once

#include <torch/detail/static.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/any.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

/// A list of `Module`s that acts as a `Module` itself.
///
/// A `AnySequential` is fundamentally a list of `Module`s, each with a
/// `forward()` method. `AnySequential` provides a `forward()` method of its
/// own, which accepts any input and forwards it to the first module it stores.
/// It then "chains" outputs to inputs sequentially for each subsequent module,
/// finally returning the output of the last module. For example:
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::AnySequential seq(
///     torch::nn::Linear(3, 4),
///     torch::nn::BatchNorm(4),
///     torch::nn::Dropout(0.5)
///   );
///
///   auto output = seq->forward(torch::ones(3));
///
/// \endrst
///
/// This can conceptually be thought of as the following loop (using Python as
/// pseudocode):
///
/// \rst
/// .. code-block:: python
///
///   def forward(sequential, input):
///     for module in sequential:
///       input = module(input)
///     return input
///
/// \endrst
///
/// Why should you use `AnySequential` instead of a simple `std::vector`? The
/// value a `AnySequential` provides over manually calling a sequence of modules
/// is that it allows treating the whole container *as a single module*, such
/// that performing a transformation on the `AnySequential` applies to each of
/// the modules it stores (which are each a registered submodule of the
/// `AnySequential`). For example, calling
/// `.to(torch::kCUDA)` on a `AnySequential` will move each module in the list
/// to CUDA memory. For example:
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::AnySequential seq(
///     torch::nn::Linear(3, 4),
///     torch::nn::BatchNorm(4),
///     torch::nn::Dropout(0.5)
///   );
///
///   // Convert all modules to CUDA.
///   seq->to(torch::kCUDA);
///
/// \endrst
///
/// Finally, `AnySequential` provides a lightweight container API, such as
/// allowing iteration over submodules, positional access, adding a new module
/// after construction via `push_back`, as well as joining two `AnySequential`s
/// via `extend`.
///
/// \rst
/// .. attention::
///   One current limitation of `AnySequential` is that all except the first
///   module must accept a single argument. If your modules need to take
///   multiple arguments, you should define them to take and return tuples.
/// \endrst
class AnySequentialImpl : public Cloneable<AnySequentialImpl> {
 public:
  using Iterator = std::vector<AnyModule>::iterator;
  using ConstIterator = std::vector<AnyModule>::const_iterator;

  AnySequentialImpl() = default;

  /// Constructs the `AnySequential` from a variadic list of modules.
  template <typename... Modules>
  explicit AnySequentialImpl(Modules&&... modules) {
    modules_.reserve(sizeof...(Modules));
    push_back(std::forward<Modules>(modules)...);
  }

  /// Constructs the `AnySequential` from an `OrderedDict` of named
  /// `AnyModule`s. Combining with `modules_ordered_dict()`, it enables the
  /// following use case: `AnySequential sequential(modules_ordered_dict({{"m1",
  /// M(1)}, {"m2", M(2)}}))`
  explicit AnySequentialImpl(
      torch::OrderedDict<std::string, AnyModule>&& ordered_dict) {
    modules_.reserve(ordered_dict.size());
    for (auto& item : ordered_dict) {
      push_back(std::move(item.key()), std::move(item.value()));
    }
  }

  /// Special cloning function for `AnySequential` because it does not use
  /// `reset()`.
  std::shared_ptr<Module> clone(
      const optional<Device>& device = nullopt) const override {
    auto clone = std::make_shared<AnySequentialImpl>();
    for (const auto& module : modules_) {
      clone->push_back(module.clone(device));
    }
    return std::move(clone);
  }

  /// `reset()` is empty for `AnySequential`, since it does not have parameters
  /// of its own.
  void reset() override {}

  /// Pretty prints the `AnySequential` module into the given `stream`.
  virtual void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::AnySequential";
  }

  /// Feeds `inputs` to the first module and then chains outputs to inputs,
  /// returning the last output.
  ///
  /// Conceptually the following loop in Python:
  ///
  /// \rst
  /// .. code-block:: python
  ///
  ///   def forward(sequential, input):
  ///     for module in sequential:
  ///       input = module(input)
  ///     return input
  ///
  /// \endrst
  ///
  /// The return type is taken as the first template parameter. It defaults to
  /// `Tensor`. If the last module in the `AnySequential` returns another type
  /// `T`, you should call `forward<T>(inputs)` instead of just
  /// `forward(inputs)`:
  ///
  /// \rst
  /// .. code-block:: cpp
  ///
  ///   torch::Tensor tensor = sequential1->forward(inputs);
  ///   int integer = sequential2->forward<int>(inputs);
  ///   float value = sequential3->forward<float>(inputs);
  ///
  /// \endrst
  template <typename ReturnType = Tensor, typename... InputTypes>
  ReturnType forward(InputTypes&&... inputs) {
    TORCH_CHECK(!is_empty(), "Cannot call forward() on an empty AnySequential");

    auto iterator = modules_.begin();
    auto input = iterator->any_forward(std::forward<InputTypes>(inputs)...);

    for (++iterator; iterator != modules_.end(); ++iterator) {
      input = iterator->any_forward(std::move(input));
    }

    // Check the return value and give a nice error message if the requsted
    // return type was incorrect.
    if (auto* return_value = input.template try_get<ReturnType>()) {
      return std::move(*return_value);
    }
    AT_ERROR(
        "The type of the return value is ",
        c10::demangle(input.type_info().name()),
        ", but you asked for type ",
        c10::demangle(typeid(ReturnType).name()));
  }

  /// Adds a new (boxed) `Module` to the `AnySequential` container.
  template <typename ModuleType>
  void push_back(std::shared_ptr<ModuleType> module_ptr) {
    push_back(std::to_string(modules_.size()), std::move(module_ptr));
  }

  /// Adds a new named (boxed) `Module` to the `AnySequential` container.
  template <typename ModuleType>
  void push_back(std::string name, std::shared_ptr<ModuleType> module_ptr) {
    push_back(std::move(name), AnyModule(std::move(module_ptr)));
  }

  /// Adds a new `Module` to the `AnySequential` container, moving or copying it
  /// into a `shared_ptr` internally. This method allows passing value types,
  /// and letting the container deal with the boxing. This means you can write
  /// `AnySequential(Module(3, 4))` instead of
  /// `AnySequential(std::make_shared<Module>(3, 4))`.
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  void push_back(M&& module) {
    push_back(std::to_string(modules_.size()), std::forward<M>(module));
  }

  /// Adds a new named `Module` to the `AnySequential` container, moving or
  /// copying it into a `shared_ptr` internally. This method allows passing
  /// value types, and letting the container deal with the boxing.
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  void push_back(std::string name, M&& module) {
    using Type = typename std::remove_reference<M>::type;
    push_back(std::move(name), std::make_shared<Type>(std::forward<M>(module)));
  }

  /// Unwraps the contained module of a `ModuleHolder` and adds it to the
  /// `AnySequential`.
  template <typename M>
  void push_back(const ModuleHolder<M>& module_holder) {
    push_back(std::to_string(modules_.size()), module_holder);
  }

  /// Unwraps the contained named module of a `ModuleHolder` and adds it to the
  /// `AnySequential`.
  template <typename M>
  void push_back(std::string name, const ModuleHolder<M>& module_holder) {
    push_back(std::move(name), module_holder.ptr());
  }

  /// Adds a type-erased `AnyModule` to the `AnySequential`.
  void push_back(AnyModule any_module) {
    push_back(std::to_string(modules_.size()), std::move(any_module));
  }

  void push_back(std::string name, AnyModule any_module) {
    modules_.push_back(std::move(any_module));
    const auto index = modules_.size() - 1;
    register_module(std::move(name), modules_[index].ptr());
  }

  /// Iterates over the container and calls `push_back()` on each value.
  template <typename Container>
  void extend(const Container& container) {
    for (const auto& module : container) {
      push_back(module);
    }
  }

  /// Returns an iterator to the start of the `AnySequential`.
  Iterator begin() {
    return modules_.begin();
  }

  /// Returns a const iterator to the start of the `AnySequential`.
  ConstIterator begin() const {
    return modules_.begin();
  }

  /// Returns an iterator to the end of the `AnySequential`.
  Iterator end() {
    return modules_.end();
  }

  /// Returns a const iterator to the end of the `AnySequential`.
  ConstIterator end() const {
    return modules_.end();
  }

  /// Attempts to return the module at the given index as the requested type.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  T& at(size_t index) {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call AnySequential::at with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    return modules_[index].get<T>();
  }

  /// Attempts to return the module at the given index as the requested type.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  const T& at(size_t index) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call AnySequential::at with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    return modules_[index].get<T>();
  }

  /// Attempts to return a `std::shared_ptr` whose dynamic type is that of the
  /// underlying module at the given index. Throws an exception if the index is
  /// out of bounds.
  std::shared_ptr<Module> ptr(size_t index) const {
    TORCH_CHECK(index < size(), "Index out of range");
    return modules_[index].ptr();
  }

  /// Attempts to return a `std::shared_ptr` whose type is the one provided.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  std::shared_ptr<T> ptr(size_t index) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call AnySequential::ptr with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    return modules_[index].ptr<T>();
  }

  /// Like `ptr(index)`.
  std::shared_ptr<Module> operator[](size_t index) const {
    // This is the only method we can call without a type.
    return ptr(index);
  }

  /// The current size of the `AnySequential` container.
  size_t size() const noexcept {
    return modules_.size();
  }

  /// True if there are no modules in the `AnySequential`.
  bool is_empty() const noexcept {
    return size() == 0;
  }

 private:
  /// Takes a First *and* Second parameter, to avoid ambiguity when a parameter
  /// pack has only one type, in which case the template would be preferred,
  /// even if the other `push_back` functions are better fits (e.g. `unique_ptr`
  /// -> `shared_ptr` overload).
  /// NOTE: We explicitly avoid matching this template with
  /// `push_back(std::string("name"), module)` or `push_back("name", module)`,
  /// since they should be handled by their respective `push_back` functions.
  template <
      typename First,
      typename Second,
      typename... Rest,
      typename = torch::disable_if_t<
          std::is_same<First, std::string>::value ||
          std::is_same<
              typename std::decay<First>::type,
              std::decay<const char (&)[]>::type>::value>>
  void push_back(First&& first, Second&& second, Rest&&... rest) {
    push_back(std::forward<First>(first));
    // Recursively calls this method, until the parameter pack only thas this
    // entry left. Then calls `push_back()` a final time (above).
    push_back(std::forward<Second>(second), std::forward<Rest>(rest)...);
  }

  /// The base case, when the list of modules is empty.
  void push_back() {}

  // Box the AnyModules to give AnySequential reference semantics, like the rest
  // of the API. Note that this is not required otherwise, this could just be a
  // `vector<AnyModule>`.
  std::vector<AnyModule> modules_;
};

/// A `ModuleHolder` subclass for `AnySequentialImpl`.
/// See the documentation for `AnySequentialImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AnySequential);

/// `Sequential` is derived from `AnySequential` and has a non-template
/// returning type allowing `Sequential`s to be nested in each other making them
/// compatible with the Python API. The template input type helps warn users
/// about the change in the API.
class SequentialImpl : public AnySequentialImpl {
 public:
  using AnySequentialImpl::AnySequentialImpl;

  /// Constructs the `Sequential` from an `AnySequential`.
  explicit SequentialImpl(std::shared_ptr<Module> module_ptr) {
    if (auto* any_seq_impl = module_ptr->as<AnySequentialImpl>()) {
      for(auto it = any_seq_impl->begin(); it != any_seq_impl->end(); it++) push_back(*it);
    } else {
      // yf225 TODO: fill this out!
      AT_ERROR("some error message");
    }
  }

  template <typename ReturnType = Tensor, typename... InputTypes>
  Tensor forward(InputTypes&&... inputs) {
    static_assert(
        std::is_same<ReturnType, Tensor>(),
        "Can only call Sequential::forward with Tensor as output type."
        " If you would like to have a non-Tensor output type, please use AnySequential instead.");
    return AnySequentialImpl::forward(std::forward<InputTypes>(inputs)...);
  }

  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::Sequential";
  }
};

/// A `ModuleHolder` subclass for `SequentialImpl`.
/// See the documentation for `SequentialImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Sequential);

} // namespace nn
} // namespace torch
