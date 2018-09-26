#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/variable_version.h"

#include <ATen/ATen.h>
#include <ATen/core/Error.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch {
namespace autograd {
Variable::Impl::Impl(bool requires_grad, Edge gradient_edge)   // yf225 TODO: do we need to call VariableImplInterface's default constructor here?
    : grad_fn_(std::move(gradient_edge.function)),
      requires_grad_(requires_grad),
      is_view_(false),
      output_nr_(gradient_edge.input_nr),
      pyobj_(nullptr) {
  AT_CHECK(
      !grad_fn_ || !requires_grad_,
      "requires_grad should be false if grad_fn is set");
}

Variable::Impl::~Impl() = default;

void Variable::Impl::release_resources() {
  grad_.reset();
  grad_fn_.reset();
  hooks_.clear();
}

Variable::ViewImpl::ViewImpl(Variable base, Edge gradient_edge)
    : Variable::Impl(false, std::move(gradient_edge)),
      base_(std::move(base)) {
  AT_CHECK(base_.defined(), "base is undefined");
  if (base_.is_view()) {
    base_ = base_.base();
  }
  is_view_ = true;
  version_counter_ = base_.version_counter();
  attr_version_ = version_counter_.current_version();
}

void Variable::ViewImpl::release_resources() {
  Variable::Impl::release_resources();
  base_.reset();
}

std::shared_ptr<Function> Variable::grad_accumulator() const {
  if (get()->grad_fn_) {
    throw std::logic_error(
        "get_grad_accumulator() should be only called on leaf Variables");
  }
  if (!get()->requires_grad_) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(get()->mutex_);

  auto result = get()->grad_accumulator_.lock();
  if (result)
    return result;

  // yf225 TODO: the refcount here could be wrong
  auto intrusive_from_this = c10::intrusive_ptr<at::TensorImpl>(this->getIntrusivePtr());
  result = std::make_shared<AccumulateGrad>(Variable(std::move(intrusive_from_this)));
  get()->grad_accumulator_ = result;
  return result;
}

void Variable::backward(
    at::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) const {
  std::vector<Edge> edges;
  edges.emplace_back(get()->grad_fn_, get()->output_nr_);

  std::vector<Variable> inputs;
  if (!gradient.has_value()) {
    gradient = make_variable(at::ones_like(*this), /*requires_grad=*/false);
  }
  inputs.push_back(std::move(as_variable_ref(*gradient)));
  Engine::get_default_engine().execute(edges, inputs, keep_graph, create_graph);
}

void Variable::rebase_history(Edge gradient_edge) {
  AT_ASSERT(gradient_edge.function != nullptr);
  if (is_view()) {
    /// Called after in-place modifications. Modifies the grad_fn of the base
    /// Variable.
    auto& impl = static_cast<Variable::ViewImpl&>(*get());
    gradient_edge = std::move(gradient_edge);

    AT_ASSERT(gradient_edge.input_nr == 0);
    AT_ASSERT(gradient_edge.function);
    AT_CHECK(
        gradient_edge.function->num_inputs() == 1,
        "Functions which modify views in-place must return a single Variable");
    impl.output_nr_ = gradient_edge.input_nr;
    auto copy_slices = std::make_shared<CopySlices>(
        impl.base_, at::TensorGeometry(*this), std::move(gradient_edge.function));
    impl.base_.set_gradient_edge(Edge(std::move(copy_slices), 0));
    grad_fn(); // trigger an update to the view's grad_fn
  } else {
    set_gradient_edge(std::move(gradient_edge));
  }
}

// NOTE: For View on Variable, this gets the up-to-date grad_fn. If the shared data or base was modified, we
// re-create the grad_fn to express the up-to-date view relationship between this and the base Variable.
const std::shared_ptr<Function>& Variable::grad_fn() const {
  if (is_view()) {
    auto& impl = static_cast<Variable::ViewImpl&>(*get());
    std::lock_guard<std::mutex> lock(impl.mutex_);
    if (!impl.grad_fn_ && !impl.base_.requires_grad()) {
      return impl.grad_fn_;
    }
    auto current_version = impl.version_counter_.current_version();
    if (impl.attr_version_ != current_version) {
      AT_ASSERT(impl.output_nr_ == 0);
      auto fn = std::make_shared<generated::AsStridedBackward>();
      fn->self_geometry = at::TensorGeometry(impl.base_);
      fn->size = sizes().vec();
      fn->stride = strides().vec();
      fn->storage_offset = storage_offset();
      fn->set_next_edges(collect_next_edges(impl.base_));
      fn->add_input_metadata(
        impl.base_.type()
      , sizes() // Note: sizes(), not base_.sizes(), is intentional
      , impl.base_.is_cuda() ? impl.base_.get_device() : -1);
      impl.grad_fn_ = std::move(fn);
      impl.attr_version_ = current_version;
    }
    return impl.grad_fn_;
  } else {
    return get()->grad_fn_;
  }
}

void Variable::set_data(Tensor new_data) {
  // Resets gradient accumulator if metadata is out of date
  std::lock_guard<std::mutex> lock(get()->mutex_);
  auto prior_accumulator = get()->grad_accumulator_.lock();
  if (prior_accumulator) {
    const auto prior_device = prior_accumulator->input_metadata(0).device();
    const auto new_device = new_data.is_cuda() ? new_data.get_device() : -1;

    if (new_data.type() != type() || prior_device != new_device) {
      get()->grad_accumulator_.reset();
    }
  }

  // yf225 TODO: the logic here could be wrong

  // Updates metadata
  auto new_tensor_impl = new_data.getIntrusivePtr()->clone();
  AT_ASSERT(new_tensor_impl.use_count() == 1);
  new_tensor_impl->set_variable_impl(tensor_impl_->get_variable_impl());
  std::cout << "Variable::set_data: new_tensor_impl->get_variable_impl().use_count(): " << new_tensor_impl->get_variable_impl().use_count() << "\n";
  new_tensor_impl->set_is_variable(true);
  tensor_impl_ = new_tensor_impl;
  AT_ASSERT(tensor_impl_->is_variable());
}

}} // namespace torch::autograd
