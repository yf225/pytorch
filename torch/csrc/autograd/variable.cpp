#include <torch/csrc/autograd/variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/generated/Functions.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/variable_version.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch {
namespace autograd {
Variable::AutogradMeta::AutogradMeta(at::TensorImpl* self_impl, bool requires_grad, Edge gradient_edge) {
  grad_fn_ = std::move(gradient_edge.function);
  requires_grad_ = false;
  is_view_ = false;
  output_nr_ = gradient_edge.input_nr;
  pyobj_ = nullptr;

  // set_requires_grad also checks error conditions.
  set_requires_grad(requires_grad, self_impl);
  AT_CHECK(
      !grad_fn_ || !requires_grad_,
      "requires_grad should be false if grad_fn is set");
}

std::shared_ptr<Function> Variable::grad_accumulator() const {
  auto autograd_meta = get_autograd_meta();
  if (autograd_meta->grad_fn_) {
    throw std::logic_error(
        "grad_accumulator() should be only called on leaf Variables");
  }
  if (!autograd_meta->requires_grad_) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(autograd_meta->mutex_);

  auto result = autograd_meta->grad_accumulator_.lock();
  if (result)
    return result;

  c10::raw::intrusive_ptr::incref(unsafeGetTensorImpl());
  auto intrusive_from_this = c10::intrusive_ptr<at::TensorImpl>::reclaim(unsafeGetTensorImpl());
  result = std::make_shared<AccumulateGrad>(Variable(std::move(intrusive_from_this)));
  autograd_meta->grad_accumulator_ = result;
  return result;
}

void Variable::detach_() {
  if (is_view()) {
    AT_ERROR("Can't detach views in-place. Use detach() instead");
  }
  auto autograd_meta = get_autograd_meta();
  autograd_meta->set_requires_grad(false, unsafeGetTensorImpl());
  autograd_meta->grad_fn_.reset();
  autograd_meta->output_nr_ = 0;
}

void Variable::backward(
    c10::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) const {
  auto autograd_meta = get_autograd_meta();
  std::vector<Edge> edges;
  edges.emplace_back(autograd_meta->grad_fn_, autograd_meta->output_nr_);

  std::vector<Variable> inputs;
  if (!gradient.has_value()) {
    gradient = make_variable(at::ones_like(*this), /*requires_grad=*/false);
  }
  inputs.push_back(std::move(as_variable_ref(*gradient)));
  Engine::get_default_engine().execute(edges, inputs, keep_graph, create_graph);
}

void Variable::set_data(Tensor new_data) const {
  AT_CHECK(is_sparse() == new_data.is_sparse(), "expected new_data.is_sparse() to be ", is_sparse(), ", but got ", new_data.is_sparse());

  // Resets gradient accumulator if metadata is out of date
  Variable::AutogradMeta* autograd_meta = get_autograd_meta();
  std::lock_guard<std::mutex> lock(autograd_meta->mutex_);
  auto prior_accumulator = autograd_meta->grad_accumulator_.lock();
  if (prior_accumulator) {
    const auto prior_device = prior_accumulator->input_metadata(0).device();
    const auto new_device = new_data.is_cuda() ? new_data.get_device() : -1;

    if (new_data.type() != type() || prior_device != new_device) {
      autograd_meta->grad_accumulator_.reset();
    }
  }

  bool allow_tensor_metadata_change = get()->allow_tensor_metadata_change();
  get()->set_allow_tensor_metadata_change(true);
  get()->shallow_copy_from(new_data.getIntrusivePtr()->shallow_copy_and_detach());
  get()->set_allow_tensor_metadata_change(allow_tensor_metadata_change);

  get()->set_autograd_meta(std::move(get()->detach_autograd_meta()));
  get()->set_is_variable(true);
}

Variable::DifferentiableViewMeta::DifferentiableViewMeta(at::TensorImpl* self_impl, Variable base, Edge gradient_edge)
    : Variable::AutogradMeta(self_impl, false, std::move(gradient_edge)) {
  base_ = std::move(base);
  AT_CHECK(base_.defined(), "base is undefined");
  if (base_.is_view()) {
    base_ = base_.base();
  }
  is_view_ = true;
  version_counter_ = base_.version_counter();
  attr_version = version_counter_.current_version();
}

const std::shared_ptr<Function>& Variable::grad_fn() const {
  if (is_view()) {
    auto diff_view_meta = static_cast<Variable::DifferentiableViewMeta*>(get_autograd_meta());
    std::lock_guard<std::mutex> lock(diff_view_meta->mutex_);
    if (!diff_view_meta->grad_fn_ && !diff_view_meta->base_.requires_grad()) {
      return diff_view_meta->grad_fn_;
    }
    auto current_version = diff_view_meta->version_counter_.current_version();
    if (diff_view_meta->attr_version != current_version) {
      AT_ASSERT(diff_view_meta->output_nr_ == 0);
      auto fn = std::make_shared<generated::AsStridedBackward>();
      fn->self_geometry = at::TensorGeometry(diff_view_meta->base_);
      fn->size = sizes().vec();
      fn->stride = strides().vec();
      fn->storage_offset = storage_offset();
      fn->set_next_edges(collect_next_edges(diff_view_meta->base_));
      fn->add_input_metadata(
        diff_view_meta->base_.type()
      , sizes() // Note: sizes(), not base_.sizes(), is intentional
      , diff_view_meta->base_.is_cuda() ? diff_view_meta->base_.get_device() : -1);
      diff_view_meta->grad_fn_ = std::move(fn);
      diff_view_meta->attr_version = current_version;
    }
    return diff_view_meta->grad_fn_;
  } else {
    return get_autograd_meta()->grad_fn_;
  }
}

void Variable::rebase_history(Edge gradient_edge) {
  AT_ASSERT(gradient_edge.function != nullptr);
  if (is_view()) {
    auto diff_view_meta = static_cast<Variable::DifferentiableViewMeta*>(get_autograd_meta());
    AT_ASSERT(gradient_edge.input_nr == 0);
    AT_ASSERT(gradient_edge.function);
    AT_CHECK(
        gradient_edge.function->num_inputs() == 1,
        "Functions which modify views in-place must return a single Variable");
    diff_view_meta->output_nr_ = gradient_edge.input_nr;
    auto copy_slices = std::make_shared<CopySlices>(
        diff_view_meta->base_, at::TensorGeometry(*this), std::move(gradient_edge.function));
    diff_view_meta->base_.set_gradient_edge({std::move(copy_slices), 0});
    grad_fn(); // trigger an update to the view's grad_fn
  } else {
    set_gradient_edge(std::move(gradient_edge));
  }
}

}} // namespace torch::autograd
