#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional avgpool functional and module.
template <size_t D>
struct AvgPoolOptions {
  AvgPoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// the size of the window to take an average over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// the stride of the window. Default value is `kernel_size`
  TORCH_ARG(ExpandingArray<D>, stride);

  /// implicit zero padding to be added on both sides
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// when True, will use `ceil` instead of `floor` to compute the output shape
  TORCH_ARG(bool, ceil_mode) = false;

  /// when True, will include the zero-padding in the averaging calculation
  TORCH_ARG(bool, count_include_pad) = true;

  /// if specified, it will be used as divisor, otherwise `kernel_size` will be used
  TORCH_ARG(c10::optional<int64_t>, divisor_override) = c10::nullopt;
};

/// `AvgPoolOptions` specialized for 1-D avgpool.
using AvgPool1dOptions = AvgPoolOptions<1>;

/// `AvgPoolOptions` specialized for 2-D avgpool.
using AvgPool2dOptions = AvgPoolOptions<2>;

/// `AvgPoolOptions` specialized for 3-D avgpool.
using AvgPool3dOptions = AvgPoolOptions<3>;

// ============================================================================

/// Options for a `D`-dimensional maxpool functional and module.
template <size_t D>
struct MaxPoolOptions {
  MaxPoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// the size of the window to take a max over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// the stride of the window. Default value is `kernel_size
  TORCH_ARG(ExpandingArray<D>, stride);

  /// implicit zero padding to be added on both sides
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// a parameter that controls the stride of elements in the window
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// if true, will return the max indices along with the outputs. Useful
  /// for `MaxUnpool1d` later
  TORCH_ARG(bool, return_indices) = false;

  /// when True, will use `ceil` instead of `floor` to compute the output shape
  TORCH_ARG(bool, ceil_mode) = false;
};

/// `MaxPoolOptions` specialized for 1-D maxpool.
using MaxPool1dOptions = MaxPoolOptions<1>;

/// `MaxPoolOptions` specialized for 2-D maxpool.
using MaxPool2dOptions = MaxPoolOptions<2>;

/// `MaxPoolOptions` specialized for 3-D maxpool.
using MaxPool3dOptions = MaxPoolOptions<3>;

// ============================================================================

/// Options for a `D`-dimensional MaxUnpool functional and module.
template <size_t D>
struct MaxUnpoolOptions {
  MaxUnpoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// Size of the max pooling window.
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// Stride of the max pooling window. It is set to `kernel_size` by default.
  TORCH_ARG(ExpandingArray<D>, stride);

  /// Padding that was added to the input.
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// The targeted output size.
  TORCH_ARG(c10::Optional<ExpandingArray<D>>, output_size);
};

/// `MaxUnpoolOptions` specialized for 1-D MaxUnpool.
using MaxUnpool1dOptions = MaxUnpoolOptions<1>;

/// `MaxUnpoolOptions` specialized for 2-D MaxUnpool.
using MaxUnpool2dOptions = MaxUnpoolOptions<2>;

/// `MaxUnpoolOptions` specialized for 3-D MaxUnpool.
using MaxUnpool3dOptions = MaxUnpoolOptions<3>;

} // namespace nn
} // namespace torch
