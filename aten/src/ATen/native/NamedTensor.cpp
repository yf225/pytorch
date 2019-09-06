#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/NamedTensorUtils.h>

#ifdef BUILD_NAMEDTENSOR
namespace at { namespace native {

Tensor& names_(Tensor& self, optional<DimnameList> names) {
  return at::internal_set_names_inplace(self, names);
}

Tensor renamed(const Tensor& self, optional<DimnameList> names) {
  auto result = self.alias();
  at::internal_set_names_inplace(result, names);
  return result;
}

static void report_moving_unnamed_dim_error(
    DimnameList names, DimnameList other, bool is_aligning_two_tensors) {
  if (is_aligning_two_tensors) {
    TORCH_CHECK(false,
        "Aligning Tensor", names, " and Tensor", other,
        " would change the absolute position from the right of an unnamed dimension. ",
        "Please name unnamed dimensions to avoid ambiguity.");
  } else {
    TORCH_CHECK(false,
        "Aligning Tensor", names, " to `names` ", other,
        " would change the absolute position from the right of an unnamed dimension. ",
        "Please name unnamed dimensions to avoid ambiguity.");
  }
}

static void report_not_a_subsequence_error(
    DimnameList names, DimnameList other, bool is_aligning_two_tensors) {
  if (is_aligning_two_tensors) {
    auto shorter = names.size() > other.size() ? other : names;
    auto longer = names.size() > other.size() ? names : other;
    TORCH_CHECK(false,
        "Could not align Tensor", shorter, " and Tensor", longer,
        " because ", shorter, " is not a subsequence of ", longer, ". ");
  } else {
    TORCH_CHECK(false,
        "Could not align Tensor", names, " to `names` ", other,
        " because ", names, " is not a subsequence of `names`.");
  }
}


// Let tensor `t` have size `tensor_sizes` and `tensor_names`.
// This helper function computes the resulting size of `t` after aligning it
// to `aligned_names`. Enforces the alignment rules in Note [Alignment rules].
static std::vector<int64_t> aligned_size(
    IntArrayRef tensor_sizes,
    DimnameList tensor_names,
    DimnameList aligned_names,
    bool is_aligning_two_tensors) {
  std::vector<int64_t> expanded_sizes(aligned_names.size(), 1);
  ptrdiff_t dim = (ptrdiff_t)tensor_sizes.size() - 1;
  ptrdiff_t idx = (ptrdiff_t)aligned_names.size() - 1;
  for (; idx >= 0 && dim >= 0; --idx) {
    TORCH_INTERNAL_ASSERT(!tensor_names[dim].is_tagged() && !aligned_names[idx].is_tagged(), "Tagged names NYI");
    if (tensor_names[dim] != aligned_names[idx]) {
      continue;
    }
    // We've found a None name in `shorter` and `longer`. If their absolute positions
    // from the right are not equal, then aligning the two names would require
    // changing the absolute position from right of one of the None names,
    // violating condition 2 of our [Alignment rules].
    //
    // For example:
    // *, c, a, b
    //       *, a
    // [*, a] is a subsequence of [*, c, a, b], but in order to align them,
    // we'd have to move the * to create [*, c: 1, a, b: 1]
    if (tensor_names[dim].is_wildcard() &&
        tensor_sizes.size() - dim != aligned_names.size() - idx) {
      report_moving_unnamed_dim_error(
          tensor_names, aligned_names, /*is_aligning_two_tensors=*/false);
    }
    expanded_sizes[idx] = tensor_sizes[dim];
    --dim;
  }
  if (dim != -1) {
    report_not_a_subsequence_error(
        tensor_names, aligned_names, /*is_aligning_two_tensors=*/false);
  }

  return expanded_sizes;
}

// [Alignment rules]
// Aligns `tensor` to names with the following rules:
// 1) Check that tensor.names is a subsequence (not necessarily contiguous) of `names`.
// 2) Aligning tensor.names to names must not change the absolute position from the
//    right of any unnamed dimension.
//
// is_aligning_two_tensors tunes the error message to better match the following cases:
// 1) tensor.align_to(names)  (is_aligning_two_tensors=false)
// 2) torch.align_tensors([tensor, other])  (is_aligning_two_tensors=true)
static Tensor align(const Tensor& tensor, DimnameList names, bool is_aligning_two_tensors) {
  std::vector<int64_t> expanded_sizes = aligned_size(
        tensor.sizes(),
        tensor.names(),
        names,
        is_aligning_two_tensors);
  auto result = tensor.renamed(nullopt).view(expanded_sizes);
  at::internal_set_names_inplace(result, names);
  return result;
}

Tensor align_to(const Tensor& tensor, DimnameList names) {
  TORCH_CHECK(
      names.size() >= tensor.dim(),
      "Cannot align tensor with dims ",
      tensor.names(),
      " to a shorter list of dims ", names, ".");
  return align(tensor, names, /*aligning_two_tensors=*/false);
}

static std::vector<Tensor> align_tensors_to(TensorList tensors, DimnameList names) {
  std::vector<Tensor> result;
  result.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    result.emplace_back(align(tensor, names, /*is_aligning_two_tensors=*/true));
  }
  return result;
}

std::vector<Tensor> align_tensors(TensorList tensors) {
  auto longest_dim = std::max_element(
      tensors.begin(), tensors.end(),
      [](const Tensor& a, const Tensor& b) {
        return a.dim() < b.dim();
      });
  return align_tensors_to(tensors, longest_dim->names());
}

}}  // namespace at::native
#endif
