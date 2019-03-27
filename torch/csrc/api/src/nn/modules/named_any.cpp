#include <torch/nn/modules/named_any.h>

namespace torch {
namespace nn {

torch::OrderedDict<std::string, AnyModule> modules_ordered_dict(
  std::initializer_list<std::pair<std::string, AnyModule>> named_modules) {
  torch::OrderedDict<std::string, AnyModule> dict;
  for (const auto& named_module : named_modules) {
    dict.insert(named_module.first, std::move(named_module.second));
  }
  return dict;
}

} // namespace nn
} // namespace torch
