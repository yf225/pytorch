#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/pickler.h>


namespace torch {
namespace jit {

void pickle(
    std::function<void(const char*, size_t)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  Pickler pickler(std::move(writer), tensor_table);

  if (tensor_table == nullptr) {
    // No tensor table provided, so tensors will be stored directly in the blob.
    // Add torch.save metadata so these tensors can be de-serialized later
    pickler.torchSaveStart();
  }

  pickler.protocol();
  pickler.pushIValue(ivalue);
  pickler.stop();

  if (tensor_table == nullptr) {
    // No tensor table provided, so tensors will be stored directly in the blob.
    // Add torch.save metadata so these tensors can be de-serialized later
    pickler.torchSaveStop();
  }
}

std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  std::vector<char> data;

  pickle(
      [&](const char* bytes, size_t len) {
        data.insert(data.end(), bytes, bytes + len);
      },
      ivalue,
      tensor_table);

  return data;
}

IValue unpickle(
    std::function<bool(char*, size_t)> reader,
    ClassResolver class_resolver,
    const std::vector<at::Tensor>* tensor_table) {
  Unpickler unpickler(
      std::move(reader), std::move(class_resolver), tensor_table);
  return unpickler.parse_ivalue();
}

IValue unpickle(
    const char* data,
    size_t size,
    ClassResolver class_resolver,
    const std::vector<at::Tensor>* tensor_table) {
  size_t bytes_read = 0;
  return unpickle(
      [&](char* buffer, size_t len) {
        if (bytes_read + len > size) {
          return false;
        }
        // Copy len bytes into buffer
        const char* start = data + bytes_read;
        std::memcpy(buffer, start, len);
        bytes_read += len;
        return true;
      },
      std::move(class_resolver),
      tensor_table);
}

} // namespace jit
} // namespace torch
