#pragma once

#include <memory>
#include <c10/macros/Macros.h>

namespace at {
class Tensor;
}

namespace c10 {

struct TensorImpl;

struct C10_API AutogradMetaInterface {
  virtual void set_requires_grad(bool requires_grad, c10::TensorImpl* self_impl) = 0;
  virtual bool requires_grad() const = 0;
  virtual at::Tensor& grad() = 0;
  virtual const at::Tensor& grad() const = 0;
  virtual ~AutogradMetaInterface();
};

struct C10_API AutogradMetaFactory {

  static std::unique_ptr<c10::AutogradMetaInterface> create_something();

};

}
