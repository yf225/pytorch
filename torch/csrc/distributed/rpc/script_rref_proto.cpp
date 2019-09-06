#include <torch/csrc/distributed/rpc/script_rref_proto.h>
#include <torch/csrc/jit/pickle.h>

namespace torch {
namespace distributed {
namespace rpc {

const at::IValue& RRefMessageBase::value() {
  return value_;
}

at::IValue& RRefMessageBase::valueRef() {
  return value_;
}

Message RRefMessageBase::toMessage() const {
  std::vector<at::IValue> ivalues;
  ivalues.push_back(value_);
  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  return Message(std::move(payload), std::move(tensor_table), type_);
}

at::IValue RRefMessageBase::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value =
      jit::unpickle(payload, payload_size, nullptr, &message.tensors());
  auto values = value.toTuple()->elements();

  AT_ASSERT(values.size() == 1, "Expect a single IValue from message.");
  return std::move(values.front());
}

ScriptRRefFetchCall ScriptRRefFetchCall::fromMessage(const Message& message) {
  return ScriptRRefFetchCall(RRefMessageBase::fromMessage(message));
}

ScriptRRefFetchRet ScriptRRefFetchRet::fromMessage(const Message& message) {
  return ScriptRRefFetchRet(RRefMessageBase::fromMessage(message));
}

ScriptRRefCreate ScriptRRefCreate::fromMessage(const Message& message) {
  return ScriptRRefCreate(RRefMessageBase::fromMessage(message));
}

ScriptRRefDelete ScriptRRefDelete::fromMessage(const Message& message) {
  return ScriptRRefDelete(RRefMessageBase::fromMessage(message));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
