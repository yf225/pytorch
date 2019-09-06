#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/jit/pickle.h>

namespace torch {
namespace distributed {
namespace rpc {

ScriptRemoteCall::ScriptRemoteCall(
    std::shared_ptr<Operator> op,
    std::vector<at::IValue>&& args,
    at::IValue retRRefId,
    at::IValue retForkId)
    : ScriptCall(std::move(op), std::move(args)),
      retRRefId_(std::move(retRRefId)),
      retForkId_(std::move(retForkId)) {}

const at::IValue& ScriptRemoteCall::retRRefId() {
  return retRRefId_;
}

const at::IValue& ScriptRemoteCall::retForkId() {
  return retForkId_;
}

Message ScriptRemoteCall::toMessage() const {
  std::vector<IValue> ivalues;
  ScriptCall::toIValues(ivalues);
  ivalues.push_back(retRRefId_);
  ivalues.push_back(retForkId_);

  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  return Message(
      std::move(payload), std::move(tensor_table), MessageType::REMOTE_CALL);
}

ScriptRemoteCall ScriptRemoteCall::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value =
      jit::unpickle(payload, payload_size, nullptr, &message.tensors());
  auto values = value.toTuple()->elements();

  // remove the last element from values and convert it back to an RRef
  auto retForkId = std::move(values.back());
  values.pop_back();
  auto retRRefId = std::move(values.back());
  values.pop_back();

  auto op = ScriptCall::fromIValues(values);
  return ScriptRemoteCall(
      op, std::move(values), std::move(retRRefId), std::move(retForkId));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
