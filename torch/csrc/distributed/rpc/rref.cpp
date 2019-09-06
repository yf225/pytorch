#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/script_rref_proto.h>

namespace torch {
namespace distributed {
namespace rpc {

std::atomic<local_id_t> RRefContext::nextLocalId_{0};

//////////////////////////  RRefForkData  /////////////////////////////////

RRefForkData::RRefForkData(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId)
    : ownerId_(ownerId), rrefId_(rrefId), forkId_(forkId) {}

at::IValue RRefForkData::toIValue() const {
  std::vector<at::IValue> ivalues = {
      (int64_t)ownerId_, rrefId_.toIValue(), forkId_.toIValue()};

  return c10::ivalue::Tuple::create(std::move(ivalues));
}

RRefForkData RRefForkData::fromIValue(const at::IValue& ivalue) {
  auto ivalues = ivalue.toTuple()->elements();

  TORCH_CHECK(
      ivalues.size() == 3,
      "Constructing RRefForkData from ivalue "
      "expects a GenericList of 3 elements, but got ",
      ivalues.size());

  int64_t ownerId = ivalues[0].toInt();
  TORCH_CHECK(
      ownerId < std::numeric_limits<worker_id_t>::max(),
      "RRefId createdOn out of range, got ",
      ownerId);

  RRefId rrefId = RRefId::fromIValue(ivalues[1]);
  ForkId forkId = ForkId::fromIValue(ivalues[2]);

  return RRefForkData(ownerId, rrefId, forkId);
}

//////////////////////////////  RRef  /////////////////////////////////////

RRef::RRef(worker_id_t ownerId, const RRefId& rrefId)
    : ownerId_(ownerId), rrefId_(rrefId) {}

worker_id_t RRef::owner() const {
  return ownerId_;
}

const RRefId& RRef::id() const {
  return rrefId_;
}

at::IValue RRef::fork() const {
  return RRefForkData(
             ownerId_, rrefId_, RRefContext::getInstance()->genRRefId())
      .toIValue();
  // NB: does not support sharing RRefs between users
  // TODO: notify the owner
}

//////////////////////////  UserRRef  /////////////////////////////////////

UserRRef::UserRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId)
    : RRef(ownerId, rrefId), forkId_(forkId) {
  AT_ASSERT(
      !(forkId_ == rrefId_),
      "User RRef's fork ID should not be the same as its rref Id");
  if (RRefContext::getInstance()->getWorkerId() == rrefId_.createdOn_) {
    // creator user, notify owner.
    auto& agent = RRefContext::getInstance()->agent();
    agent->send(
        agent->getWorkerId(ownerId_),
        ScriptRRefCreate(RRefForkData(ownerId_, rrefId_, forkId_).toIValue())
            .toMessage());
  } else {
    AT_ERROR("Does not support sharing RRefs between users yet");
  }
}

UserRRef::~UserRRef() {
  auto& ctx = RRefContext::getInstance();
  if (ctx->getWorkerId() != ownerId_) {
    ctx->agent()->send(
        ctx->agent()->getWorkerId(ownerId_),
        ScriptRRefDelete(RRefForkData(ownerId_, rrefId_, forkId_).toIValue())
            .toMessage());
  }
}

const ForkId& UserRRef::forkId() const {
  return forkId_;
}

bool UserRRef::isOwner() const {
  return false;
}

IValue UserRRef::toHere() {
  auto& agent = RRefContext::getInstance()->agent();
  std::shared_ptr<FutureMessage> fm = agent->send(
      agent->getWorkerId(ownerId_),
      ScriptRRefFetchCall(id().toIValue()).toMessage());
  auto srv = ScriptRRefFetchRet::fromMessage(fm->wait());
  return srv.value();
}

} // namespace rpc
} // namespace distributed
} // namespace torch
