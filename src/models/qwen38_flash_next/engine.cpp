#include "src/models/qwen38_flash_next/engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

#include <cstdio>
#include <cstdlib>
#include <gufo/ext/gguf_reader.hpp>
#include <limits>
#include <optional>
#include <type_traits>

#include "src/families/qwen/draft_vocab/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/continuation_hooks.hpp"
#include "src/models/qwen38_flash_next/distributed/tp_partition.hpp"
#include "src/families/qwen/draft_vocab/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/communicator.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

// gfx1151, served repository prompts: 4096-token chunks prefill 24K-102K
// tokens 6-11% faster than 2048 (pp4096 +5% at d0, +6% at d32K). 16,384-row
// passes prefill long prompts 3-5% faster, but their scratch holds 4.1 GiB
// more device memory from load (#952). Each chunk's n-gram gather grows with
// it; next-chunk prefetch overlaps that read. Prefill logits do not depend on
// the chunk size (pre-budget queries always take the dense attention tiles).
constexpr std::uint32_t kPrefillChunkTokens = 16384;
// TP2 exchanges a whole batch's partial sums through one RDMA window (48 MiB,
// about 4,900 rows of 2,560 floats) and overlaps two batches a layer apart,
// so its ranks keep 4,096-token chunks.
constexpr std::uint32_t kTpPrefillChunkTokens = 4096;
// A prompt ending this close past a chunk finishes in that chunk: a separate
// tail pass costs about as much as this many more chunk rows.
constexpr std::uint32_t kPrefillTailTokens = 128;

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

constexpr std::array<char, 8> kSessionSnapshotMagic{'Q', 'F', 'N', 'S',
                                                    'E', 'S', 'S', '1'};

/// Host-side session fields ahead of the executor payload: the token
/// history and the logits of the last token.
struct SessionSnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t vocab_size;
  std::uint32_t token_count;
  std::uint32_t hidden_rows;
  std::uint64_t executor_bytes;
  MtpLengthState draft_policy;
  std::uint32_t image_identity_bytes;
  std::uint32_t policy_concurrency;
  /// Dynamic YaRN factor the saved keys are rotated with (1: native).
  float yarn_factor{1.0F};
  speculative::PromptLookup<>::State lookup;
};
static_assert(std::is_trivially_copyable_v<SessionSnapshotHeader>);

std::uint64_t SessionSnapshotHostBytes(std::uint32_t token_count,
                                       std::uint32_t vocab_size,
                                       std::size_t image_bytes) {
  return sizeof(SessionSnapshotHeader) + image_bytes +
         std::uint64_t{token_count} * sizeof(std::int32_t) +
         std::uint64_t{vocab_size} * sizeof(float);
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.decode_concurrency == 0 || options.decode_concurrency > 8) {
    AssignError(error_msg, "decode concurrency must be between one and eight");
    return nullptr;
  }
  if (options.hip_device < 0) {
    AssignError(error_msg, "HIP device index must be nonnegative");
    return nullptr;
  }
  if (options.tp_world_size != 1 && options.tp_world_size != 2) {
    AssignError(error_msg,
                "Flash-Next currently supports TP world sizes one or two");
    return nullptr;
  }
  if (options.tp_world_size == 1 && options.tp_rank != 0) {
    AssignError(error_msg, "single-rank Flash-Next requires rank zero");
    return nullptr;
  }
  if (options.tp_world_size > 1 && !options.communicator) {
    AssignError(error_msg, "TP=2 Flash-Next requires a communicator");
    return nullptr;
  }
  if (options.tp_world_size > 1 &&
      (options.communicator->world_size() != options.tp_world_size ||
       options.communicator->rank() != options.tp_rank ||
       options.communicator->device_index() != options.hip_device)) {
    AssignError(error_msg,
                "TP communicator identity does not match model rank/device");
    return nullptr;
  }
  if (hipSetDevice(options.hip_device) != hipSuccess) {
    AssignError(error_msg, "HIP device selection failed");
    return nullptr;
  }
  if (options.max_draft_tokens == 0) {
    AssignError(error_msg, "draft token limit must be positive");
    return nullptr;
  }
  if (!options.mtp_model_path.empty() &&
      options.max_draft_tokens > kMaxMtpDraftTokens) {
    AssignError(error_msg,
                "Flash-Next MTP supports at most seven draft tokens");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  if (options.rope_scaling.has_value()) {
    if (!ValidateRopeScaling(*options.rope_scaling,
                             m->weights_->config.context_length, error_msg)) {
      return nullptr;
    }
    m->weights_->config.rope_scaling = *options.rope_scaling;
  }
  const Config& c = m->weights_->config;
  std::optional<distributed::TpPartition> partition;
  const distributed::TpPartition* partition_ptr = nullptr;
  if (options.tp_world_size > 1) {
    partition = distributed::TpPartition::Create(
        c.expert_ff, options.tp_rank, options.tp_world_size, error_msg);
    if (!partition) {
      return nullptr;
    }
    partition_ptr = &*partition;
  }
  if (options.max_context == 0 || options.max_context > c.MaxContextLength()) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.MaxContextLength()) +
                               " tokens");
    return nullptr;
  }
  // Under TP2 each rank runs the whole (replicated) vision encoder itself.
  try {
    m->vision_ = qwen::vision::Encoder::Open(
        model_path, options.vision_model_path, c.hidden_size);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return nullptr;
  }
  m->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  if (c.ple_layer >= 0) {
    const auto& t = m->weights_->ple_table;
    m->ngram_ = NgramTable::Open(
        m->reader_->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, c.ple_head_dim, t.type, error_msg);
    if (!m->ngram_) {
      return nullptr;
    }
  }
  if (!options.mtp_model_path.empty()) {
    m->mtp_reader_ =
        core::GgufReader::OpenFile(options.mtp_model_path, error_msg);
    if (!m->mtp_reader_) {
      return nullptr;
    }
    auto mtp = MtpWeights::Bind(*m->mtp_reader_, c, error_msg);
    if (!mtp) {
      return nullptr;
    }
    m->mtp_weights_ = std::make_unique<MtpWeights>(std::move(*mtp));
  }
  m->device_ = rocm::DeviceModel::Upload(
      *m->weights_, *m->reader_, m->mtp_weights_.get(), m->mtp_reader_.get(),
      error_msg, partition_ptr);
  if (!m->device_) {
    return nullptr;
  }
  if (m->mtp_weights_ &&
      m->device_->output().rows > speculative::kDraftVocabPrefix) {
    std::vector<std::int32_t> specials;
    for (std::uint32_t id = speculative::kDraftVocabPrefix;
         id < m->device_->output().rows; ++id)
      if (m->tokenizer_->IsSpecialToken(id))
        specials.push_back(static_cast<std::int32_t>(id));
    // A vocabulary with more special tokens than table rows drafts over the
    // full head.
    if (specials.size() < speculative::kDraftVocabExtraRows) {
      m->draft_specials_ = static_cast<std::uint32_t>(specials.size());
      if (!m->device_->SetDraftVocab(speculative::kDraftVocabPrefix,
                                     std::move(specials), error_msg))
        return nullptr;
      m->draft_subset_ = true;
    }
  }
  rocm::Executor::Options exec;
  exec.device_index = options.hip_device;
  exec.max_batch = m->PrefillThroughCapacity();
  exec.max_logit_rows =
      m->mtp_weights_
          ? static_cast<std::uint32_t>(std::min<std::uint64_t>(
                exec.max_batch, std::uint64_t{options.max_draft_tokens} + 1))
          : 1;
  exec.max_speculative = exec.max_logit_rows;
  if (options.communicator) {
    // One exchange carries a whole prefill batch's partial sums.
    if (std::size_t{exec.max_batch} * c.hidden_size * sizeof(float) >
        options.communicator->MaxPartialBytes()) {
      AssignError(error_msg, "a prefill batch exceeds the TP exchange window");
      return nullptr;
    }
    exec.all_reduce =
        rocm::Executor::TwoRankAllReduce(options.communicator, c.hidden_size);
    exec.reduce_status =
        rocm::Executor::TwoRankReduceStatus(options.communicator);
    exec.split_reduce =
        rocm::Executor::TwoRankSplitReduce(options.communicator, c.hidden_size);
  }
  exec.moe_observer = options.moe_observer;
  exec.max_context = options.max_context;
  m->executor_ =
      rocm::Executor::Create(*m->device_, m->ngram_.get(), exec, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(core::SessionMode mode,
                                              std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native = executor_->CreateSession(mode, max_context, error_msg);
  if (!native) {
    return nullptr;
  }
  if (vision_) {
    // Image-capable slots can first serve text. Prepare the stable RoPE
    // descriptor before any graphs capture a null pointer: replacing it
    // later destroys those graphs during image admission (5–12 ms here).
    try {
      native->PrepareVision(executor_->stream());
    } catch (const std::exception& exception) {
      AssignError(error_msg, exception.what());
      return nullptr;
    }
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> out;
  for (auto id : tokenizer_->Encode(text)) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<tokenization::TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::uint32_t Model::PrefillChunkTokens() const noexcept {
  return options_.tp_world_size > 1 ? kTpPrefillChunkTokens
                                    : kPrefillChunkTokens;
}

std::uint32_t Model::PrefillCapacity() const noexcept {
  // Whole chunks per trunk batch: two batches per step under TP2.
  const std::uint32_t batches =
      executor_->PrefillChunk() / executor_->max_batch();
  return std::min(batches * PrefillChunkTokens(), options_.max_context);
}

std::uint32_t Model::PrefillPairLead(std::uint32_t tokens) const noexcept {
  return executor_->PairLead(tokens);
}

std::uint32_t Model::PrefillThroughCapacity() const noexcept {
  return std::min(PrefillChunkTokens() + kPrefillTailTokens,
                  options_.max_context);
}

bool Model::HasMtp() const noexcept {
  return device_->has_mtp();
}

std::size_t Model::ExpandedQ8_0Tensors() const noexcept {
  std::size_t count = 0;
  for (const LayerWeights& layer : weights_->layers) {
    count += qwen38_flash_next::ExpandedQ8_0Tensors(layer);
  }
  return count;
}

bool Model::InDraftSubset(std::int32_t token) const {
  return token >= 0 &&
         (static_cast<std::uint32_t>(token) < speculative::kDraftVocabPrefix ||
          tokenizer_->IsSpecialToken(
              static_cast<tokenization::TokenId>(token)));
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("Qwen3.8-Flash-Next"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes() + (vision_ ? vision_->ResidentBytes() : 0);
}

std::size_t Model::SessionBytes(core::SessionMode mode,
                                std::uint32_t context) const noexcept {
  const std::size_t vision =
      vision_ ? std::size_t{context} * (config().hidden_size * sizeof(float) +
                                        3 * sizeof(std::int32_t)) +
                    64
              : 0;
  return executor_->SessionBytes(mode, context,
                                 mode == core::SessionMode::kSpeculative
                                     ? executor_->max_speculative() - 1
                                     : 0) +
         vision;
}

std::size_t Model::DeferredScratchBytes() const {
  return executor_->DeferredScratchBytes();
}

void Model::LimitCopyRows(std::uint32_t rows) noexcept {
  executor_->LimitCopyRows(rows);
}

std::size_t Session::AllocatedBytes() const noexcept {
  return session_->AllocatedBytes();
}

NgramTotals Session::NgramRowTotals() const noexcept {
  return session_->NgramRowTotals();
}

bool Session::MtpEnabled() const noexcept {
  return session_->mtp_enabled();
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      draft_length_(model_->options_.max_draft_tokens,
                    model_->DecodeConcurrency()),
      draft_vocab_(speculative::kDraftVocabExtraRows -
                   model_->draft_specials_) {
  logits_.resize(model_->VocabSize());
}

bool Session::SyncDraftVocab(std::string* error_msg) {
  if (!model_->draft_subset_ || !MtpEnabled())
    return true;
  auto& exec = *model_->executor_;
  const auto first = model_->draft_specials_;
  if (draft_vocab_stale_ || draft_vocab_.scanned() > tokens_.size()) {
    if (!exec.ClearDraftRows(*session_, first, error_msg))
      return false;
    draft_vocab_.Reset();
    draft_vocab_stale_ = false;
  }
  const auto tabled = static_cast<std::uint32_t>(draft_vocab_.rare().size());
  draft_added_.clear();
  draft_vocab_.CatchUp(
      tokens_, [&](std::int32_t token) { return model_->InDraftSubset(token); },
      &draft_added_);
  return exec.AppendDraftRows(*session_, first + tabled, draft_added_,
                              error_msg);
}

Session::~Session() = default;

void Session::ResetDraftPolicy() {
  if (continuation_hooks_)
    continuation_hooks_->BeforeExecution();
  draft_length_.Reset();
  // A fully reused prompt can start decoding without SyncImpl. It still
  // starts a new request: do not inherit a copy cursor or widen its first step.
  lookup_.BeginRequest();
  copy_rows_ = 0;
  copy_streak_ = 0;
}

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}
std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

void Session::Reset() {
  ResetImpl(false);
}

void Session::ResetImpl(bool reuse_mtp) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeReset();
  valid_ = false;
  session_->Reset(reuse_mtp);
  tokens_.clear();
  ForgetDraftVocab();
  lookup_.Clear();
  hidden_base_ = 0;
  draft_length_.Reset();
  model_->executor_->MtpRewind(*session_, 0);
  if (continuation_hooks_)
    continuation_hooks_->AfterReset();
  valid_ = true;
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  session_->SetCancellationCheck(std::move(check));
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeExecution();
  const auto identity = prompt ? prompt->IdentityForPrefix(tokens_.size())
                               : std::span<const std::uint8_t>{};
  if (!tokens_.empty() &&
      !std::ranges::equal(identity, ImageIdentity(tokens_.size())))
    Reset();
  const bool was_valid = valid_;
  valid_ = false;
  session_->ConfigureVision(prompt, model_->vision_,
                            model_->executor_->stream());
  image_prompt_ = std::move(prompt);
  valid_ = was_valid;
}

std::span<const std::uint8_t> Session::ImageIdentity(
    std::size_t token_count) const {
  return image_prompt_ ? image_prompt_->IdentityForPrefix(token_count)
                       : std::span<const std::uint8_t>{};
}

std::uint32_t Session::KeptHiddenRows() const noexcept {
  return MtpEnabled()
             ? static_cast<std::uint32_t>(tokens_.size() - hidden_base_)
             : 0;
}

std::uint64_t Session::SnapshotBytes() const {
  if (!valid_)
    return 0;
  return SessionSnapshotHostBytes(static_cast<std::uint32_t>(tokens_.size()),
                                  model_->VocabSize(),
                                  ImageIdentity(tokens_.size()).size()) +
         model_->executor_->SnapshotBytes(*session_, KeptHiddenRows());
}

std::uint64_t Session::PrefillCheckpointBytes(std::uint32_t position) const {
  if (model_->TpWorldSize() > 1 || !session_->VisionLayout().images.empty())
    return 0;
  return SessionSnapshotHostBytes(position, model_->VocabSize(), 0) +
         model_->executor_->PrefillCheckpointBytes(*session_, position);
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg, SnapshotMode mode) const {
  if (!valid_ || tokens_.empty() || tokens_.size() != session_->position() ||
      tokens_.size() > std::numeric_limits<std::uint32_t>::max()) {
    AssignError(error_msg, "snapshot needs a synced, non-empty context");
    return nullptr;
  }
  if (!session_->CheckCancellation(error_msg))
    return nullptr;
  return SaveSnapshotImpl(*session_, tokens_, logits_, KeptHiddenRows(),
                          error_msg, mode);
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshotImpl(
    const rocm::Session& state, std::span<const std::int32_t> tokens,
    std::span<const float> logits, std::uint32_t hidden_rows,
    std::string* error_msg, SnapshotMode mode) const {
  const auto token_count = static_cast<std::uint32_t>(tokens.size());
  const auto identity = ImageIdentity(token_count);
  const std::uint64_t executor_bytes =
      model_->executor_->SnapshotBytes(state, hidden_rows);
  const std::uint64_t host_bytes = SessionSnapshotHostBytes(
      token_count, model_->VocabSize(), identity.size());
  std::unique_ptr<SessionSnapshot> snapshot(
      new SessionSnapshot(host_bytes + executor_bytes));
  std::uint8_t* out = snapshot->data_.get();
  const SessionSnapshotHeader header{
      .magic = kSessionSnapshotMagic,
      .version = kSnapshotPayloadVersion,
      .vocab_size = model_->VocabSize(),
      .token_count = token_count,
      .hidden_rows = hidden_rows,
      .executor_bytes = executor_bytes,
      .draft_policy = draft_length_.State(),
      .image_identity_bytes = static_cast<std::uint32_t>(identity.size()),
      .policy_concurrency = model_->DecodeConcurrency(),
      .yarn_factor = model_->executor_->YarnFactor(state),
      .lookup = lookup_.Save(),
  };
  std::memcpy(out, &header, sizeof(header));
  out += sizeof(header);
  if (!identity.empty())
    std::memcpy(out, identity.data(), identity.size());
  out += identity.size();
  std::memcpy(out, tokens.data(), tokens.size_bytes());
  out += tokens.size_bytes();
  std::memcpy(out, logits.data(), logits.size_bytes());
  out += logits.size_bytes();
  if (!model_->executor_->SaveSnapshot(
          state, hidden_rows,
          std::span<std::uint8_t>(out,
                                  static_cast<std::size_t>(executor_bytes)),
          error_msg,
          mode == SnapshotMode::kBorrowed ? &snapshot->deferred_ : nullptr,
          snapshot->data_)) {
    return nullptr;
  }
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeRestore();
  return RestoreSnapshotImpl({snapshot.data_.get(), snapshot.size_},
                             snapshot.deferred_.get(), error_msg);
}

bool Session::OwnsSnapshot(const SessionSnapshot& snapshot) const noexcept {
  return snapshot.deferred_ && session_->OwnsSnapshot(*snapshot.deferred_);
}

std::uint64_t SessionSnapshot::DeviceBytes() const {
  return deferred_ ? deferred_->DeviceBytes() : 0;
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeRestore();
  return RestoreSnapshotImpl(payload, nullptr, error_msg);
}

bool Session::RestoreSnapshotImpl(std::span<const std::uint8_t> payload,
                                  const rocm::SnapshotState* deferred,
                                  std::string* error_msg) {
  SessionSnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    AssignError(error_msg, "session snapshot is truncated");
    return false;
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSessionSnapshotMagic ||
      header.version != kSnapshotPayloadVersion) {
    AssignError(error_msg, "session snapshot format is not supported");
    return false;
  }
  if (header.vocab_size != model_->VocabSize() || header.token_count == 0 ||
      header.policy_concurrency != model_->DecodeConcurrency() ||
      header.token_count > ContextSize() ||
      (header.image_identity_bytes != 0 && header.image_identity_bytes != 32) ||
      payload.size() != SessionSnapshotHostBytes(header.token_count,
                                                 header.vocab_size,
                                                 header.image_identity_bytes) +
                            header.executor_bytes) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  auto restored_policy = draft_length_;
  if (!restored_policy.Restore(header.draft_policy)) {
    AssignError(error_msg, "session snapshot draft policy is invalid");
    return false;
  }
  if (!speculative::PromptLookup<>::Valid(header.lookup, header.token_count)) {
    AssignError(error_msg, "session snapshot prompt-lookup state is invalid");
    return false;
  }
  const std::uint8_t* in = payload.data() + sizeof(header);
  std::vector<std::uint8_t> image_identity(in,
                                           in + header.image_identity_bytes);
  if (!image_identity.empty() &&
      !std::ranges::equal(image_identity, ImageIdentity(header.token_count))) {
    AssignError(error_msg,
                "image snapshot requires its matching prompt attachment");
    return false;
  }
  in += header.image_identity_bytes;
  std::vector<std::int32_t> tokens(header.token_count);
  std::memcpy(tokens.data(), in, tokens.size() * sizeof(std::int32_t));
  in += tokens.size() * sizeof(std::int32_t);
  std::vector<float> logits(header.vocab_size);
  std::memcpy(logits.data(), in, logits.size() * sizeof(float));
  in += logits.size() * sizeof(float);

  if (image_identity.empty())
    ConfigureVision(nullptr);
  valid_ = false;
  rocm::Executor::SnapshotInfo info;
  const auto remaining = ContextSize() - header.token_count;
  auto next_drafts = MtpEnabled()
                         ? restored_policy.Choose(remaining ? remaining - 1 : 0,
                                                  header.token_count)
                         : 0;
  if (MtpEnabled() && model_->draft_subset_) {
    // The next request's sampler/head is not known yet. Avoid discarding
    // existing scratch that either eligible policy can use after restore.
    next_drafts = std::max(next_drafts,
                           restored_policy.Choose(remaining ? remaining - 1 : 0,
                                                  header.token_count, 0, true));
  }
  if (!model_->executor_->RestoreSnapshot(
          *session_,
          std::span<const std::uint8_t>(
              in, static_cast<std::size_t>(header.executor_bytes)),
          &info, error_msg, next_drafts, deferred)) {
    Reset();
    return false;
  }
  if (info.position != header.token_count ||
      info.hidden_rows != header.hidden_rows) {
    Reset();
    AssignError(error_msg, "session snapshot positions are inconsistent");
    return false;
  }
  tokens_ = std::move(tokens);
  ForgetDraftVocab();
  lookup_.Restore(header.lookup);
  logits_ = std::move(logits);
  hidden_base_ = info.position - info.hidden_rows;
  draft_token_ = 0;
  draft_length_ = restored_policy;
  stats_ = {};
  valid_ = true;
  // The restored keys keep the factor they were saved with; the next Feed or
  // decode moves the session to the factor its positions need.
  model_->executor_->AdoptYarnFactor(*session_, header.yarn_factor);
  return true;
}

SessionSnapshot::SessionSnapshot(std::uint64_t size)
    : data_(new std::uint8_t[size]), size_(size) {
  // Leave append-only sections untouched: a warm checkpoint borrows those
  // rows from the live session, so faulting their pages here is wasted work.
}

SessionSnapshot::~SessionSnapshot() = default;

std::span<const std::uint8_t> SessionSnapshot::bytes() const {
  if (deferred_)
    deferred_->Materialize();
  return {data_.get(), size_};
}

bool SessionSnapshot::CopyTo(
    std::span<std::uint8_t> destination) const noexcept {
  if (destination.size() != size_)
    return false;
  try {
    const auto payload = bytes();
    std::memcpy(destination.data(), payload.data(), payload.size());
    return true;
  } catch (...) {
    return false;
  }
}

bool Session::DraftReplay(std::int32_t next_token,
                          std::vector<std::int32_t>* replay,
                          std::int32_t* hidden_row,
                          std::string* error_msg) const {
  // The draft block trails the trunk: MTP position i consumes token i+1 and
  // the trunk's hidden of position i, so positions up to the current one are
  // replayed once their successor token is known. The session keeps hidden
  // rows of positions [hidden_base_, tokens_.size()).
  rocm::Executor& exec = *model_->executor_;
  const auto size = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t mp = exec.MtpPosition(*session_);
  if (mp >= size) {
    return true;
  }
  if (mp < hidden_base_) {
    AssignError(error_msg, "draft block fell behind the kept hidden rows");
    return false;
  }
  replay->assign(tokens_.begin() + mp + 1, tokens_.end());
  replay->push_back(next_token);
  *hidden_row = static_cast<std::int32_t>(mp - hidden_base_);
  return true;
}

bool Session::DraftCatchUp(std::int32_t next_token, bool propose,
                           std::string* error_msg,
                           MtpCandidateLogits* candidates, bool draft_subset) {
  std::vector<std::int32_t> replay;
  std::int32_t hidden_row = 0;
  if (!DraftReplay(next_token, &replay, &hidden_row, error_msg))
    return false;
  if (replay.empty())
    return true;
  auto& exec = *model_->executor_;
  if (!exec.MtpForward(
          *session_, replay, hidden_row,
          {.token = propose && candidates == nullptr ? &draft_token_ : nullptr,
           .confidence = model_->options_.tp_world_size == 1
                             ? &draft_confidence_
                             : nullptr,
           .candidates = candidates,
           .kv_only = !propose,
           .draft_subset = draft_subset},
          error_msg)) {
    return false;
  }
  return true;
}

bool Session::DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg,
                                std::span<const std::uint8_t> propose) {
  if (requests.empty())
    return true;
  if (!propose.empty() && propose.size() != requests.size()) {
    AssignError(error_msg, "draft catch-up head count differs from requests");
    return false;
  }
  auto& exec = *requests.front().session->model_->executor_;
  std::vector<std::vector<std::int32_t>> replays(requests.size());
  std::vector<rocm::Executor::MtpBatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    if (!session.MtpEnabled())
      continue;
    std::int32_t hidden_row = 0;
    if (!session.DraftReplay(requests[i].token, &replays[i], &hidden_row,
                             error_msg))
      return false;
    if (replays[i].empty())
      continue;
    items.push_back({session.session_.get(), replays[i], hidden_row,
                     propose.empty() || !propose[i]});
  }
  return items.empty() || exec.MtpForwardBatch(items, error_msg);
}

bool Session::Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
                   bool prefill, std::span<const std::uint32_t> boundaries,
                   std::vector<std::unique_ptr<SessionSnapshot>>* checkpoints,
                   double* capture_ms, std::span<const std::int32_t> after,
                   const PrefillCapture& consume) {
  const bool capturing = checkpoints != nullptr || bool(consume);
  if (checkpoints != nullptr)
    checkpoints->resize(boundaries.size());
  // The first boundary no pass has captured yet.
  std::size_t next_boundary = 0;
  rocm::Executor& exec = *model_->executor_;
  const std::size_t ratio =
      std::max<std::uint32_t>(1, model_->config().compress_ratio);
  for (std::size_t off = 0; off < tokens.size();) {
    const auto remaining = tokens.size() - off;
    const auto capacity = remaining <= exec.max_batch()
                              ? exec.max_batch()
                              : model_->PrefillCapacity();
    std::size_t n = std::min<std::size_t>(capacity, remaining);
    // A following pass starts on a selection block (WmmaCausalAttention).
    if (const auto end = tokens_.size() + n;
        prefill && off + n < tokens.size() && end % ratio != 0 &&
        n > end % ratio)
      n -= end % ratio;
    // This pass captures boundaries [next_boundary, last_boundary) in place.
    std::size_t last_boundary = next_boundary;
    // True when the pass ends at its only boundary and the live session is
    // captured there, without an in-pass checkpoint.
    bool live = false;
    if (capturing) {
      while (last_boundary < boundaries.size() &&
             boundaries[last_boundary] <= tokens_.size() + n)
        ++last_boundary;
      // A pass holds a bounded number of checkpoints; the next one starts
      // where the last of them ends.
      if (last_boundary - next_boundary > rocm::kMaxPassCheckpoints) {
        last_boundary = next_boundary + rocm::kMaxPassCheckpoints;
        n = boundaries[last_boundary - 1] - tokens_.size();
      }
      // Extended-context attention can fall back from WMMA to the per-token
      // kernel. Its checkpoint must end a pass: the fused in-pass path
      // requires WMMA for every query group. Preserve ordinary split-pass
      // arithmetic on either side of the boundary. Use the actual launch
      // capability rather than the model's independently configured native
      // context length.
      if (last_boundary > next_boundary &&
          boundaries[next_boundary] < tokens_.size() + n &&
          tokens_.size() + n >
              rocm::WmmaCausalAttentionContextLimit(
                  model_->config().num_heads, model_->config().num_kv_heads,
                  model_->config().head_dim, model_->config().compress_ratio)) {
        n = boundaries[next_boundary] - tokens_.size();
        last_boundary = next_boundary + 1;
      }
      // Without device room for the checkpoint arena, end the pass at the
      // first boundary and capture the live session there instead:
      // split-pass rows are the same as in-pass rows.
      if (last_boundary > next_boundary &&
          !exec.ReservePrefillCheckpoint(*session_)) {
        n = boundaries[next_boundary] - tokens_.size();
        last_boundary = next_boundary + 1;
        live = true;
      }
    }
    const auto chunk = tokens.subspan(off, n);
    if (exec.DynamicYarn() &&
        !exec.SetYarnFactor(*session_, exec.YarnLevelFor(tokens_.size() + n),
                            error_msg)) {
      return false;
    }
    if (MtpEnabled() && !tokens_.empty() &&
        !DraftCatchUp(chunk[0], false, error_msg)) {
      return false;
    }
    const auto mode = prefill ? rocm::Executor::ForwardMode::kPrefill
                              : rocm::Executor::ForwardMode::kDecode;
    std::vector<std::unique_ptr<rocm::PrefillCheckpoint>> captures;
    std::vector<rocm::PrefillCheckpoint*> views;
    for (auto i = next_boundary; !live && i < last_boundary; ++i) {
      const auto prefix =
          static_cast<std::uint32_t>(boundaries[i] - tokens_.size());
      auto capture = exec.MakePrefillCheckpoint(
          *session_, chunk.first(n), prefix, error_msg,
          static_cast<std::uint32_t>(i - next_boundary),
          views.empty() ? 0 : views.back()->tokens);
      if (!capture) {
        if (views.empty())
          return false;
        // No device room for another checkpoint in this pass: end it at
        // the previous one, which the next pass follows.
        if (error_msg)
          error_msg->clear();
        n = views.back()->tokens;
        last_boundary = i;
        break;
      }
      views.push_back(capture.get());
      captures.push_back(std::move(capture));
    }
    const auto chunk_rows = tokens.subspan(off, n);
    // The chunk this loop, or the following Sync, takes next.
    const auto rest = off + n < tokens.size() ? tokens.subspan(off + n) : after;
    const auto next = rest.first(std::min<std::size_t>(
        rest.size() <= exec.max_batch() ? exec.max_batch()
                                        : model_->PrefillCapacity(),
        rest.size()));
    if (!exec.Forward(*session_, chunk_rows, 1, logits_.data(), mode, error_msg,
                      views, next)) {
      return false;
    }
    const auto kept = !views.empty() && views.back()->tokens < n
                          ? n - views.back()->tokens
                          : n;
    hidden_base_ = static_cast<std::uint32_t>(
        tokens_.size() + n -
        std::min<std::size_t>(kept, exec.max_speculative()));
    tokens_.insert(tokens_.end(), chunk_rows.begin(), chunk_rows.end());
    for (auto i = next_boundary; i < last_boundary; ++i) {
      const auto* capture = live ? nullptr : captures[i - next_boundary].get();
      const auto started = std::chrono::steady_clock::now();
      if (consume) {
        consume(i, capture);
      } else {
        auto& checkpoint = (*checkpoints)[i];
        checkpoint = SaveSnapshotImpl(
            capture ? *capture->state : *session_,
            std::span<const std::int32_t>(tokens_).first(boundaries[i]),
            capture ? std::span<const float>(capture->logits)
                    : std::span<const float>(logits_),
            capture ? capture->hidden_rows : KeptHiddenRows(), error_msg,
            SnapshotMode::kBorrowed);
        if (capture_ms)
          *capture_ms += std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
        if (!checkpoint)
          return false;
      }
    }
    if (captures.size() > 1)
      exec.ReleasePassCheckpointStorage();
    next_boundary = last_boundary;
    off += n;
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt, std::string* error_msg,
                   std::span<const std::int32_t> next) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeExecution();
  return SyncImpl(prompt, error_msg, {}, nullptr, nullptr, next);
}

bool Session::ValidCheckpointBoundaries(
    std::span<const std::int32_t> prompt,
    std::span<const std::uint32_t> boundaries, std::string* error_msg) const {
  for (std::size_t i = 0; i < boundaries.size(); ++i)
    if (boundaries[i] <= tokens_.size() || boundaries[i] > prompt.size() ||
        (i != 0 && boundaries[i] <= boundaries[i - 1])) {
      AssignError(error_msg,
                  "checkpoints must ascend beyond the frontier in the prompt");
      return false;
    }
  return !boundaries.empty();
}

bool Session::SyncThrough(std::span<const std::int32_t> prompt,
                          std::uint32_t boundary,
                          std::unique_ptr<SessionSnapshot>* checkpoint,
                          std::string* error_msg, double* capture_ms) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeExecution();
  if (!checkpoint) {
    AssignError(error_msg, "missing checkpoint destination");
    return false;
  }
  checkpoint->reset();
  std::vector<std::unique_ptr<SessionSnapshot>> checkpoints;
  const std::uint32_t boundaries[] = {boundary};
  if (!SyncThrough(prompt, boundaries, &checkpoints, error_msg, capture_ms))
    return false;
  *checkpoint = std::move(checkpoints.front());
  return true;
}

bool Session::SyncThrough(
    std::span<const std::int32_t> prompt,
    std::span<const std::uint32_t> boundaries,
    std::vector<std::unique_ptr<SessionSnapshot>>* checkpoints,
    std::string* error_msg, double* capture_ms) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeExecution();
  if (!checkpoints ||
      !ValidCheckpointBoundaries(prompt, boundaries, error_msg)) {
    AssignError(error_msg,
                "checkpoints must ascend beyond the frontier in the prompt");
    return false;
  }
  checkpoints->clear();
  if (capture_ms)
    *capture_ms = 0;
  return SyncImpl(prompt, error_msg, boundaries, checkpoints, capture_ms);
}

bool Session::SyncThroughComponents(std::span<const std::int32_t> prompt,
                                    std::span<const std::uint32_t> boundaries,
                                    const PrefillCapture& capture,
                                    std::string* error_msg,
                                    std::span<const std::int32_t> next) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeExecution();
  if (!capture || !ValidCheckpointBoundaries(prompt, boundaries, error_msg)) {
    AssignError(error_msg,
                "component checkpoints must ascend beyond the frontier in the "
                "prompt");
    return false;
  }
  return SyncImpl(prompt, error_msg, boundaries, nullptr, nullptr, next,
                  capture);
}

bool Session::SyncImpl(
    std::span<const std::int32_t> prompt, std::string* error_msg,
    std::span<const std::uint32_t> boundaries,
    std::vector<std::unique_ptr<SessionSnapshot>>* checkpoints,
    double* capture_ms, std::span<const std::int32_t> next,
    const PrefillCapture& capture) {
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  lookup_.BeginRequest();
  copy_rows_ = 0;
  copy_streak_ = 0;
  if (!valid_)
    Reset();
  // Recurrent state cannot be rewound, so any divergence restarts the
  // session; an extension only feeds the new tail.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    if (std::getenv("GUFO_TP_TRACE") != nullptr) { std::fprintf(stderr, "[tptrace-e] %s\n", std::string("sync SHORT-CIRCUIT identical common=") + std::to_string(common)); }
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  valid_ = false;
  const bool ok = Feed(prompt.subspan(common), error_msg, true, boundaries,
                       checkpoints, capture_ms, next, capture);
  valid_ = ok;
  return ok;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeExecution();
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  valid_ = false;
  const bool ok = Feed(std::span<const std::int32_t>(&token, 1), error_msg);
  valid_ = ok;
  return ok;
}

struct Session::PendingDecode {
  std::vector<std::int32_t> chain;
  std::vector<MtpProposal> proposals;
  std::uint32_t base{0};
  bool speculative{false};
  bool sampled{false};
  bool gpu_greedy{false};
  bool gpu_verification{false};
  std::size_t width{0};
  /// The chain stops after a proposal below its gate (GateStops).
  bool gated{false};
  float draft_confidence{1.0F};
  std::optional<sampling::SamplerState> draft_sampler;
  std::uint64_t draft_rng{0};
  MtpCandidateLogits candidates;
  std::int32_t draft{0};
  bool draft_subset{false};  ///< this cycle drafts through the subset head
  bool sampled_subset_cost{false};
  /// Trailing chain tokens copied by prompt lookup (after the MTP drafts).
  std::uint32_t lookup{0};
  /// Committed index the copied tokens were taken from.
  std::size_t lookup_start{0};
  /// Identical copy budget in isolated and batched execution; zero on the
  /// request's first speculative step.
  std::size_t lookup_cap{0};
  std::uint32_t policy_concurrency{0};
  /// A copy-only cycle here may verify more than max_speculative() rows.
  bool wide{false};
};

bool Session::GateStops(const PendingDecode& pending) {
  if (!pending.gated)
    return false;
  return pending.sampled
             ? pending.proposals.back().probability < kMtpSampledGateProbability
             : pending.draft_confidence < kMtpGateProbability;
}

void Session::AppendDraft(PendingDecode& pending) {
  if (pending.sampled) {
    pending.proposals.push_back(SampleMtpProposal(
        pending.candidates, *pending.draft_sampler, &pending.draft_rng));
    pending.draft = static_cast<std::int32_t>(pending.proposals.back().token);
    pending.draft_sampler->Accept(pending.proposals.back().token);
  }
  if (std::getenv("GUFO_TP_TRACE") != nullptr) {
    static std::uint64_t cd_seq = 0;
    std::fprintf(stderr,
                 "[ch] seq=%llu kind=draft draft=%d conf=%.6e gated=%d "
                 "chain=%zu\n",
                 static_cast<unsigned long long>(cd_seq++), pending.draft,
                 pending.draft_confidence, pending.gated ? 1 : 0,
                 pending.chain.size());
  }
  pending.chain.push_back(pending.draft);
}

bool Session::AppendLookup(PendingDecode& pending, bool stop_at_eos,
                           std::size_t minimum, bool require_resume) {
  const auto cap = pending.lookup_cap;
  if (pending.chain.size() >= cap) {
    return false;
  }
  if (stop_at_eos && model_->IsStopToken(pending.chain.back())) {
    return false;
  }
  const auto match = lookup_.Find(
      tokens_, pending.chain,
      pending.sampled ? speculative::PromptLookup<>::kSampledMinMatch
                      : speculative::PromptLookup<>::kMinMatch,
      require_resume);
  if (std::getenv("GUFO_TP_TRACE") != nullptr) {
    static std::uint64_t lk_seq = 0;
    std::fprintf(stderr,
                 "[lk] seq=%llu match_len=%zu start=%zu indexed=%zu "
                 "chain=%zu cap=%zu avail_hint=%zd\n",
                 static_cast<unsigned long long>(lk_seq++),
                 static_cast<std::size_t>(match.length),
                 static_cast<std::size_t>(match.start),
                 lookup_.IndexedTokens(), pending.chain.size(), cap,
                 static_cast<std::ptrdiff_t>(
                     lookup_.IndexedTokens() -
                     static_cast<std::size_t>(match.start)));
  }
  if (match.length == 0) {
    return false;
  }
  const std::size_t available = std::min(cap - pending.chain.size(),
                                         lookup_.IndexedTokens() - match.start);
  if (available < minimum)
    return false;
  std::size_t count = 0;
  for (; count < available; ++count) {
    const std::int32_t token = tokens_[match.start + count];
    if (std::getenv("GUFO_TP_TRACE") != nullptr && count < 2) {
      std::fprintf(stderr, "[lk] copy tok=%d at=%zu\n", token,
                   static_cast<std::size_t>(match.start) + count);
    }
    pending.chain.push_back(token);
    if (pending.sampled) {
      // A point-mass proposal: accepted with the target's p(token), and a
      // rejection resamples from the target with the token removed.
      MtpProposal proposal;
      proposal.ids[0] = static_cast<sampling::TokenId>(token);
      proposal.probabilities[0] = 1.0F;
      proposal.size = 1;
      proposal.token = proposal.ids[0];
      proposal.probability = 1.0F;
      pending.proposals.push_back(proposal);
    }
    if (stop_at_eos && model_->IsStopToken(token)) {
      ++count;
      break;
    }
  }
  pending.lookup = static_cast<std::uint32_t>(count);
  pending.lookup_start = match.start;
  pending.width = pending.chain.size();
  return count != 0;
}

bool Session::PrepareDecode(const DecodeRequest& request,
                            PendingDecode* pending, std::string* error_msg,
                            bool defer_head,
                            std::optional<std::uint32_t> batch_drafts,
                            std::uint32_t occupancy) {
  const auto max_tokens = request.max_tokens;
  auto& sampler = *request.sampler;
  auto* result = request.result;
  const bool stop_at_eos = request.stop_at_eos;
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };
  rocm::Executor& exec = *model_->executor_;
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t cap =
      std::min<std::size_t>({max_tokens, room, exec.max_speculative()});
  if (exec.DynamicYarn() &&
      !exec.SetYarnFactor(
          *session_, exec.YarnLevelFor(tokens_.size() + cap + 1), error_msg)) {
    return false;
  }
  // Head eligibility is speculative::DraftSubsetAllowed. Sampled cohorts are
  // never jointly planned (DecodeBatchImpl), so a sampled request decides the
  // same way alone and in any batch. A greedy request decoding alone has no
  // peers to delay, so it drafts as on one session. Every request keeps the
  // established length controller and AR backoff.
  const bool allow_subset = speculative::DraftSubsetAllowed(
      sampler.config().uses_random_sampling(), model_->DecodeConcurrency(),
      occupancy, batch_drafts.has_value());
  pending->policy_concurrency = MtpPolicyConcurrency(
      sampler.config().uses_random_sampling(), sampler.config().seed >= 0,
      batch_drafts.has_value(), occupancy);
  // A peer's plan fixes the maximum width on both ranks.
  const auto& forced = request.drafts ? request.drafts : batch_drafts;
  std::uint32_t drafts = 0;
  if (MtpEnabled() && cap > 1) {
    drafts = forced ? std::min<std::uint32_t>(*forced, cap - 1)
                    : draft_length_.Choose(
                          static_cast<std::uint32_t>(cap - 1),
                          static_cast<std::uint32_t>(tokens_.size()),
                          pending->policy_concurrency);
    if (model_->options_.tp_world_size == 1 && !forced && drafts != 0)
      drafts = std::min<std::uint32_t>(static_cast<std::uint32_t>(cap - 1),
                                       model_->options_.max_draft_tokens);
  }
  const std::size_t width = MtpEnabled() && cap > 1 ? 1 + drafts : cap;
  if (std::getenv("GUFO_TP_TRACE") != nullptr) {
    static std::uint64_t w_seq = 0;
    std::fprintf(stderr,
                 "[wd] seq=%llu max_tokens=%zu room=%zu cap=%zu forced=%d "
                 "fd=%u drafts=%u width=%zu toks=%zu pos=%u mtp=%d\n",
                 static_cast<unsigned long long>(w_seq++), max_tokens, room,
                 cap, forced.has_value() ? static_cast<int>(*forced) : -1,
                 forced.has_value() ? *forced : 0U, drafts, width,
                 tokens_.size(), static_cast<unsigned>(Position()),
                 MtpEnabled() ? 1 : 0);
    std::fflush(stderr);
  }
  if (width == 0) {
    result->stop = true;
    return true;
  }
  const auto anchor = static_cast<std::int32_t>(sampler.Sample(logits_));
  auto anchor_logprobs =
      sampling::CollectLogprobs(logits_, anchor, sampler.config().logprobs);
  if (is_stop(anchor)) {
    result->terminal_logprobs = std::move(anchor_logprobs);
    result->stop = true;
    return true;
  }
  if (anchor_logprobs)
    result->logprobs.push_back(std::move(*anchor_logprobs));
  if (!MtpEnabled() || width < 2) {
    if (MtpEnabled() && !defer_head &&
        !DraftCatchUp(anchor, false, error_msg)) {
      return false;
    }
    pending->chain = {anchor};
    pending->base = static_cast<std::uint32_t>(tokens_.size());
    return true;
  }

  const std::uint32_t base = static_cast<std::uint32_t>(tokens_.size());
  const bool sampled = sampler.config().uses_random_sampling();
  const bool gpu_greedy = sampler.config().temperature == 0.0F;
  const bool gpu_verification =
      gpu_greedy || rocm::Executor::CanCompactMtpVerification(sampler);
  // A cycle-local proposal stream needs no pending RNG state in snapshots.
  // Target verification keeps its own draws after this independent seed.
  pending->draft_rng =
      sampled ? sampling::NextRandom(sampler.mutable_rng_state()) : 0;
  pending->draft_sampler = sampler.WithoutConstraint();
  pending->draft_sampler->Accept(static_cast<sampling::TokenId>(anchor));
  pending->chain = {anchor};
  pending->draft = draft_token_;
  pending->width = width;
  // Confidence is computed locally and can straddle the threshold across
  // ranks; a forced plan must keep both collective sequences identical.
  pending->gated = model_->options_.tp_world_size == 1;
  pending->base = base;
  pending->speculative = true;
  pending->sampled = sampled;
  pending->gpu_greedy = gpu_greedy;
  pending->gpu_verification = gpu_verification;
  if (!gpu_verification &&
      verify_logits_.size() < exec.max_speculative() * model_->VocabSize()) {
    verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
  }
  // Both execution paths use the same request-local copy rule. Scheduling
  // a peer must not change a sampled request's proposals or RNG sequence.
  // A request's first speculative step copies nothing (see
  // speculative::PromptLookup<>).
  pending->lookup_cap = lookup_.Step(base) ? cap : 0;
  if (pending->lookup_cap != 0) {
    lookup_.Extend(tokens_, sampled);
  }
  // A session decoding alone may verify a copy wider than the MTP
  // chain (Executor::ForwardWide): the shared projections read their
  // weights once for every row. The width grows only while whole copies
  // are accepted, so a copy that diverges costs at most the last width.
  // Sampled requests qualify when nothing replays them (no seed) and their
  // rows verify compactly on the device.
  pending->wide = !defer_head && pending->lookup_cap != 0 &&
                  (gpu_greedy || (sampled && sampler.config().seed < 0 &&
                                  rocm::Executor::CanCompactWideVerification(
                                      sampler, exec.max_copy_rows())));
  const std::size_t wide_rows =
      pending->wide
          ? std::min<std::size_t>(
                {max_tokens, room, exec.max_copy_rows(),
                 std::max<std::size_t>(copy_rows_, exec.max_speculative())})
          : 0;
  if (wide_rows > pending->lookup_cap)
    pending->lookup_cap = wide_rows;
  // A sampled copy can skip the neural head only after an accepted copy
  // established its source and a full sampled match still holds. New or
  // short matches retain the neural proposal: replacing it based only on
  // shared syntax previously regressed an identical Pi code write by 9%.
  if (AppendLookup(*pending, stop_at_eos, 3, sampled)) {
    return defer_head || DraftCatchUp(anchor, false, error_msg);
  }
  pending->lookup_cap = std::min(pending->lookup_cap, cap);
  if (allow_subset && !SyncDraftVocab(error_msg))
    return false;
  pending->draft_subset = allow_subset && model_->draft_subset_ &&
                          draft_vocab_.Fits(anchor, [&](std::int32_t token) {
                            return model_->InDraftSubset(token);
                          });
  // Wider verification slowed short resumed replies in matched controls.
  // Keep the established costs within two blocks of the remaining budget.
  // Full target rows (for example, active tool constraints) also keep their
  // original costs; the smaller-head curve does not price their transfers.
  // With a confidence gate the curve only prices AR backoff (FinishDecode).
  // Without one, it also selects the unplanned chain width.
  pending->sampled_subset_cost =
      sampled && gpu_verification && pending->draft_subset &&
      std::min(max_tokens, room) > 2 * exec.max_speculative();
  if (!pending->gated && pending->sampled_subset_cost && !forced) {
    // Without the confidence gate, price the smaller head after the anchor
    // establishes that it is actually usable, as the TP2 controller did
    // before the gate was introduced.
    pending->width =
        1 + draft_length_.Choose(static_cast<std::uint32_t>(cap - 1), base,
                                 pending->policy_concurrency, true);
  }

  if (!defer_head && !DraftCatchUp(anchor, true, error_msg,
                                   sampled ? &pending->candidates : nullptr,
                                   pending->draft_subset)) {
    return false;
  }
  pending->draft = draft_token_;
  pending->draft_confidence = draft_confidence_;
  if (!defer_head) {
    while (pending->chain.size() < pending->width) {
      AppendDraft(*pending);
      if (stop_at_eos && model_->IsStopToken(pending->chain.back())) {
        pending->width = pending->chain.size();
        break;
      }
      if (AppendLookup(*pending, stop_at_eos)) {
        break;
      }
      if (GateStops(*pending)) {
        pending->width = pending->chain.size();
        break;
      }
      if (pending->chain.size() < pending->width &&
          !exec.MtpForward(
              *session_, std::span<const std::int32_t>(&pending->draft, 1), -1,
              {.token = sampled ? nullptr : &pending->draft,
               .confidence =
                   pending->gated ? &pending->draft_confidence : nullptr,
               .candidates = sampled ? &pending->candidates : nullptr,
               .draft_subset = pending->draft_subset},
              error_msg)) {
        return false;
      }
    }
  }
  return true;
}

bool Session::FinishDecode(const DecodeRequest& request,
                           const PendingDecode& pending,
                           std::string* error_msg) {
  auto& sampler = *request.sampler;
  auto* result = request.result;
  auto& exec = *model_->executor_;
  const auto& chain = pending.chain;
  const auto& proposals = pending.proposals;
  const auto base = pending.base;
  const bool sampled = pending.sampled;
  const bool gpu_greedy = pending.gpu_greedy;
  const bool gpu_verification = pending.gpu_verification;
  const auto anchor = chain.front();
  const auto k = static_cast<std::uint32_t>(chain.size());
  const auto vocab = model_->VocabSize();
  const auto is_stop = [&](std::int32_t token) {
    return request.stop_at_eos && model_->IsStopToken(token);
  };
  if (!pending.speculative) {
    draft_length_.ObserveArToken();
    hidden_base_ = base;
    tokens_.push_back(anchor);
    sampler.Accept(static_cast<sampling::TokenId>(anchor));
    result->tokens.push_back(anchor);
    return true;
  }
  sampler.Accept(static_cast<sampling::TokenId>(anchor));
  std::array<rocm::ArgmaxCandidate, rocm::Executor::kWideRows - 1> greedy{};
  if (gpu_greedy &&
      !exec.GreedyMtpPredictions(std::span(greedy).first(k - 1), sampler,
                                 std::span(chain).subspan(1), error_msg)) {
    return false;
  }
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  std::optional<std::int32_t> stop_token;
  bool cpu_rows = false;
  const auto download_remaining = [&](std::uint32_t first) {
    verify_logits_.resize(std::max<std::size_t>(exec.max_speculative(), k) *
                          vocab);
    auto rows =
        std::span(verify_logits_).subspan(first * vocab, (k - first) * vocab);
    if (!exec.ReadVerificationRows(first, rows, error_msg))
      return false;
    cpu_rows = true;
    return true;
  };
  std::array<std::span<const sampling::LogitCandidate>,
             rocm::Executor::kWideRows - 1>
      candidates{};
  if (gpu_verification && !gpu_greedy) {
    // Accepting the anchor may activate a tool grammar, reasoning budget,
    // or exceed the bounded penalty workspace. Keep the complete-row path.
    if (!rocm::Executor::CanCompactMtpVerification(sampler) ||
        (k > exec.max_speculative() &&
         !rocm::Executor::CanCompactWideVerification(sampler, k))) {
      if (!download_remaining(0))
        return false;
    } else if (!exec.SampledMtpCandidates(std::span(candidates).first(k - 1),
                                          sampler, std::span(chain).subspan(1),
                                          error_msg)) {
      return false;
    }
  }
  while (keep < k) {
    if (gpu_greedy && !cpu_rows) {
      const auto& prediction = greedy[keep - 1];
      if (std::getenv("GUFO_TP_TRACE") != nullptr) {
        static std::uint64_t g_seq = 0;
        std::fprintf(stderr,
                     "[vg] seq=%llu keep=%zu draft=%d idx=%d val=%.6e "
                     "match=%d\n",
                     static_cast<unsigned long long>(g_seq++), keep,
                     chain[keep], prediction.index, prediction.value,
                     prediction.index == chain[keep] ? 1 : 0);
        std::fflush(stderr);
      }
      if (!std::isfinite(prediction.value)) {
        AssignError(error_msg, "logit distribution contains no finite values");
        return false;
      }
      if (!sampler.CanSelectArgmax(prediction.index,
                                   /*penalties_applied=*/true)) {
        // Most native tool tokens already obey the grammar. On the first
        // forbidden argmax, download the remaining rows once and use exact
        // masked selection. Avoid one synchronization per rejected candidate.
        // SelectBatchLogits has already installed this session's row offset.
        if (!download_remaining(keep - 1))
          return false;
        continue;
      }
      if (is_stop(prediction.index)) {
        stop_token = prediction.index;
        result->stop = true;
        break;
      }
      if (prediction.index != chain[keep]) {
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(prediction.index));
      ++keep;
      continue;
    }
    if (sampled) {
      if (gpu_verification && !cpu_rows && sampler.NeedsConstraintMask() &&
          !download_remaining(keep - 1))
        return false;
      const auto verified =
          gpu_verification && !cpu_rows
              ? VerifyMtpProposal(candidates[keep - 1], vocab,
                                  proposals[keep - 1], sampler)
              : VerifyMtpProposal(std::span<const float>(verify_logits_)
                                      .subspan((keep - 1) * vocab, vocab),
                                  proposals[keep - 1], sampler);
      const auto token = static_cast<std::int32_t>(verified.token);
      const bool accepted = verified.accepted;
      if (is_stop(token)) {
        stop_token = token;
        result->stop = true;
        break;
      }
      if (!accepted) {
        correction = token;
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      ++keep;
      continue;
    }
    if (std::getenv("GUFO_TP_TRACE") != nullptr) {
      static std::uint64_t vd_seq = 0;
      const auto& row =
          std::span<const float>(verify_logits_).subspan((keep - 1) * vocab, vocab);
      std::int32_t am = -1;
      float best = -1e30F, second = -1e30F;
      for (std::uint32_t i = 0; i < vocab; ++i) {
        if (row[i] > best) {
          second = best;
          best = row[i];
          am = static_cast<std::int32_t>(i);
        } else if (row[i] > second) {
          second = row[i];
        }
      }
      std::fprintf(stderr,
                   "[vd] seq=%llu keep=%zu draft=%d argmax=%d best=%.6e "
                   "second=%.6e gap=%.3e accept_hint=%d\n",
                   static_cast<unsigned long long>(vd_seq++), keep,
                   chain[keep], am, best, second, best - second,
                   chain[keep] == am ? 1 : 0);
      std::fflush(stderr);
    }
    const auto decision = VerifyDraft(std::span<const float>(verify_logits_)
                                          .subspan((keep - 1) * vocab, vocab),
                                      chain[keep], sampler, is_stop);
    if (decision != DraftDecision::kAccept) {
      result->stop = decision == DraftDecision::kStop;
      if (result->stop && sampler.config().logprobs) {
        auto copy = sampler;
        stop_token = static_cast<std::int32_t>(
            copy.Sample(std::span<const float>(verify_logits_)
                            .subspan((keep - 1) * vocab, vocab)));
      }
      break;
    }
    ++keep;
  }
  if (sampler.config().logprobs) {
    const auto rows = keep - 1 + (stop_token ? 1 : 0);
    std::vector<float> raw(rows * vocab);
    if (rows && !exec.ReadVerificationRows(0, raw, error_msg))
      return false;
    for (std::uint32_t i = 1; i < keep; ++i)
      result->logprobs.push_back(*sampling::CollectLogprobs(
          std::span<const float>(raw).subspan((i - 1) * vocab, vocab), chain[i],
          sampler.config().logprobs));
    if (stop_token)
      result->terminal_logprobs = sampling::CollectLogprobs(
          std::span<const float>(raw).subspan((keep - 1) * vocab, vocab),
          *stop_token, sampler.config().logprobs);
  }
  if (!exec.Rollback(
          *session_, keep, error_msg,
          gpu_verification && !cpu_rows ? logits_.data() : nullptr)) {
    return false;
  }
  if (!gpu_verification || cpu_rows) {
    std::copy_n(verify_logits_.data() + (keep - 1) * vocab, vocab,
                logits_.begin());
  }
  hidden_base_ = base;
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  result->tokens.assign(chain.begin(), chain.begin() + keep);
  stats_.cycles += 1;
  stats_.drafted += k - 1;
  stats_.accepted += keep - 1;
  stats_.small_head_cycles += pending.draft_subset;
  // Copied proposals trail the MTP drafts. After a fully accepted MTP part,
  // the length controller also counts the accepted copied tokens as accepted
  // drafts: a copy that ended drafting early would otherwise make a perfect
  // chain look short, and the controller would probe shallower afterwards. A
  // rejected copy says nothing about the MTP head, so it is not a failure.
  const std::uint32_t mtp_drafts = k - 1 - pending.lookup;
  const std::uint32_t mtp_accepted = std::min(keep - 1, mtp_drafts);
  stats_.lookup += pending.lookup;
  stats_.lookup_accepted += keep - 1 - mtp_accepted;
  if (pending.lookup != 0) {
    lookup_.Observe(pending.lookup_start, pending.lookup,
                    keep - 1 - mtp_accepted);
    // Only a copy-only cycle that may verify wide sets the width
    // (PrepareDecode): two whole copies in a row at the current width
    // double it, and any rejected copy returns it to the ordinary width.
    if (mtp_drafts == 0 && pending.wide) {
      const auto width = std::max(copy_rows_, exec.max_speculative());
      if (keep < k) {
        copy_rows_ = exec.max_speculative();
        copy_streak_ = 0;
      } else if (k >= width && ++copy_streak_ >= 2) {
        copy_rows_ = std::min(
            2 * width, std::max(exec.max_copy_rows(), exec.max_speculative()));
        copy_streak_ = 0;
      }
    }
  }
  const bool mtp_all = mtp_accepted == mtp_drafts;
  const std::uint32_t observed = mtp_all ? keep - 1 : mtp_accepted;
  // A target stop ends the request; it does not classify the remaining
  // proposals as failed predictions.
  draft_length_.Observe(
      observed, result->stop || mtp_all ? observed : mtp_drafts, base,
      pending.policy_concurrency, pending.sampled_subset_cost);

  // The next call knows the next sampled anchor. Defer draft catch-up until
  // then, retaining this session's target hidden rows across interleaving.
  exec.MtpRewind(*session_, base);
  if (keep > exec.max_speculative()) {
    // Snapshots and batched cycles keep at most max_speculative() hidden
    // rows. Catch the draft block up to the last kept row now (K/V only,
    // as for prompt rows) and keep that row's hidden.
    const std::vector<std::int32_t> replay(tokens_.begin() + base + 1,
                                           tokens_.begin() + base + keep);
    if (!exec.MtpForward(*session_, replay, 0, {.kv_only = true}, error_msg) ||
        !exec.KeepHiddenRow(*session_, keep - 1, error_msg))
      return false;
    hidden_base_ = base + keep - 1;
  }
  if (correction) {
    // Evaluate the residual as the next cycle's anchor, avoiding a separate
    // target pass. Preserve the actual draw: resampling p would be biased.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos,
                         std::uint32_t occupancy,
                         std::optional<std::uint32_t> forced_drafts) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeExecution();
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  valid_ = false;
  const DecodeRequest request{this,        max_tokens, &sampler,     result,
                              stop_at_eos, nullptr,    forced_drafts};
  PendingDecode pending;
  if (!PrepareDecode(request, &pending, error_msg, false, std::nullopt,
                     occupancy)) {
    return false;
  }
  if (pending.chain.empty()) {
    valid_ = true;
    return true;
  }
  if (pending.lookup != 0)
    model_->batch_policy_.SkipCycle();
  float* logits = !pending.speculative       ? logits_.data()
                  : pending.gpu_verification ? nullptr
                                             : verify_logits_.data();
  auto& exec = *model_->executor_;
  if (pending.chain.size() > exec.max_speculative()) {
    // A wide copy (PrepareDecode).
    if (!pending.gpu_verification)
      verify_logits_.resize(std::max(
          verify_logits_.size(), pending.chain.size() * model_->VocabSize()));
    if (std::getenv("GUFO_TP_TRACE") != nullptr) {
      static std::uint64_t wwd_seq = 0;
      std::fprintf(stderr, "[vw] seq=%llu chain=%zu wide=1 gpu_ver=%d\n",
                   static_cast<unsigned long long>(wwd_seq++),
                   pending.chain.size(), pending.gpu_verification ? 1 : 0);
      std::fflush(stderr);
    }
    if (!exec.ForwardWide(*session_, pending.chain, error_msg) ||
        !exec.SelectBatchLogits(
            0, static_cast<std::uint32_t>(pending.chain.size()),
            pending.gpu_verification ? nullptr : verify_logits_.data(),
            error_msg))
      return false;
  } else {
    if (std::getenv("GUFO_TP_TRACE") != nullptr) {
      static std::uint64_t vw_seq = 0;
      std::fprintf(stderr,
                   "[vw] seq=%llu chain=%zu wide=%d gpu_ver=%d toks=%zu "
                   "forced=%d\n",
                   static_cast<unsigned long long>(vw_seq++),
                   pending.chain.size(), 0,
                   pending.gpu_verification ? 1 : 0, tokens_.size(),
                   forced_drafts.has_value()
                       ? static_cast<int>(*forced_drafts)
                       : -1);
      std::fflush(stderr);
    }
    if (!exec.Forward(
            *session_, pending.chain, pending.chain.size(), logits,
            pending.speculative ? rocm::Executor::ForwardMode::kVerify
                                : rocm::Executor::ForwardMode::kDecode,
            error_msg, {}, pending.chain)) {
      return false;
    }
  }
  const bool ok = FinishDecode(request, pending, error_msg);
  valid_ = ok;
  return ok;
}

std::optional<std::uint32_t> Session::PlanDecode(
    std::size_t max_tokens, const sampling::SamplerState& sampler,
    std::uint32_t occupancy) const {
  if (!valid_ || !MtpEnabled()) {
    return std::nullopt;
  }
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t cap = std::min<std::size_t>(
      {max_tokens, room, model_->executor_->max_speculative()});
  if (cap <= 1) {
    return std::nullopt;
  }
  // Priced as PrepareDecode prices an unplanned cycle. The head is chosen
  // after the anchor is drawn, which only the cycle itself can do, so a
  // sampled request that is eligible for the small head is planned on its
  // curve; rank 1 runs the width chosen here instead of deciding its own.
  const bool sampled = sampler.config().uses_random_sampling();
  const bool gpu_verification =
      sampler.config().temperature == 0.0F ||
      rocm::Executor::CanCompactMtpVerification(sampler);
  const bool subset_cost =
      sampled && gpu_verification && model_->draft_subset_ &&
      std::min(max_tokens, room) > 2 * model_->executor_->max_speculative();
  return draft_length_.Choose(
      static_cast<std::uint32_t>(cap - 1),
      static_cast<std::uint32_t>(tokens_.size()),
      MtpPolicyConcurrency(sampled, sampler.config().seed >= 0, false,
                           occupancy),
      subset_cost);
}

template<class Request>
bool Session::RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg,
                               const std::optional<std::uint32_t>* plan) {
  if (requests.empty() || requests.size() > 8) {
    AssignError(error_msg, "batch must contain 1..8 sessions");
    return false;
  }
  std::array<BatchOutcome, 8> outcomes{};
  std::array<std::uint64_t, 8> epochs{};
  std::array<sampling::SamplerState::DrawState, 8> draws{};
  std::array<speculative::PromptLookup<>::State, 8> lookup_states{};
  std::vector<Request> active;
  active.reserve(requests.size());
  Model* model = nullptr;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (r.session)
      epochs[i] = r.session->session_->MutationEpoch();
    bool valid = r.session && r.session->valid_;
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      valid = valid && r.sampler && r.result && r.max_tokens > 0 &&
              !r.session->tokens_.empty();
    } else {
      valid = valid && r.token >= 0 &&
              static_cast<std::uint32_t>(r.token) <
                  r.session->model_->VocabSize() &&
              r.session->Position() < r.session->ContextSize();
    }
    for (std::size_t j = 0; j < requests.size(); ++j) {
      if (i == j)
        continue;
      valid = valid && r.session != requests[j].session;
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        valid = valid && r.sampler != requests[j].sampler &&
                r.result != requests[j].result;
    }
    if (valid && model && r.session->model_.get() != model)
      valid = false;
    if (!valid) {
      outcome.error = "invalid or non-independent batch request";
      continue;
    }
    try {
      if (r.session->continuation_hooks_)
        r.session->continuation_hooks_->BeforeExecution();
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
      continue;
    }
    if (!r.session->session_->CheckCancellation(&outcome.error))
      continue;
    model = r.session->model_.get();
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      draws[i] = r.sampler->SaveDrawState();
      lookup_states[i] = r.session->lookup_.Save();
    }
    auto copy = r;
    copy.outcome = &outcome;
    active.push_back(copy);
  }
  std::string shared_error;
  try {
    if (!active.empty()) {
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        (void)DecodeBatchImpl(active, plan, &shared_error);
      else
        (void)EvaluateBatchImpl(active, &shared_error);
    }
  } catch (const std::exception& exception) {
    shared_error = exception.what();
  }
  bool success = true;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (!outcome.completed && outcome.error.empty()) {
      auto& session = *r.session;
      if (session.session_->Cancelled()) {
        outcome.error = "generation cancelled";
      } else if (session.session_->MutationEpoch() == epochs[i]) {
        // A preparation/batch-allocation failure did not touch this peer.
        // Retry it independently, without changing its cached frontier.
        session.valid_ = true;
        if constexpr (std::is_same_v<Request, DecodeRequest>) {
          r.sampler->RestoreDrawState(draws[i]);
          session.lookup_.Restore(lookup_states[i]);
        }
        try {
          if constexpr (std::is_same_v<Request, DecodeRequest>) {
            // The retry must run the cycle width the batch planned, or a
            // TP2 peer's own policy could pick another and meet a different
            // collective size.
            const auto forced =
                r.drafts ? r.drafts
                         : (plan != nullptr ? *plan
                                            : std::optional<std::uint32_t>{});
            outcome.completed =
                session.DecodeStep(r.max_tokens, *r.sampler, r.result,
                                   &outcome.error, r.stop_at_eos, 1, forced);
          } else {
            outcome.completed = session.Evaluate(r.token, &outcome.error);
          }
        } catch (const std::exception& exception) {
          outcome.error = exception.what();
        }
      } else {
        outcome.error =
            shared_error.empty() ? "batch execution failed" : shared_error;
      }
    }
    if (!outcome.completed) {
      if (outcome.error.empty())
        outcome.error = "batch request failed";
      if (r.session && r.session->session_->MutationEpoch() != epochs[i])
        r.session->valid_ = false;
      if (success)
        AssignError(error_msg, outcome.error);
      success = false;
    } else {
      r.session->valid_ = true;
    }
    if (r.outcome)
      *r.outcome = std::move(outcome);
  }
  return success;
}

bool Session::DecodeBatch(std::span<const DecodeRequest> requests,
                          std::string* error_msg,
                          const std::optional<std::uint32_t>* plan) {
  return RunIsolatedBatch(requests, error_msg, plan);
}

std::optional<std::uint32_t> Session::PlanBatch(
    std::span<const DecodeRequest> requests) {
  if (requests.size() > 8)
    return std::nullopt;
  // Only the requests the batch would run: a TP2 peer plans before the batch
  // sets aside invalid ones.
  std::array<MtpBatchController::Row, 8> rows{};
  std::size_t count = 0;
  std::uint32_t batch_context = 0;
  const Session* first = nullptr;
  for (const auto& r : requests) {
    if (r.session == nullptr || r.sampler == nullptr || !r.session->valid_ ||
        r.session->tokens_.empty() || r.max_tokens == 0)
      continue;
    if (!r.session->MtpEnabled() || r.sampler->config().uses_random_sampling())
      return std::nullopt;
    const auto& exec = *r.session->model_->executor_;
    const auto cap = std::min<std::size_t>(
        r.max_tokens, r.session->ContextSize() - r.session->Position());
    rows[count++] = {&r.session->draft_length_,
                     static_cast<std::uint32_t>(
                         std::min<std::size_t>(cap, exec.max_speculative())) -
                         (cap != 0)};
    batch_context = std::max(batch_context, r.session->Position());
    if (first == nullptr)
      first = r.session;
  }
  if (count < 2)
    return std::nullopt;
  return first->model_->batch_policy_.Choose(std::span(rows).first(count),
                                             batch_context);
}

bool Session::EvaluateBatch(std::span<const AdvanceRequest> requests,
                            std::string* error_msg) {
  return RunIsolatedBatch(requests, error_msg);
}

bool Session::DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              const std::optional<std::uint32_t>* plan,
                              std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    // The forced draft count (a TP2 peer's plan) overrides this session's
    // acceptance-history policy, exactly as in the batched path.
    const auto forced =
        r.drafts ? r.drafts
                 : (plan != nullptr ? *plan : std::optional<std::uint32_t>{});
    r.outcome->completed =
        r.session->DecodeStep(r.max_tokens, *r.sampler, r.result,
                              &r.outcome->error, r.stop_at_eos, 1, forced);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  std::optional<std::uint32_t> batch_drafts;
  std::uint32_t batch_context = 0;
  auto& policy = requests.front().session->model_->batch_policy_;
  // A given plan (a TP2 peer's) replaces the timing-based choice.
  batch_drafts = plan != nullptr ? *plan : PlanBatch(requests);
  if (batch_drafts) {
    for (const auto& r : requests) {
      batch_context = std::max(batch_context, r.session->Position());
    }
  }
  const auto cycle_start = std::chrono::steady_clock::now();
  std::vector<PendingDecode> pending(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    try {
      if (!r.session->PrepareDecode(
              r, &pending[i], &r.outcome->error, true, batch_drafts,
              static_cast<std::uint32_t>(requests.size())))
        pending[i] = {};
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
      pending[i] = {};
    }
  }
  std::vector<AdvanceRequest> catchup;
  std::vector<std::uint8_t> propose;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (!pending[i].chain.empty()) {
      catchup.push_back({requests[i].session, pending[i].chain.front()});
      propose.push_back(pending[i].speculative &&
                        pending[i].chain.size() < pending[i].width);
    }
  }
  if (!DraftCatchUpBatch(catchup, error_msg, propose))
    return false;
  // Each round shares predictor projections across ready sessions.
  // Attention state, proposal distributions and RNG streams stay private.
  for (;;) {
    std::vector<rocm::Executor::MtpHeadItem> heads;
    std::vector<rocm::Executor::MtpBatchItem> bodies;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (!requests[i].session->session_->Cancelled() && p.speculative &&
          p.chain.size() < p.width) {
        heads.push_back(
            {requests[i].session->session_.get(),
             {.token = p.sampled ? nullptr : &p.draft,
              .confidence =
                  p.sampled || !p.gated ? nullptr : &p.draft_confidence,
              .candidates = p.sampled ? &p.candidates : nullptr,
              .draft_subset = p.draft_subset}});
      }
    }
    if (heads.empty())
      break;
    if (!exec.MtpHeads(heads, error_msg))
      return false;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (requests[i].session->session_->Cancelled() || !p.speculative ||
          p.chain.size() >= p.width)
        continue;
      AppendDraft(p);
      auto& session = *requests[i].session;
      if (requests[i].stop_at_eos &&
          session.model_->IsStopToken(p.chain.back())) {
        p.width = p.chain.size();
        continue;
      }
      if (session.AppendLookup(p, requests[i].stop_at_eos))
        continue;
      if (GateStops(p)) {
        if (batch_drafts && !p.sampled &&
            p.draft_confidence < kMtpBatchDropProbability) {
          // Without proposals left the anchor decodes as an ordinary step;
          // the catch-up already wrote its predictor row.
          p.chain.pop_back();
          p.speculative = p.chain.size() > 1;
        }
        p.width = p.chain.size();
        continue;
      }
      if (p.chain.size() < p.width)
        bodies.push_back({requests[i].session->session_.get(),
                          std::span<const std::int32_t>(&p.draft, 1), -1});
    }
    if (!bodies.empty() && !exec.MtpForwardBatch(bodies, error_msg))
      return false;
  }
  // In an all-greedy cohort, full copies can widen verification far beyond
  // the neural planner's chosen shape and delay peers producing new text.
  // Keep full width when every active row copies. In a mixed cohort, bound
  // copies to half the verifier, or the already planned width if larger.
  // Only copied suffixes are trimmed; sampled proposal/RNG streams are fixed.
  const bool mixed_lookup =
      batch_drafts && std::ranges::any_of(pending, [](const auto& p) {
        return !p.chain.empty() && p.lookup == 0;
      });
  if (mixed_lookup) {
    for (auto& p : pending) {
      if (p.lookup == 0)
        continue;
      const auto cap =
          std::max<std::size_t>(1 + *batch_drafts, (p.lookup_cap + 1) / 2);
      if (p.chain.size() > cap) {
        p.lookup -= static_cast<std::uint32_t>(p.chain.size() - cap);
        p.chain.resize(cap);
        p.width = cap;
      }
    }
  }
  std::vector<rocm::Executor::BatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!r.session->session_->Cancelled() && !pending[i].chain.empty()) {
      items.push_back({r.session->session_.get(), pending[i].chain,
                       pending[i].speculative});
    }
  }
  if (!items.empty() && !exec.ForwardBatch(items, error_msg))
    return false;
  std::uint32_t offset = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& p = pending[i];
    const auto& r = requests[i];
    auto& session = *r.session;
    const bool included =
        std::any_of(items.begin(), items.end(), [&](const auto& item) {
          return item.session == session.session_.get();
        });
    const auto row_offset = offset;
    if (included)
      offset += p.chain.size();
    if (!r.outcome->error.empty())
      continue;
    if (session.session_->Cancelled()) {
      r.outcome->error = "generation cancelled";
      continue;
    }
    if (p.chain.empty()) {
      r.outcome->completed = true;
      continue;
    }
    float* logits = !p.speculative       ? session.logits_.data()
                    : p.gpu_verification ? nullptr
                                         : session.verify_logits_.data();
    try {
      r.outcome->completed =
          exec.SelectBatchLogits(row_offset, p.chain.size(), logits,
                                 &r.outcome->error) &&
          session.FinishDecode(r, p, &r.outcome->error);
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
    }
  }
  const bool used_lookup =
      std::ranges::any_of(pending, [](const auto& p) { return p.lookup != 0; });
  if (used_lookup)
    policy.SkipCycle();
  if (!used_lookup && batch_drafts && items.size() == requests.size() &&
      std::ranges::all_of(requests,
                          [](const auto& r) { return r.outcome->completed; })) {
    const auto ms = std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - cycle_start)
                        .count();
    policy.Observe(requests.size(), batch_context, *batch_drafts, ms);
  }
  return true;
}

bool Session::EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->Evaluate(r.token, &r.outcome->error);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  if (!DraftCatchUpBatch(requests, error_msg))
    return false;
  std::vector<rocm::Executor::BatchItem> items;
  for (const auto& r : requests) {
    items.push_back({r.session->session_.get(), {&r.token, 1}, false});
  }
  if (!exec.ForwardBatch(items, error_msg)) {
    return false;
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    auto& outcome = *requests[i].outcome;
    if (session.session_->Cancelled()) {
      outcome.error = "generation cancelled";
      continue;
    }
    try {
      if (!exec.SelectBatchLogits(i, 1, session.logits_.data(), &outcome.error))
        continue;
      session.hidden_base_ = static_cast<std::uint32_t>(session.tokens_.size());
      session.tokens_.push_back(requests[i].token);
      outcome.completed = true;
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
    }
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
