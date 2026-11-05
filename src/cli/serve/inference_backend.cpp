#include "src/cli/serve/inference_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/text.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/cli/serve/tp_control.hpp"
#include "src/cli/serve/tp_executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/communicator.hpp"


#include "src/models/qwen38_flash_next/rope_scaling.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"

namespace gufo::server {

#if defined(GUFO_HAS_TP2_CONTROL)
/// Binds the communicator's operation scope to the instruction index of the
/// model call running on this rank, so a call's exchanges pair only with the
/// same call's exchanges on the peer, whichever requests are open.
class CommunicatorCallScope final : public TpCallScope {
public:
  explicit CommunicatorCallScope(
      std::shared_ptr<::gufo::models::qwen38_flash_next::rocm::Communicator>
          communicator)
      : communicator_(std::move(communicator)) {
    if (communicator_ == nullptr) {
      throw std::invalid_argument("TP call scope needs a communicator");
    }
  }

  [[nodiscard]] bool Begin(std::uint64_t index, std::string* error) override {
    return communicator_->BeginOperation(index, error);
  }
  [[nodiscard]] bool End(std::uint64_t index, std::string* error) override {
    return communicator_->EndOperation(index, error);
  }

private:
  std::shared_ptr<::gufo::models::qwen38_flash_next::rocm::Communicator>
      communicator_;
};

#endif


namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool DiskCacheEnabled(const TextDiskCacheConfig& config) {
  return !config.directory.empty() && config.capacity_bytes > 0;
}

}  // namespace

std::size_t CountReasoningTokens(
    std::span<const tokenization::TokenId> tokens,
    TextGenerationBackend::InitialOutputState initial,
    const ext::ReasoningMarkup& reasoning,
    const std::function<std::string(tokenization::TokenId)>& decode) {
  using State = TextGenerationBackend::InitialOutputState;
  if (initial == State::kContent || reasoning.close.empty() ||
      (initial == State::kAuto && reasoning.open.empty()))
    return 0;
  // The parsers match delimiters in text, and tokens at their edges can merge
  // with neighbouring output, so match in text too and map back to tokens.
  std::string text;
  std::vector<std::size_t> ends;
  ends.reserve(tokens.size());
  for (const auto token : tokens) {
    text += decode(token);
    ends.push_back(text.size());
  }
  const std::string_view view = text;
  std::size_t begin = 0;
  if (initial == State::kAuto) {
    const auto leading = view.find_first_not_of(" \t\r\n");
    if (leading == std::string_view::npos ||
        !view.substr(leading).starts_with(reasoning.open))
      return 0;
    begin = leading + reasoning.open.size();
  } else if (view.starts_with(reasoning.open)) {
    begin = reasoning.open.size();
  }
  const auto end = std::min(view.find(reasoning.close, begin), view.size());
  if (begin >= end)
    return 0;
  const auto first = std::ranges::upper_bound(ends, begin);
  const auto last = std::lower_bound(first, ends.end(), end);
  return static_cast<std::size_t>(last - first) + 1;
}

struct InferenceBackend::Impl {
#if defined(ENGINE_ENABLE_HIP)
  struct State {
    std::shared_ptr<TextGenerationScheduler> scheduler;
    /// Rank 0 of a TP2 pair: the runner that sends rank 1 each model call.
    std::shared_ptr<TpMirroredRunner> tp_runner;
    /// Rank 1 of a TP2 pair: executes rank 0's model calls. Rank 1 serves no
    /// requests of its own, so it has no scheduler.
    std::shared_ptr<TpExecutor> tp_executor;
    std::shared_ptr<TpControlChannel> control;
    std::shared_ptr<TpResponseBroker> response_broker;
    std::shared_ptr<models::qwen38_flash_next::rocm::Communicator> communicator;
    std::uint32_t tp_rank{0};
    std::uint32_t tp_world_size{1};
    std::string model_id;
    std::vector<std::string> extra_model_ids;
    SamplingDefaults sampling_defaults;
    std::uint32_t max_context{0};
    bool supports_images{false};
    ReasoningOptions reasoning_defaults;
    /// TP2: how many requests may be open at once, running or waiting.
    std::size_t tp_max_requests{0};
    std::string model_path;
    std::string build_info;
  };

#if defined(GUFO_HAS_TP2_CONTROL)
  /// One TP2 request on rank 0, from the command that opens it on rank 1 to
  /// rank 1's verdict. Requests run concurrently: each model call names its
  /// request, so rank 1 keeps every open request apart.
  class TpRequest {
  public:
    TpRequest(std::shared_ptr<const State> state,
              std::shared_ptr<std::atomic<std::size_t>> open,
              std::uint64_t sequence)
        : state_(std::move(state)),
          open_(std::move(open)),
          sequence_(sequence) {}

    TpRequest(const TpRequest&) = delete;
    TpRequest& operator=(const TpRequest&) = delete;

    [[nodiscard]] bool finished() const noexcept { return finished_; }

    /// Ends the request on rank 1 (`kEnd` with rank 0's instruction count and
    /// digest) and collects rank 1's verdict, and the verdicts on any state
    /// the request reused before they arrived. Call once rank 0 makes no
    /// further model calls for the request. Returns the first problem, or an
    /// empty string when both ranks agree.
    [[nodiscard]] std::string Finish() noexcept {
      if (finished_) {
        return {};
      }
      finished_ = true;
      std::string problem;
      bool settled = false;
      try {
        const auto note = [&](std::string message) {
          if (problem.empty()) {
            problem = std::move(message);
          }
        };
        std::string error;
        if (!state_->tp_runner->EndRequest(sequence_, &error)) {
          note(error);
          state_->response_broker->FailAll("TP request end failed: " + error);
        } else {
          TpControlResponse response;
          if (!state_->response_broker->WaitForResponse(sequence_, &response,
                                                        &error)) {
            note("TP worker response failed: " + error);
          } else if (!response.error.empty()) {
            note("TP worker failed: " + response.error);
          }
        }
        if (!state_->tp_runner->AwaitDependencies(sequence_)) {
          note("TP request reused a state rank 1 rejected");
        }
        // A request the ranks disagree about leaves a state that may differ
        // between them; it is never reused, and the cache replaces it in
        // time. Other requests keep theirs.
        state_->tp_runner->Settle(sequence_, problem.empty());
        settled = true;
      } catch (...) {
        try {
          if (problem.empty()) {
            problem = "TP request end failed";
          }
        } catch (...) {
        }
      }
      if (!settled) {
        // Requests that reused this one's state wait for its verdict.
        try {
          state_->tp_runner->Settle(sequence_, false);
        } catch (...) {
        }
      }
      open_->fetch_sub(1);
      return problem;
    }

  private:
    std::shared_ptr<const State> state_;
    std::shared_ptr<std::atomic<std::size_t>> open_;
    std::uint64_t sequence_{0};
    bool finished_{false};
  };

#else
  class TpRequest {
  public:
    [[nodiscard]] bool finished() const noexcept { return true; }
    [[nodiscard]] std::string Finish() { return {}; }
  };
#endif

  class ScheduledGenerationRequest final : public GenerationRequest {
  public:
    ScheduledGenerationRequest(
        std::shared_ptr<const State> model_state,
        TextGenerationScheduler::Request scheduled_request,
        InitialOutputState initial = InitialOutputState::kContent,
        std::optional<sampling::JsonConstraint::ToolFormat> tool_format = {},
        std::string tool_dialect = {},
        std::unique_ptr<TpRequest> tp = {})
        : state_(std::move(model_state)),
          request_(std::move(scheduled_request)),
          tool_format_(tool_format),
          tool_dialect_(std::move(tool_dialect)),
          initial_(initial),
          reasoning_(state_->scheduler->runner().ReasoningDelimiters()),
          tp_(std::move(tp)) {}

    ~ScheduledGenerationRequest() override {
      // A TP2 request whose result was never collected must still end on rank
      // 1, which otherwise keeps waiting for its next model call.
      if (tp_ != nullptr && !tp_->finished()) {
        request_.Cancel();
        try {
          (void)request_.Wait({});
        } catch (...) {
        }
        (void)tp_->Finish();
      }
    }

    Result Wait(const TokenCallback& on_token,
                const ProgressCallback& on_progress,
                const StartCallback& on_start) override {
      std::exception_ptr local_error;
      Result result;
      try {
        result = request_.Wait(on_token, on_progress, on_start);
      } catch (...) {
        local_error = std::current_exception();
      }
      if (tp_ != nullptr) {
        // Rank 0 makes no further model calls for the request once its own
        // wait returns, so the request can end on rank 1 too.
        const auto problem = tp_->Finish();
        if (local_error == nullptr && !problem.empty()) {
          throw std::runtime_error(problem);
        }
      }
      if (local_error != nullptr) {
        std::rethrow_exception(local_error);
      }
      const auto& runner = state_->scheduler->runner();
      result.reasoning_tokens =
          CountReasoningTokens(result.tokens, initial_, reasoning_,
                               [&runner](tokenization::TokenId token) {
                                 return runner.Decode(std::span(&token, 1));
                               });
      return result;
    }

    /// Never blocks: it may run inside `Wait`'s token callback. The scheduler
    /// stops the request between model calls, and `Wait` then ends it.
    void Cancel() noexcept override { request_.Cancel(); }

    ext::ReasoningMarkup ReasoningDelimiters() const override {
      return reasoning_;
    }

    std::optional<sampling::JsonConstraint::ToolFormat> ToolFormat()
        const override {
      return tool_format_;
    }

    std::optional<std::string> ToolDialectId() const override {
      return tool_dialect_;
    }

  private:
    std::shared_ptr<const State> state_;
    TextGenerationScheduler::Request request_;
    const std::optional<sampling::JsonConstraint::ToolFormat> tool_format_;
    const std::string tool_dialect_;
    const InitialOutputState initial_;
    const ext::ReasoningMarkup reasoning_;
    std::unique_ptr<TpRequest> tp_;
  };

  [[nodiscard]] std::shared_ptr<const State> Snapshot() const {
    const std::lock_guard<std::mutex> lock(state_mutex);
    return state;
  }

#if defined(GUFO_HAS_TP2_CONTROL)
  /// Starts a TP2 request on rank 0: tells rank 1 about it, then submits it to
  /// the scheduler, whose model calls the mirrored runner sends to rank 1.
  /// The scheduler decides everything else -- sampling, stop sequences,
  /// cancellation between model calls, streaming -- exactly as on one host.
  [[nodiscard]] std::shared_ptr<ScheduledGenerationRequest> StartTpRequest(
      const std::shared_ptr<const State>& state,
      std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled, bool stream_output,
      const std::string& client_id, std::vector<std::string> stop_sequences,
      InitialOutputState initial, Clock::time_point request_start,
      bool cache_prompt, bool cache_persist, std::size_t cache_prefix_tokens,
      std::shared_ptr<const TextPromptContext> context, bool return_progress,
      std::optional<sampling::JsonConstraint::ToolFormat> tool_format,
      bool stop_at_eos = true,
      std::optional<TpConstraintSource> constraint_source =
          std::nullopt) const {
    if (state->tp_runner == nullptr || state->response_broker == nullptr ||
        state->communicator == nullptr || state->scheduler == nullptr) {
      throw std::logic_error("TP2 rank 0 is not fully configured");
    }
    // The scheduler would refuse these too, but only after rank 1 was told
    // about the request, and a refused command is a failed request rather than
    // a client error.
    if (prompt.size() >= state->max_context) {
      throw std::length_error(
          "prompt has " + std::to_string(prompt.size()) +
          " tokens but the context is " + std::to_string(state->max_context) +
          "; increase --context or shorten the conversation");
    }
    // Resolve the budget as the scheduler does, so rank 1 is told the one
    // rank 0 uses: zero, a request without `max_tokens`, means "until the
    // context is full".
    const std::size_t available = state->max_context - prompt.size();
    if (max_tokens == 0 || max_tokens > available) {
      max_tokens = available;
    }
    if (max_tokens > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument(
          "TP2 token budget exceeds the protocol range");
    }
    if (cache_prefix_tokens > prompt.size()) {
      throw std::invalid_argument("TP cache prefix exceeds request prompt");
    }
    cache_prompt = cache_prompt &&
                   state->tp_runner->Descriptor().capabilities.prefix_reuse;
    TpControlCommand begin{
        .max_tokens = static_cast<std::uint32_t>(max_tokens),
        .cache_prompt = cache_prompt,
        .cache_prefix_tokens = static_cast<std::uint32_t>(cache_prefix_tokens),
        .client_id = client_id,
        .sampling = sampling,
        .constraint_source = std::move(constraint_source),
        .stop_at_eos = stop_at_eos,
    };
    if (sampling.reasoning_budget != nullptr) {
      begin.reasoning_budget = sampling.reasoning_budget->tokens;
    }
    if (context != nullptr) {
      begin.prompt_context = state->tp_runner->EncodePromptContext(*context);
    }
    begin.prompt_tokens.reserve(prompt.size());
    for (const auto token : prompt) {
      if (token > static_cast<TextRunnerToken>(
                      std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            "TP2 prompt token exceeds the protocol range");
      }
      begin.prompt_tokens.push_back(static_cast<std::int32_t>(token));
    }
    // Bounded like the scheduler's queue plus its running requests, so the
    // response broker always has room for the verdict.
    if (tp_open->fetch_add(1) >= state->tp_max_requests) {
      tp_open->fetch_sub(1);
      throw TextGenerationError(TextGenerationErrorCode::kQueueFull,
                                "text generation pending queue is full");
    }
    begin.sequence = tp_sequence.fetch_add(1);
    std::string error;
    if (!state->response_broker->RegisterPendingResponse(begin.sequence,
                                                         &error)) {
      tp_open->fetch_sub(1);
      throw std::runtime_error("TP response registration failed: " + error);
    }
    // Rank 1 must know the request before its first model call, and rank 0's
    // runner before the scheduler can lease it a state. The channel keeps
    // order, so the command reaches rank 1 before any call for the request.
    if (!state->control->SendCommand(begin, &error)) {
      std::string cancel_error;
      (void)state->response_broker->CancelUnsentResponse(begin.sequence,
                                                         &cancel_error);
      tp_open->fetch_sub(1);
      throw std::runtime_error("TP worker command failed: " + error);
    }
    state->tp_runner->BeginRequest(begin.sequence,
                                   begin.constraint_source.has_value());
    auto tp = std::make_unique<TpRequest>(state, tp_open, begin.sequence);
    try {
      auto scheduled = state->scheduler->Submit(
          std::move(prompt), max_tokens, sampling, is_cancelled, stream_output,
          TextRequestMetadata{
              .client_id = client_id,
              .deadline = std::nullopt,
              .request_start = request_start,
              .prompt_context = std::make_shared<TpRequestContext>(
                  begin.sequence, std::move(context)),
              .cache_prompt = cache_prompt,
              .cache_persist = cache_persist,
              .cache_prefix_tokens = cache_prefix_tokens,
              .stop_at_eos = stop_at_eos,
              .stop_sequences = std::move(stop_sequences),
              .return_progress = return_progress,
          });
      return std::make_shared<ScheduledGenerationRequest>(
          state, std::move(scheduled), initial, tool_format, std::string(),
          std::move(tp));
    } catch (...) {
      // Rank 0 made no model call for the request, so it ends on rank 1 with
      // none either.
      (void)tp->Finish();
      throw;
    }
  }

#endif
  Result GenerateScheduled(
      std::shared_ptr<const State> current,
      std::vector<TextRunnerToken> prompt_tokens,
      Clock::time_point request_start, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled, const TokenCallback& on_token,
      std::string client_id,
      std::shared_ptr<const TextPromptContext> context = {},
      bool cache_prompt = true, std::size_t cache_prefix_tokens = 0,
      const std::vector<std::string>& stop_sequences = {},
      InitialOutputState initial = InitialOutputState::kContent,
      bool cache_persist = true) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    result.client_id = client_id.empty() ? "anonymous" : client_id;
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      return result;
    }

#if defined(GUFO_HAS_TP2_CONTROL)
    if (current->control != nullptr) {
      auto generation = StartTpRequest(
          current, std::move(prompt_tokens), max_tokens, sampling, is_cancelled,
          static_cast<bool>(on_token), client_id, stop_sequences, initial,
          request_start, cache_prompt, cache_persist, cache_prefix_tokens,
          std::move(context), false, std::nullopt);
      return generation->Wait(on_token, {}, {});
    }

#endif
    auto request = current->scheduler->Submit(
        std::move(prompt_tokens), max_tokens, sampling, is_cancelled,
        static_cast<bool>(on_token),
        TextRequestMetadata{
            .client_id = std::move(client_id),
            .deadline = std::nullopt,
            .request_start = request_start,
            .prompt_context = std::move(context),
            .cache_prompt = cache_prompt,
            .cache_persist = cache_persist,
            .cache_prefix_tokens = cache_prefix_tokens,
            .stop_sequences = stop_sequences,
        });
    ScheduledGenerationRequest generation(std::move(current),
                                          std::move(request), initial);
    return generation.Wait(on_token, {}, {});
  }

  mutable std::mutex state_mutex;
  /// TP2 requests between their command to rank 1 and its verdict.
  mutable std::shared_ptr<std::atomic<std::size_t>> tp_open{
      std::make_shared<std::atomic<std::size_t>>(0)};
  /// Starts at 1: zero means "no single request" to the instruction protocol.
  mutable std::atomic<std::uint64_t> tp_sequence{1};
  std::shared_ptr<const State> state;
#endif
};

InferenceBackend::InferenceBackend() : impl_(std::make_unique<Impl>()) {}

InferenceBackend::~InferenceBackend() = default;

#if defined(ENGINE_ENABLE_HIP)
bool InferenceBackend::load(
    std::shared_ptr<ext::TextModelRunner> runner, std::string* error,
    std::size_t session_count, TextPrefillPolicy prefill_policy,
    TextSchedulerPolicy scheduler_policy,
    const TextDiskCacheConfig& disk_cache_config,
    ext::TextRunnerRamCacheOptions ram_cache_config,
    const TextTpConfig& tp_config) {
  try {
    const std::uint32_t tp_world_size = tp_config.world_size;
    const std::uint32_t tp_rank = tp_config.rank;
    const bool has_mtp = true;  // TP2 text production always runs MTP
    const bool has_vision = false;
    const TextSpeculativeConfig speculative_config{};
    const bool model_yarn = false;
    const auto desc = runner->Descriptor();
    if (desc.sampling_defaults.has_value()) {
      sampling::ValidateTextSamplingDefaults(*desc.sampling_defaults);
    }
    auto new_state = std::make_shared<Impl::State>();
    new_state->model_path = desc.model_id;
    new_state->sampling_defaults.max_tokens = 0;
    new_state->sampling_defaults.provider =
        desc.sampling_defaults.has_value()
            ? desc.sampling_defaults
            : sampling::PresetDefaults(desc.sampling_preset);
    new_state->sampling_defaults.sampling = sampling::SamplingConfig{};
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = false;  // TODO(v0.28 port): descriptor modalities
    // runner arrives pre-built from the provider path
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = runner->Descriptor().max_context;
#if defined(GUFO_HAS_TP2_CONTROL)
    new_state->tp_max_requests =
        session_count + scheduler_policy.max_pending_requests;
    if (tp_world_size > 1) {
      // Each rank offers its own RAM-cache limit and the pair uses the
      // smaller one, so a limit above the automatic budget needs the same
      // --cache-ram-bytes on both ranks.
      const TpControlConfig control_config{
          .snapshot_budget_bytes =
              ram_cache_config.capacity_bytes == 0
                  ? ext::HostSnapshotBudgetBytes()
                  : std::min<std::uint64_t>(ram_cache_config.capacity_bytes,
                                            ext::HostSnapshotCeilingBytes()),
          .rank = tp_rank,
          .world_size = tp_world_size,
          .max_context = new_state->max_context,
          .max_draft_tokens = has_mtp ? speculative_config.max_draft_tokens : 0,
          .use_mtp = has_mtp,
          .auth_token = tp_config.auth_token,
          .prefill_chunk_tokens =
              static_cast<std::uint32_t>(prefill_policy.decode_active_tokens),
          .sessions = static_cast<std::uint32_t>(session_count),
          .vision = has_vision,
          .disk_cache = DiskCacheEnabled(disk_cache_config),
          .yarn = model_yarn,
      };
      std::string control_error;
      if (!tp_config.control->Handshake(control_config, &control_error)) {
        SetError(error, "TP worker handshake failed: " + control_error);
        return false;
      }
      if (tp_rank == 0) {
        new_state->response_broker = std::make_shared<TpResponseBroker>(
            tp_config.control, new_state->tp_max_requests);
      }
    }
    if (tp_world_size > 1 && tp_rank != 0) {
      // Rank 1 serves nothing itself: it executes rank 0's model calls on as
      // many states as rank 0's pool creates, and builds no pool of its own.
      // The v0.28-era mirrored disk store rode the deleted TextRunnerSnapshot
      // persistence API; TP2 serves with the disk cache off until the
      // component store grows a mirrored protocol.
      new_state->tp_executor = std::make_shared<TpExecutor>(
          std::move(runner), session_count,
          std::make_shared<CommunicatorCallScope>(tp_config.communicator));
    } else {
      std::shared_ptr<TextModelRunner> pool_runner = runner;
      if (tp_world_size > 1) {
        new_state->tp_runner = std::make_shared<TpMirroredRunner>(
            std::move(runner),
            std::make_shared<TpControlInstructionSink>(
                tp_config.control, new_state->response_broker),
            tp_config.control->snapshot_budget_bytes(),
            std::make_shared<CommunicatorCallScope>(tp_config.communicator));
        pool_runner = new_state->tp_runner;
      }
      if (tp_world_size > 1) {
        // The mirrored pool takes the classic continuation branch, whose
        // snapshot restore the component-based Flash-Next runner does not
        // implement; keep TP2 pools snapshot-free.
        ram_cache_config.enabled = false;
        ram_cache_config.capacity_bytes = 0;
      }
#else
    std::shared_ptr<TextModelRunner> pool_runner = runner;
#endif
      std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
      if (DiskCacheEnabled(disk_cache_config)) {
        runner_disk_cache = TextRunnerDiskCacheOptions{
            .directory = std::move(disk_cache_config.directory),
            .capacity_bytes = disk_cache_config.capacity_bytes,
            .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
        };
      }
      auto runner_pool = std::make_shared<TextRunnerPool>(
          std::move(pool_runner), session_count, std::move(runner_disk_cache),
          ram_cache_config);
      new_state->scheduler = std::make_shared<TextGenerationScheduler>(
          std::move(runner_pool), prefill_policy, scheduler_policy);
#if defined(GUFO_HAS_TP2_CONTROL)
    }
#endif
    new_state->control = tp_config.control;
    new_state->communicator = tp_config.communicator;
    new_state->tp_rank = tp_rank;
    new_state->tp_world_size = tp_world_size;

    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}
#else
bool InferenceBackend::load(
    std::shared_ptr<ext::TextModelRunner>, std::string* error, std::size_t,
    TextPrefillPolicy, TextSchedulerPolicy, const TextDiskCacheConfig&,
    ext::TextRunnerRamCacheOptions, const TextTpConfig&) {
  SetError(error, "HIP engine is disabled");
  return false;
}
#endif

bool InferenceBackend::load(const std::string& model_path, std::string* error,
                            std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            const TextSpeculativeConfig& speculative_config,
                            const TextDiskCacheConfig& disk_cache_config,
                            const std::string& vision_model_path,
                            TextRunnerRamCacheOptions ram_cache_config,
                            const TextTpConfig& tp_config, bool enable_yarn) {
#if !defined(GUFO_HAS_TP2_CONTROL)
  if (tp_config.world_size > 1) {
    SetError(error, "TP2 control is unavailable in this build");
    return false;
  }
#endif

  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    SetError(error, "Failed to open GGUF: " + load_error);
    return false;
  }
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  const auto* provider = ext::ProbeTextProvider(*reader);
  if (provider == nullptr || provider->create == nullptr) {
    SetError(error, "No text provider found for model architecture");
    return false;
  }
  ext::TextRunnerOptions options{
      .model_path = model_path,
      .reader = reader,
      .max_context = max_context,
      .session_count = session_count,
      .speculative = speculative_config,
      .vision_model_path = vision_model_path,
      .disk_cache = disk_cache_config,
      .ram_cache = ram_cache_config,
      .enable_yarn = enable_yarn,
      .tp_rank = tp_config.rank,
      .tp_world_size = tp_config.world_size,
      .tp_hip_device = tp_config.hip_device,
      .tp_communicator = tp_config.communicator,
  };
  auto runner = provider->create(options, error);
  if (runner == nullptr) {
    return false;
  }
  return load(std::move(runner), error, session_count, prefill_policy,
              scheduler_policy, disk_cache_config, ram_cache_config,
              tp_config);
}
bool InferenceBackend::run_worker(std::string* error) {
#if defined(ENGINE_ENABLE_HIP) && defined(GUFO_HAS_TP2_CONTROL)
  const auto state = impl_->Snapshot();
  if (state == nullptr || state->control == nullptr ||
      state->communicator == nullptr || state->tp_executor == nullptr ||
      state->tp_world_size != 2 || state->tp_rank != 1) {
    SetError(error, "TP worker requires a loaded rank-1 Flash-Next model");
    return false;
  }
  const auto respond = [control = state->control](
                           const TpControlResponse& response,
                           std::string* respond_error) {
    return control->SendResponse(response, respond_error);
  };
  for (;;) {
    TpControlCommand command;
    std::string control_error;
    if (!state->control->ReceiveCommand(&command, &control_error)) {
      SetError(error, "TP worker command receive failed: " + control_error);
      return false;
    }
    if (std::getenv("GUFO_TP_TRACE") != nullptr) {
      const TpInstruction& ti = command.instruction;
      std::fprintf(stderr,
          "[tptrace-c] recv seq=%llu kind=%u op=%u idx=%llu state=%u "
          "off=%u cnt=%u psize=%u tok=%d snap=%llu nbatch=%zu\n",
          static_cast<unsigned long long>(command.sequence),
          static_cast<unsigned>(command.kind), static_cast<unsigned>(ti.op),
          static_cast<unsigned long long>(ti.index), ti.state, ti.offset,
          ti.count, ti.prompt_size, ti.token,
          static_cast<unsigned long long>(ti.snapshot_id), ti.batch.size());
    }
    if (!state->tp_executor->Execute(command, respond, &control_error)) {
      SetError(error, "TP worker: " + control_error);
      return false;
    }
  }
#else
  SetError(error, "TP worker requires the HIP backend");
  return false;
#endif
}


std::string InferenceBackend::model_id() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_id : "unknown";
#else
  return "unknown";
#endif
}

std::vector<std::string> InferenceBackend::model_ids() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr)
    return {"unknown"};
  std::vector<std::string> ids{state->model_id};
  ids.insert(ids.end(), state->extra_model_ids.begin(),
             state->extra_model_ids.end());
  return ids;
#else
  return {"unknown"};
#endif
}

bool InferenceBackend::ready() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->Snapshot() != nullptr && tp_failure().empty() && !device_lost();
#else
  return false;
#endif
}

std::string InferenceBackend::tp_failure() const {
#if defined(ENGINE_ENABLE_HIP) && defined(GUFO_HAS_TP2_CONTROL)
  const auto state = impl_->Snapshot();
  if (state == nullptr || state->tp_runner == nullptr) {
    return {};
  }
  auto failure = state->tp_runner->Failure();
  if (failure.empty() && state->response_broker != nullptr) {
    failure = state->response_broker->Failure();
  }
  return failure;
#else
  return {};
#endif
}

bool InferenceBackend::device_lost() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  // Rank 1 of a TP2 pair runs no scheduler of its own.
  return state != nullptr && state->scheduler != nullptr &&
         state->scheduler->device_lost();
#else
  return false;
#endif
}

bool InferenceBackend::supports_images() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->supports_images;
#else
  return false;
#endif
}

std::uint32_t InferenceBackend::max_context() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->max_context : 0;
#else
  return 0;
#endif
}

std::vector<InferenceBackend::SessionState> InferenceBackend::session_states()
    const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  // Rank 1 of a TP2 pair runs no scheduler of its own.
  return state != nullptr && state->scheduler != nullptr
             ? state->scheduler->SessionStates()
             : std::vector<SessionState>{};
#else
  return {};
#endif
}

InferenceBackend::SamplingDefaults InferenceBackend::sampling_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->sampling_defaults : SamplingDefaults{};
#else
  return {};
#endif
}

ReasoningOptions InferenceBackend::reasoning_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->reasoning_defaults : ReasoningOptions{};
#else
  return {};
#endif
}

InferenceBackend::InitialOutputState InferenceBackend::initial_output_state(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  // A TP2 rank 1 has no scheduler: it only executes rank 0's calls.
  return state != nullptr && state->scheduler != nullptr
             ? state->scheduler->runner().InitialOutputState(request)
             : InitialOutputState::kAuto;
#else
  (void)request;
  return InitialOutputState::kAuto;
#endif
}

std::string InferenceBackend::model_path() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_path : std::string{};
#else
  return {};
#endif
}

std::string InferenceBackend::build_info() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->build_info : std::string{};
#else
  return {};
#endif
}

void InferenceBackend::set_served_artifact(const std::string& model_path,
                                           const std::string& build_info) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_path = model_path;
  updated->build_info = build_info;
  impl_->state = std::move(updated);
#else
  (void)model_path;
  (void)build_info;
#endif
}

void InferenceBackend::set_model_ids(std::vector<std::string> model_ids) {
#if defined(ENGINE_ENABLE_HIP)
  if (model_ids.empty()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_id = std::move(model_ids.front());
  updated->extra_model_ids.assign(
      std::make_move_iterator(model_ids.begin() + 1),
      std::make_move_iterator(model_ids.end()));
  impl_->state = std::move(updated);
#else
  (void)model_ids;
#endif
}

void InferenceBackend::set_sampling_defaults(
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    sampling::SamplingOverrides supplied) {
#if defined(ENGINE_ENABLE_HIP)
  sampling_config.Validate();
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->sampling_defaults.max_tokens = max_tokens;
  updated->sampling_defaults.sampling = sampling_config;
  updated->sampling_defaults.supplied = supplied;
  impl_->state = std::move(updated);
#else
  (void)max_tokens;
  (void)sampling_config;
  (void)supplied;
#endif
}

void InferenceBackend::set_reasoning_defaults(
    const ReasoningOptions& reasoning) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->reasoning_defaults = reasoning;
  impl_->state = std::move(updated);
#else
  (void)reasoning;
#endif
}

InferenceBackend::Result InferenceBackend::complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool cache_prompt, bool cache_persist) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
  return impl_->GenerateScheduled(
      state, std::move(prompt_tokens), request_start, max_tokens,
      sampling_config, is_cancelled, on_token, std::string(client_id), {},
      cache_prompt, 0, stop_sequences, InitialOutputState::kContent,
      cache_persist);
#else
  (void)cache_prompt;
  (void)cache_persist;
  (void)client_id;
  (void)stop_sequences;
  (void)prompt;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const ChatRequest& request, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto effective_sampling = sampling_config;
  auto constrained = ConstrainChatRequest(request, state->scheduler->runner(),
                                          &effective_sampling);
  const auto& effective_request = constrained ? *constrained : request;
  auto prompt = state->scheduler->runner().PreparePrompt(effective_request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return {};
  }
  const auto initial =
      state->scheduler->runner().InitialOutputState(effective_request);
  if (request.reasoning.budget_tokens &&
      initial != TextGenerationBackend::InitialOutputState::kContent)
    effective_sampling.reasoning_budget =
        state->scheduler->runner().BuildReasoningBudget(
            *request.reasoning.budget_tokens, prompt->tokens, initial);
  return impl_->GenerateScheduled(
      state, std::move(prompt->tokens), request_start, max_tokens,
      effective_sampling, is_cancelled, on_token, request.client_id,
      std::move(prompt->context), request.cache_prompt,
      prompt->cache_prefix_tokens, request.stop_sequences, initial,
      request.cache_persist);
#else
  (void)request;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, bool stream_output, bool ignore_eos,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool return_progress, bool cache_prompt, bool cache_persist) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_complete(
        prompt, max_tokens, sampling_config, is_cancelled, stream_output,
        ignore_eos, client_id, stop_sequences, return_progress, cache_prompt,
        cache_persist);
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
#if defined(GUFO_HAS_TP2_CONTROL)
  if (state->control != nullptr) {
    // Raw completions carry no prompt context; rank 1 still has to execute
    // every model call of the request.
    return impl_->StartTpRequest(
        state, std::move(prompt_tokens), max_tokens, sampling_config,
        is_cancelled, stream_output,
        client_id.empty() ? "anonymous" : std::string(client_id),
        stop_sequences, InitialOutputState::kContent, request_start,
        cache_prompt, cache_persist, 0, nullptr, return_progress, std::nullopt,
        !ignore_eos);
  }
#endif
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt_tokens), max_tokens, sampling_config, is_cancelled,
      stream_output,
      TextRequestMetadata{
          .client_id = client_id.empty() ? "anonymous" : std::string(client_id),
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = {},
          .cache_prompt = cache_prompt,
          .cache_persist = cache_persist,
          .cache_prefix_tokens = 0,
          .stop_at_eos = !ignore_eos,
          .stop_sequences = stop_sequences,
          .return_progress = return_progress,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request));
#else
  return TextGenerationBackend::start_complete(
      prompt, max_tokens, sampling_config, is_cancelled, stream_output,
      ignore_eos, client_id, stop_sequences, return_progress, cache_prompt,
      cache_persist);
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_chat(const ChatRequest& request, std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling_config,
                             const CancellationCheck& is_cancelled,
                             bool stream_output) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  auto effective_sampling = sampling_config;
  std::optional<sampling::JsonConstraint::ToolFormat> tool_format;
  std::string tool_dialect;
  auto constrained =
      ConstrainChatRequest(request, state->scheduler->runner(),
                           &effective_sampling, &tool_format, &tool_dialect);
  const auto& effective_request = constrained ? *constrained : request;
  auto prompt = state->scheduler->runner().PreparePrompt(effective_request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  const auto initial =
      state->scheduler->runner().InitialOutputState(effective_request);
  if (request.reasoning.budget_tokens &&
      initial != TextGenerationBackend::InitialOutputState::kContent)
    effective_sampling.reasoning_budget =
        state->scheduler->runner().BuildReasoningBudget(
            *request.reasoning.budget_tokens, prompt->tokens, initial);
  const std::string client_id =
      request.client_id.empty() ? "anonymous" : request.client_id;
#if defined(GUFO_HAS_TP2_CONTROL)
  if (state->control != nullptr) {
    // Rank 1 rebuilds a constraint from what ConstrainChatRequest built it
    // from, so its multi-token cycles decide what rank 0's do.
    std::optional<TpConstraintSource> constraint_source;
    if (effective_sampling.constraint != nullptr &&
        (request.response_format == nullptr ||
         !request.response_format_json.empty())) {
      auto& source = constraint_source.emplace();
      if (request.response_format != nullptr) {
        source.response_format_json = request.response_format_json;
        source.response_format_responses = request.response_format_responses;
      }
      for (const auto& tool : request.tools)
        source.tools.push_back({.name = tool.name,
                                .description = tool.description,
                                .parameters_json = tool.parameters_json,
                                .definition_json = tool.definition_json});
      source.tool_choice = static_cast<std::uint8_t>(request.tool_choice);
      source.parallel_tool_calls = request.parallel_tool_calls;
      source.reasoning =
          state->scheduler->runner().InitialOutputState(request) ==
          TextGenerationBackend::InitialOutputState::kReasoning;
    }
    return impl_->StartTpRequest(
        state, std::move(prompt->tokens), max_tokens, effective_sampling,
        is_cancelled, stream_output, client_id, request.stop_sequences, initial,
        request_start, request.cache_prompt, request.cache_persist,
        prompt->cache_prefix_tokens, std::move(prompt->context),
        request.return_progress, tool_format, true,
        std::move(constraint_source));
  }
#endif
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt->tokens), max_tokens, effective_sampling, is_cancelled,
      stream_output,
      TextRequestMetadata{
          .client_id = client_id,
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = std::move(prompt->context),
          .cache_prompt = request.cache_prompt,
          .cache_persist = request.cache_persist,
          .cache_prefix_tokens = prompt->cache_prefix_tokens,
          .stop_sequences = request.stop_sequences,
          .return_progress = request.return_progress,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request), initial, tool_format,
      std::move(tool_dialect));
#else
  return TextGenerationBackend::start_chat(request, max_tokens, sampling_config,
                                           is_cancelled, stream_output);
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<tokenization::ChatMessage>& messages,
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled) {
  return chat(ChatRequest{messages}, max_tokens, sampling_config, is_cancelled);
}

std::size_t InferenceBackend::count_tokens(std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return 0;
  }
  return state->scheduler->runner().Tokenize(text).size();
#else
  (void)text;
  return 0;
#endif
}

std::optional<std::size_t> InferenceBackend::count_chat_tokens(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return std::nullopt;
  }
  // The same preparation as chat(): constrained tool or format requests add
  // prompt text before the template is rendered.
  auto sampling = sampling::SamplingConfig{};
  auto constrained =
      ConstrainChatRequest(request, state->scheduler->runner(), &sampling);
  const auto& effective_request = constrained ? *constrained : request;
  return state->scheduler->runner().CountPromptTokens(effective_request);
#else
  (void)request;
  return std::nullopt;
#endif
}

std::optional<std::string> InferenceBackend::render_chat_prompt(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return std::nullopt;
  }
  // The same preparation as count_chat_tokens(), so the text is what chat()
  // would tokenize.
  auto sampling = sampling::SamplingConfig{};
  auto constrained =
      ConstrainChatRequest(request, state->scheduler->runner(), &sampling);
  const auto& effective_request = constrained ? *constrained : request;
  return state->scheduler->runner().RenderPrompt(effective_request);
#else
  (void)request;
  return std::nullopt;
#endif
}

}  // namespace gufo::server
