#include "src/cli/serve/tp_disk_store.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using gufo::server::ChatRequest;
using gufo::server::TextDecodeSelection;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillStep;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerSnapshot;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;
using gufo::server::TpDiskStore;

constexpr std::size_t kPayloadBytes = 8;
constexpr std::size_t kHeaderBytes = 96;
constexpr std::size_t kEntryBytes = kHeaderBytes + kPayloadBytes;
constexpr std::uint32_t kPayloadVersion = 3;

void Require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

template<typename Call>
bool Throws(Call&& call) {
  try {
    call();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "gufo-tp-disk-XXXXXX")
            .string();
    if (::mkdtemp(pattern.data()) == nullptr)
      throw std::runtime_error("failed to create temporary directory");
    path_ = pattern;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

class FakeState final : public TextRunnerState {
public:
  void Invalidate() noexcept override { value = 0; }
  std::uint64_t value{0};
};

class FakeSnapshot final : public TextRunnerSnapshot {
public:
  explicit FakeSnapshot(std::uint64_t snapshot_value) : value(snapshot_value) {}
  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return kPayloadBytes;
  }
  std::uint64_t value;
};

/// Persists one 64-bit value; only the persistence calls are used.
class FakeRunner final : public TextModelRunner {
public:
  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "fake",
            .state_abi = "fake-v1",
            .max_context = 16,
            .capabilities = {},
            .persistence = std::nullopt};
  }
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {};
  }
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view) const override {
    return {};
  }
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken>) const override {
    return {};
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<FakeState>();
  }
  [[nodiscard]] TextPrefillStep Prefill(TextRunnerState&,
                                        std::span<const TextRunnerToken>,
                                        std::size_t,
                                        std::size_t) const override {
    throw std::logic_error("not used");
  }
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState&, gufo::sampling::SamplerState&) const override {
    throw std::logic_error("not used");
  }
  void Advance(TextRunnerState&, TextRunnerToken) const override {
    throw std::logic_error("not used");
  }
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState&) const override {
    return 0;
  }
  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot&) const override {
    return kPayloadBytes;
  }
  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    const auto value = dynamic_cast<const FakeSnapshot&>(snapshot).value;
    std::uint8_t bytes[kPayloadBytes];
    for (std::size_t i = 0; i < kPayloadBytes; ++i)
      bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
    sink(std::span<const std::uint8_t>(bytes, 3));
    sink(std::span<const std::uint8_t>(bytes + 3, kPayloadBytes - 3));
  }
  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    if (payload.size() != kPayloadBytes)
      throw std::invalid_argument("payload size");
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < kPayloadBytes; ++i)
      value |= std::uint64_t{payload[i]} << (8 * i);
    dynamic_cast<FakeState&>(state).value = value;
  }
};

std::unique_ptr<TpDiskStore> Open(const std::filesystem::path& directory,
                                  std::size_t capacity = 1U << 20,
                                  std::string_view identity = "model-a",
                                  std::uint32_t version = kPayloadVersion,
                                  std::size_t staging = 1U << 20) {
  return std::make_unique<TpDiskStore>(
      TpDiskStore::Options{.directory = directory,
                           .capacity_bytes = capacity,
                           .staging_capacity_bytes = staging},
      std::vector<std::uint8_t>(identity.begin(), identity.end()), version);
}

bool Persist(TpDiskStore& store, const std::shared_ptr<FakeRunner>& runner,
             std::uint64_t key, std::uint64_t value) {
  const bool queued =
      store.PersistAsync(key, runner, std::make_shared<FakeSnapshot>(value));
  store.Flush();
  return queued;
}

/// The restored value, or nullopt when Restore threw.
std::optional<std::uint64_t> Restored(TpDiskStore& store,
                                      const FakeRunner& runner,
                                      std::uint64_t key) {
  FakeState state;
  try {
    store.Restore(key, runner, state);
  } catch (const std::exception&) {
    return std::nullopt;
  }
  return state.value;
}

std::filesystem::path File(const std::filesystem::path& directory,
                           const char* name) {
  return directory / name;
}

void TestRoundTripAndRestart(const std::shared_ptr<FakeRunner>& runner) {
  TemporaryDirectory dir;
  {
    auto store = Open(dir.path());
    Require(Persist(*store, runner, 0x1234, 77), "persist is queued");
    Require(store->entry_count() == 1, "one entry");
    Require(store->retained_bytes() == kEntryBytes, "header plus payload");
    Require(Restored(*store, *runner, 0x1234) == 77, "round trip");
    Require(!Restored(*store, *runner, 0x9999), "missing key throws");
  }
  // An unfinished write from a previous run is removed; the entry survives.
  std::ofstream(File(dir.path(), "tp2-0000000000000042.tmp")) << "partial";
  auto store = Open(dir.path());
  Require(
      !std::filesystem::exists(File(dir.path(), "tp2-0000000000000042.tmp")),
      "restart removes temporary files");
  Require(store->entry_count() == 1 && store->retained_bytes() == kEntryBytes,
          "restart indexes the entry");
  Require(Restored(*store, *runner, 0x1234) == 77, "restore after restart");
}

void TestRejectedPersists(const std::shared_ptr<FakeRunner>& runner) {
  TemporaryDirectory dir;
  Require(!Open(dir.path())
               ->PersistAsync(0, runner, std::make_shared<FakeSnapshot>(1)),
          "key 0 is rejected");
  auto small = Open(dir.path(), kEntryBytes - 1);
  Require(!Persist(*small, runner, 1, 1), "entry beyond capacity is skipped");
  auto unstaged = Open(dir.path(), 1U << 20, "model-a", kPayloadVersion, 0);
  Require(!Persist(*unstaged, runner, 1, 1), "staging full skips the write");
  Require(unstaged->entry_count() == 0, "skipped writes leave no entry");
}

void TestForeignAndCorrupt(const std::shared_ptr<FakeRunner>& runner) {
  TemporaryDirectory dir;
  const auto name = File(dir.path(), "tp2-0000000000000007.snap");
  Persist(*Open(dir.path()), runner, 7, 5);
  {
    auto other = Open(dir.path(), 1U << 20, "model-b");
    Require(!Restored(*other, *runner, 7), "another identity is refused");
    Require(other->entry_count() == 0 && !std::filesystem::exists(name),
            "a foreign entry is removed");
  }
  Persist(*Open(dir.path()), runner, 7, 5);
  {
    auto other = Open(dir.path(), 1U << 20, "model-a", kPayloadVersion + 1);
    Require(!Restored(*other, *runner, 7), "another version is refused");
  }
  Persist(*Open(dir.path()), runner, 7, 5);
  {
    std::fstream file(name, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(static_cast<std::streamoff>(kHeaderBytes));
    file.put('\xff');
  }
  auto store = Open(dir.path());
  Require(!Restored(*store, *runner, 7), "a corrupt payload is refused");
  Require(store->entry_count() == 0 && store->retained_bytes() == 0 &&
              !std::filesystem::exists(name),
          "a corrupt entry is removed");
}

void TestLeastRecentEviction(const std::shared_ptr<FakeRunner>& runner) {
  TemporaryDirectory dir;
  auto store = Open(dir.path(), 2 * kEntryBytes);
  Persist(*store, runner, 1, 10);
  Persist(*store, runner, 2, 20);
  Require(Restored(*store, *runner, 1) == 10, "restore touches entry 1");
  Persist(*store, runner, 3, 30);
  Require(
      store->entry_count() == 2 && store->retained_bytes() == 2 * kEntryBytes,
      "capacity bounds the store");
  Require(!Restored(*store, *runner, 2), "the least recent entry is evicted");
  Require(
      !std::filesystem::exists(File(dir.path(), "tp2-0000000000000002.snap")),
      "the evicted file is deleted");
  Require(
      Restored(*store, *runner, 1) == 10 && Restored(*store, *runner, 3) == 30,
      "recent entries stay");
}

void TestRewriteKeepsTheNewFile(const std::shared_ptr<FakeRunner>& runner) {
  TemporaryDirectory dir;
  auto store = Open(dir.path());
  Persist(*store, runner, 5, 1);
  Persist(*store, runner, 5, 2);
  Require(store->entry_count() == 1 && store->retained_bytes() == kEntryBytes,
          "a rewrite replaces the entry");
  Require(
      std::filesystem::exists(File(dir.path(), "tp2-0000000000000005.snap")),
      "a rewrite keeps the file it wrote");
  Require(Restored(*store, *runner, 5) == 2,
          "a rewrite restores the new value");
}

}  // namespace

int main() {
  const auto runner = std::make_shared<FakeRunner>();
  TestRoundTripAndRestart(runner);
  TestRejectedPersists(runner);
  TestForeignAndCorrupt(runner);
  TestLeastRecentEviction(runner);
  TestRewriteKeepsTheNewFile(runner);
  std::cout << "tp_disk_store_test passed\n";
  return 0;
}
