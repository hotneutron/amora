// SPDX-License-Identifier: Apache-2.0
// Kernel activity timestamps for an unchanged portable CUDA executable.
#include <cupti.h>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

namespace {
std::mutex &guard() { static auto *value = new std::mutex; return *value; }
std::ofstream &output() {
  static auto *value = new std::ofstream(std::getenv("AMORA_CUPTI_JSONL") ? std::getenv("AMORA_CUPTI_JSONL") : "/dev/null");
  return *value;
}
uint64_t dropped = 0;
int error_code = 0;
void check(CUptiResult result) {
  if (result != CUPTI_SUCCESS) error_code = int(result);
}
std::string escape(const char *text) {
  std::string out;
  for (const char *p = text; p && *p; ++p) {
    if (*p == '"' || *p == '\\') out += '\\';
    if (*p == '\n') out += "\\n";
    else out += *p;
  }
  return out;
}
void CUPTIAPI requested(uint8_t **buffer, size_t *size, size_t *records) {
  *size = 8 * 1024 * 1024;
  *buffer = static_cast<uint8_t *>(std::malloc(*size));
  *records = 0;
}
void CUPTIAPI completed(CUcontext context, uint32_t stream, uint8_t *buffer, size_t, size_t valid) {
  std::lock_guard<std::mutex> hold(guard());
  CUpti_Activity *record = nullptr;
  CUptiResult result;
  while ((result = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
    if (record->kind != CUPTI_ACTIVITY_KIND_KERNEL && record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
    const auto *kernel = reinterpret_cast<const CUpti_ActivityKernel12 *>(record);
    output() << "{\"kind\":\"kernel\",\"name\":\"" << escape(kernel->name)
             << "\",\"start_ns\":" << kernel->start << ",\"end_ns\":" << kernel->end
             << ",\"device\":" << kernel->deviceId << ",\"context\":" << kernel->contextId
             << ",\"stream\":" << kernel->streamId << ",\"correlation\":" << kernel->correlationId
             << ",\"grid\":[" << kernel->gridX << ',' << kernel->gridY << ',' << kernel->gridZ
             << "],\"block\":[" << kernel->blockX << ',' << kernel->blockY << ',' << kernel->blockZ << "]}\n";
  }
  if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) check(result);
  size_t missing = 0;
  check(cuptiActivityGetNumDroppedRecords(context, stream, &missing));
  dropped += missing;
  output().flush();
  std::free(buffer);
}
__attribute__((constructor)) void init() {
  if (!std::getenv("AMORA_CUPTI_JSONL")) return;
  output();
  check(cuptiActivityRegisterCallbacks(requested, completed));
  check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
}
__attribute__((destructor)) void finish() {
  if (!std::getenv("AMORA_CUPTI_JSONL")) return;
  check(cuptiActivityFlushAll(1));
  std::lock_guard<std::mutex> hold(guard());
  output() << "{\"kind\":\"complete\",\"dropped_records\":" << dropped << ",\"error_code\":" << error_code << "}\n";
  output().flush();
}
}
