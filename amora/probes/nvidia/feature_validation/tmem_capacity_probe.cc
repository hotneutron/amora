// Independent audit of capacity and reclamation against NVIDIA's TMEM contract.
// Compile directly with the pinned simulator's tmem.cc; no simulator source edits.
#include "gpgpu-sim/flash/tcgen05/tmem.h"
#include <iostream>
using namespace flash_gpgpu_sim;

int main() {
  tcgen05_tmem_manager_t manager;
  tcgen05_tmem_scope_t scope{0, 0, 1};
  auto first = manager.alloc(scope, 512);
  auto second = manager.alloc(scope, 512);
  std::cout << "two_live_512_column_allocations first=" << first
            << " second=" << second
            << " count=" << manager.allocation_count(scope) << "\n";
  manager.write_words(scope, second, {0x12345678});
  std::cout << "column_512_readback=" << manager.read_words(scope, second, 1)[0] << "\n";
  manager.dealloc(scope, first, 512);
  manager.dealloc(scope, second, 512);
  auto after = manager.alloc(scope, 512);
  std::cout << "allocation_after_dealloc base=" << after << "\n";
  auto outside_row = tcgen05_encode_tmem_address(128, after);
  manager.write_words(scope, outside_row, {0x87654321});
  std::cout << "row_128_readback=" << manager.read_words(scope, outside_row, 1)[0] << "\n";
  manager.dealloc(scope, after, 512);
}

