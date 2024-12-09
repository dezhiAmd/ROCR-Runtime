#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include "../utility.hpp"

// Assigns agent to output_iterator
template <typename OutputIterator>
hsa_status_t find_all_agents(hsa_agent_t agent, void* output_iterator) {
  auto& out_it = *static_cast<OutputIterator*>(output_iterator);
  *out_it = agent;
  return HSA_STATUS_SUCCESS;
}

// Returns the size of the memory pool
std::size_t get_pool_size(hsa_amd_memory_pool_t pool) {
  std::size_t size = {};
  HSA_CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size));
  return size;
}

// Returns the flags of the memory pool
std::uint32_t get_pool_flags(hsa_amd_memory_pool_t pool) {
  std::uint32_t flags = {};
  HSA_CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags));
  return flags;
}

// Pretty prints agent memory pool flags
std::string format_pool_flags(std::uint32_t pool_flags) {
  std::string s;
  if (pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
    s += "KERNARG_INIT ";
  }
  if (pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
    s += "FINE_GRAINED ";
  }
  if (pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
    s += "COARSE_GRAINED ";
  }
  if (pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED) {
    s += "EXTENDED_SCOPE_FINE_GRAINED ";
  }
  return s;
}

// Prints the agent name, iterates over its pools and prints some information
void print_agent_name_and_pools(hsa_agent_t agent) {
  std::cout << "Agent name: " << get_agent_name(agent) << '\n';

  std::size_t pool_count = 0;
  HSA_CHECK(hsa_amd_agent_iterate_memory_pools(
      agent,
      [](hsa_amd_memory_pool_t pool, void* pool_count_ptr) {
        auto& pool_count = *static_cast<std::size_t*>(pool_count_ptr);
        auto pool_size = get_pool_size(pool);
        auto pool_flags = get_pool_flags(pool);

        std::cout << "\t Pool " << pool_count << " size: " << pool_size
                  << " flags: " << format_pool_flags(pool_flags) << '\n';

        ++pool_count;

        return HSA_STATUS_SUCCESS;
      },
      &pool_count));
}

int main(int, char*[]) {
  // initialize HSA
  HSA_CHECK(hsa_init());

  // find all agents
  std::vector<hsa_agent_t> agents;
  auto out_it = std::back_inserter(agents);
  HSA_CHECK(hsa_iterate_agents(&find_all_agents<decltype(out_it)>, &out_it));

  // print all agents and their memory pools
  std::for_each(agents.begin(), agents.end(), &print_agent_name_and_pools);

  // finalize HSA
  HSA_CHECK(hsa_shut_down());

  return 0;
}