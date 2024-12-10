#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <fcntl.h>

#include <boost/program_options.hpp>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include "../utility.hpp"

// Returns the fine-grained pool of the agent
hsa_amd_memory_pool_t get_fine_grained_pool(hsa_agent_t agent) {
  auto pool_visitor = [](hsa_amd_memory_pool_t pool, void* data) {
    hsa_amd_segment_t segment = {};

    // check if it's global pool
    HSA_CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));
    if (segment != HSA_AMD_SEGMENT_GLOBAL) {
      return HSA_STATUS_SUCCESS;
    }

    // check if it's fine-grained pool
    std::uint32_t flags = 0;
    HSA_CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags));
    if ((flags & (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
                  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED)) != 0x0) {
      *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
      return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
  };

  hsa_amd_memory_pool_t pool = {};
  auto const status = hsa_amd_agent_iterate_memory_pools(agent, pool_visitor, &pool);
  if (status != HSA_STATUS_INFO_BREAK) {
    HSA_CHECK(status);
    return {};
  }
  return pool;
}

// Returns the first CPU agent in the system
hsa_agent_t find_first_cpu_agent() {
  auto cpu_agent_visitor = [](hsa_agent_t agent, void* data) {
    hsa_device_type_t type = {};
    HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type));
    if (type != HSA_DEVICE_TYPE_CPU) {
      return HSA_STATUS_SUCCESS;
    }

    *static_cast<hsa_agent_t*>(data) = agent;
    return HSA_STATUS_INFO_BREAK;
  };

  hsa_agent_t cpu_agent = {};
  if (hsa_iterate_agents(cpu_agent_visitor, &cpu_agent) != HSA_STATUS_INFO_BREAK) {
    std::cerr << "No CPU agent found\n";
    std::exit(EXIT_FAILURE);
  }
  return cpu_agent;
}

// Returns all the GPU agents in the system
std::vector<hsa_agent_t> find_gpu_agents() {
  auto gpu_agent_visitor = [](hsa_agent_t agent, void* data) {
    hsa_device_type_t type = {};
    HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type));
    if (type != HSA_DEVICE_TYPE_GPU) {
      return HSA_STATUS_SUCCESS;
    }

    // only agents with fine-grained pools are valid agents
    auto fine_grained_pool = get_fine_grained_pool(agent);
    if (fine_grained_pool.handle != 0) {
      static_cast<std::vector<hsa_agent_t>*>(data)->push_back(agent);
    }

    return HSA_STATUS_SUCCESS;
  };

  std::vector<hsa_agent_t> gpu_agents;
  HSA_CHECK(hsa_iterate_agents(gpu_agent_visitor, &gpu_agents));
  return gpu_agents;
}

// Returns the page size of the pool
std::size_t get_pool_granule_size(hsa_amd_memory_pool_t pool) {
  std::size_t size = 0;
  HSA_CHECK(
      hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &size));
  return size;
}

// Code object information
struct code_object_info_t {
  hsa_file_t file;
  hsa_code_object_reader_t reader;
  hsa_executable_t executable;
};

// Loads code from kernel_file for the specified agent
code_object_info_t code_object_load(std::filesystem::path const& kernel_file, hsa_agent_t agent) {
  code_object_info_t info = {};

  // open file for read
  info.file = open(kernel_file.c_str(), O_RDONLY);
  if (info.file == -1) {
    std::cerr << "File not found: " << kernel_file << '\n';
    std::exit(EXIT_FAILURE);
  }

  // read code object
  HSA_CHECK(hsa_code_object_reader_create_from_file(info.file, &info.reader));

  HSA_CHECK(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &info.executable));
  HSA_CHECK(
      hsa_executable_load_agent_code_object(info.executable, agent, info.reader, nullptr, nullptr));
  HSA_CHECK(hsa_executable_freeze(info.executable, nullptr));

  return info;
}

// Cleanups all object related to code_object_info_t
void code_object_free(code_object_info_t& info) {
  HSA_CHECK(hsa_executable_destroy(info.executable));
  HSA_CHECK(hsa_code_object_reader_destroy(info.reader));
  close(info.file);
  info = {};
}

// GPU kernel information
struct kernel_info_t {
  std::uint64_t handle;
  std::uint32_t scratch;
  std::uint32_t group;
  std::uint32_t kernarg_size;
};

// Returns the information for the kernel from the executable
kernel_info_t get_kernel(hsa_executable_t executable,
                         std::string const& kernel_name,
                         hsa_agent_t agent) {
  kernel_info_t kernel_info = {};

  hsa_executable_symbol_t symbol = {};
  hsa_status_t status =
      hsa_executable_get_symbol_by_name(executable, kernel_name.c_str(), &agent, &symbol);
  if (status != HSA_STATUS_SUCCESS) {
    HSA_CHECK(hsa_executable_get_symbol_by_name(executable, (kernel_name + ".kd").c_str(), &agent,
                                                &symbol));
  }

  HSA_CHECK(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                           &kernel_info.handle));

  HSA_CHECK(hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &kernel_info.scratch));

  HSA_CHECK(hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &kernel_info.group));

  HSA_CHECK(hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernel_info.kernarg_size));

  return kernel_info;
}

// Submits a packet to the queue
void submit_packet(hsa_queue_t* queue,
                   hsa_signal_t dispatch_signal,
                   kernel_info_t const& kernel,
                   void* args,
                   int workgroup_size_x) {
  // create packet
  std::uint16_t const ndim = 1;
  hsa_kernel_dispatch_packet_t packet = {};
  packet.header = HSA_PACKET_TYPE_INVALID;
  packet.setup = ndim << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet.workgroup_size_x = workgroup_size_x;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = workgroup_size_x;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.private_segment_size = kernel.scratch;
  packet.group_segment_size = kernel.group;
  packet.kernel_object = kernel.handle;
  packet.kernarg_address = args;
  packet.completion_signal = dispatch_signal;

  // reserve write index
  auto const write_index = hsa_queue_add_write_index_screlease(queue, 1);

  // wait until the queue has space
  while ((write_index - hsa_queue_load_read_index_scacquire(queue)) >= queue->size) {
    ;
  }

  // add packet to queue
  auto* queue_buffer = static_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address);
  auto const mask = queue->size - 1;
  queue_buffer[write_index & mask] = packet;

  // mark the packet ready
  constexpr std::uint16_t dispatchPacketheader =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  std::atomic_ref packet_header_ref(queue_buffer[write_index & mask].header);
  packet_header_ref.store(dispatchPacketheader, std::memory_order_release);

  // update doorbell
  hsa_signal_store_screlease(queue->doorbell_signal, write_index);
}

// Find a kernarg_region
hsa_region_t find_kernarg_region(hsa_agent_t gpu_agent) {
  auto kernarg_region_visitor = [](hsa_region_t region, void* data) {
    hsa_region_segment_t segment = {};
    HSA_CHECK(hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment));
    if (HSA_REGION_SEGMENT_GLOBAL != segment) {
      return HSA_STATUS_SUCCESS;
    }

    hsa_region_global_flag_t flags = {};
    HSA_CHECK(hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags));
    if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
      *static_cast<hsa_region_t*>(data) = region;
      return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
  };

  hsa_region_t kernarg_region = {};
  auto const status = hsa_agent_iterate_regions(gpu_agent, kernarg_region_visitor, &kernarg_region);
  if (status != HSA_STATUS_INFO_BREAK) {
    HSA_CHECK(status);
    std::cerr << "No region found on the GPU agent to store kernel args\n";
    std::exit(EXIT_FAILURE);
  }
  return kernarg_region;
}

// Creates a queue for each of the agents, runs the kernel and waits for it to
// finish
void run_add_one(code_object_info_t const& code_object,
                 std::vector<hsa_agent_t> const& gpu_agents,
                 std::uint32_t* buffer,
                 std::size_t element_count) {
  // add_one kernel arguments
  struct alignas(16) args_t {
    std::uint32_t n;
    std::uint32_t* buffer;
  };

  static const std::string kernel_name = "add_one";

  for (auto const& gpu_agent : gpu_agents) {
    std::cout << "Running on: " << get_agent_name(gpu_agent) << '\n';

    // create queue
    hsa_queue_t* queue = nullptr;
    std::uint32_t const min_queue_size = get_agent_min_queue_size(gpu_agent);
    HSA_CHECK(hsa_queue_create(gpu_agent, min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                               0, 0, &queue));

    // create signal to wait for the dispatch
    hsa_signal_t dispatch_signal;
    HSA_CHECK(hsa_signal_create(1, 0, nullptr, &dispatch_signal));

    // get kernel
    kernel_info_t kernel = get_kernel(code_object.executable, kernel_name, gpu_agent);

    // allocate memory for arguments
    hsa_region_t kernarg_region = find_kernarg_region(gpu_agent);
    void* kernarg_address = nullptr;
    HSA_CHECK(hsa_memory_allocate(kernarg_region, kernel.kernarg_size, &kernarg_address));
    auto& args = *static_cast<args_t*>(kernarg_address);
    args.n = element_count;
    args.buffer = buffer;

    submit_packet(queue, dispatch_signal, kernel, kernarg_address,
                  element_count * sizeof(std::uint32_t));

    // wait for completion
    if (hsa_signal_wait_scacquire(dispatch_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                HSA_WAIT_STATE_BLOCKED) != 0) {
      std::cerr << "Error in waiting for completion\n";
      std::exit(EXIT_FAILURE);
    }

    // cleanup
    HSA_CHECK(hsa_signal_destroy(dispatch_signal));
    HSA_CHECK(hsa_memory_free(kernarg_address));
    HSA_CHECK(hsa_queue_destroy(queue));
  }
}

// Test modes
enum class test_mode_t { AllowAccess, DmaBuf, Vmem };

std::ostream& operator<<(std::ostream& os, test_mode_t const& test_mode) {
  switch (test_mode) {
    case test_mode_t::AllowAccess:
      return os << "AllowAccess";
    case test_mode_t::DmaBuf:
      return os << "DmaBuf";
    case test_mode_t::Vmem:
      return os << "Vmem";
    default:
      std::cerr << "Unknown test mode: "
                << static_cast<std::underlying_type_t<test_mode_t>>(test_mode) << '\n';
      std::exit(EXIT_FAILURE);
      return os;
  }
}

std::istream& operator>>(std::istream& in, test_mode_t& test_mode) {
  std::string token;
  in >> token;

  if (token == "AllowAccess") {
    test_mode = test_mode_t::AllowAccess;
  } else if (token == "DmaBuf") {
    test_mode = test_mode_t::DmaBuf;
  } else if (token == "Vmem") {
    test_mode = test_mode_t::Vmem;
  } else {
    std::cerr << "Unknown test mode: " << token << '\n';
    std::exit(EXIT_FAILURE);
  }

  return in;
}

// Allocation information
struct allocation_info_t {
  // Pool to allocate from
  hsa_amd_memory_pool_t pool = {};
  // Buffer number of elements
  std::size_t buffer_element_count = {};
  // Buffer size in bytes
  std::size_t buffer_byte_size = {};
  // Host accessible buffer
  std::uint32_t* buffer = nullptr;
  // Device accessible buffer
  std::uint32_t* device_ptr = nullptr;
  // File descriptor for dma-buf export / import
  int buffer_fd = -1;
  // VMem memory handle
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
};

int main() {
  HSA_CHECK(hsa_init());

  auto const test_mode = test_mode_t::AllowAccess;
  auto const gpu_kernel_file = "./add_one_kernel.hsaco";
  std::filesystem::path gpu_kernel_path(gpu_kernel_file);
  if (!std::filesystem::exists(gpu_kernel_path)) {
    std::cerr << "GPU kernel file not found: " << gpu_kernel_path << '\n';
    return EXIT_FAILURE;
  }

  // get CPU agent
  auto cpu_agent = find_first_cpu_agent();

  // get all GPU agents
  auto gpu_agents = find_gpu_agents();
  if (gpu_agents.empty()) {
    std::cerr << "Usable devices not found.\n";
    return EXIT_FAILURE;
  }
  auto gpu_agent_0 = gpu_agents[0];

  // load GPU code
  std::cout << "Loading GPU kernel from: " << gpu_kernel_path.string() << '\n';
  code_object_info_t code_object = code_object_load(gpu_kernel_path, gpu_agent_0);

  // allocate buffer
  allocation_info_t allocation_info;
  allocation_info.buffer_element_count = 8;
  allocation_info.pool = get_fine_grained_pool(gpu_agent_0);
  if (allocation_info.pool.handle == 0) {
    std::cerr << "No fine-grained pool found\n";
    return EXIT_FAILURE;
  }

  switch (test_mode) {
    case test_mode_t::AllowAccess:
    case test_mode_t::DmaBuf:
      // allowing any size
      allocation_info.buffer_byte_size =
          allocation_info.buffer_element_count * sizeof(std::uint32_t);
      break;
    case test_mode_t::Vmem: {
      // allocation has to be multiple of the granule size
      std::size_t const granule_size = get_pool_granule_size(allocation_info.pool);
      std::size_t const allocation_byte_size =
          allocation_info.buffer_element_count * sizeof(std::uint32_t);
      allocation_info.buffer_byte_size =
          ((allocation_byte_size + granule_size - 1) / granule_size) * granule_size;
    } break;
  }

  std::cout << "Allocating memory from: " << get_agent_name(gpu_agent_0) << '\n';
  switch (test_mode) {
    case test_mode_t::AllowAccess:
    case test_mode_t::DmaBuf: {
      // allocate memory in pool
      std::uint64_t const flags = 0;
      HSA_CHECK(hsa_amd_memory_pool_allocate(allocation_info.pool, allocation_info.buffer_byte_size,
                                             flags,
                                             reinterpret_cast<void**>(&allocation_info.buffer)));
    } break;
    case test_mode_t::Vmem: {
      {
        // allocate memory in pool
        std::uint64_t const flags = 0;
        HSA_CHECK(hsa_amd_vmem_handle_create(allocation_info.pool, allocation_info.buffer_byte_size,
                                             MEMORY_TYPE_PINNED, flags,
                                             &allocation_info.memory_handle));
      }

      {
        // reserve virtual address space
        std::uint64_t const address = 0;
        std::uint64_t const alignment = 0;
        std::uint64_t const flags = 0;
        HSA_CHECK(hsa_amd_vmem_address_reserve_align(
            reinterpret_cast<void**>(&allocation_info.buffer), allocation_info.buffer_byte_size,
            address, alignment, flags));
      }

      {
        // map memory to host
        std::uint64_t const offset = 0;
        std::uint64_t const flags = 0;
        HSA_CHECK(hsa_amd_vmem_map(allocation_info.buffer, allocation_info.buffer_byte_size, offset,
                                   allocation_info.memory_handle, flags));
      }

      {
        // allow host access
        hsa_amd_memory_access_desc_t memory_access_desc = {HSA_ACCESS_PERMISSION_RW, cpu_agent};

        HSA_CHECK(hsa_amd_vmem_set_access(allocation_info.buffer, allocation_info.buffer_byte_size,
                                          &memory_access_desc, 1));
      }
    } break;
  }
  std::vector<std::uint32_t> initial_values(allocation_info.buffer_element_count);

  // initialize data from host
  std::iota(initial_values.begin(), initial_values.end(), 0);
  std::copy(initial_values.begin(), initial_values.end(), allocation_info.buffer);

  // export memory
  switch (test_mode) {
    case test_mode_t::AllowAccess:
      std::cout << "Sharing via: hsa_amd_agents_allow_access\n";

      // give access to all agents
      HSA_CHECK(hsa_amd_agents_allow_access(gpu_agents.size(), gpu_agents.data(), nullptr,
                                            allocation_info.buffer));

      allocation_info.device_ptr = allocation_info.buffer;
      break;
    case test_mode_t::DmaBuf: {
      std::cout << "Sharing via: hsa_amd_portable_export_dmabuf / "
                   "hsa_amd_interop_map_buffer\n";

      // export from agent via dma-buf
      int buffer_fd = -1;
      std::uint64_t buffer_offset = 0;
      HSA_CHECK(hsa_amd_portable_export_dmabuf(
          allocation_info.buffer, allocation_info.buffer_byte_size, &buffer_fd, &buffer_offset));

      // import to all agents via dma-buf
      auto const num_agents = gpu_agents.size();
      auto const agents = gpu_agents.data();
      constexpr std::uint32_t flags = 0;
      std::uint32_t* buffer_import = nullptr;
      std::size_t buffer_import_size = 0;
      std::size_t* metadata_size_ptr = nullptr;
      void const** metadata_ptr = nullptr;
      HSA_CHECK(hsa_amd_interop_map_buffer(
          num_agents, agents, buffer_fd, flags, &buffer_import_size,
          reinterpret_cast<void**>(&buffer_import), metadata_size_ptr, metadata_ptr));

      allocation_info.buffer_fd = buffer_fd;
      allocation_info.device_ptr = buffer_import;
    } break;
    case test_mode_t::Vmem: {
      std::cout << "Sharing via: vmem API\n";

      std::vector<hsa_amd_memory_access_desc_t> memory_access_desc;
      memory_access_desc.reserve(gpu_agents.size() + 1);
      memory_access_desc.emplace_back(HSA_ACCESS_PERMISSION_RO, cpu_agent);
      for (auto const& agent : gpu_agents) {
        memory_access_desc.emplace_back(HSA_ACCESS_PERMISSION_RW, agent);
      }

      HSA_CHECK(hsa_amd_vmem_set_access(allocation_info.buffer, allocation_info.buffer_byte_size,
                                        memory_access_desc.data(), memory_access_desc.size()));

      allocation_info.device_ptr = allocation_info.buffer;

    } break;
    default:
      std::cerr << "Unknown test mode " << test_mode << '\n';
      return EXIT_FAILURE;
  }

  // run kernel
  run_add_one(code_object, gpu_agents, allocation_info.device_ptr,
              allocation_info.buffer_element_count);

  // check results
  auto const sum = gpu_agents.size();
  std::size_t errors = 0;
  for (std::size_t i = 0; i < allocation_info.buffer_element_count; i++) {
    auto const expected_value = initial_values[i] + sum;
    auto const result = allocation_info.buffer[i];
    if (result != expected_value) {
      ++errors;
      std::cout << "Error in output [" << i << "]: " << result << "!=" << expected_value << '\n';
    }
  }
  if (errors != 0) {
    std::cout << "FAILED!\n";
    return EXIT_FAILURE;
  }
  std::cout << "PASS!\n";

  // cleanup
  std::cout << "Cleaning up\n";
  switch (test_mode) {
    case test_mode_t::AllowAccess:
      // deallocate memory
      HSA_CHECK(hsa_amd_memory_pool_free(allocation_info.buffer));
      break;
    case test_mode_t::DmaBuf: {
      // unmap buffer
      HSA_CHECK(
          hsa_amd_interop_unmap_buffer(reinterpret_cast<void**>(&allocation_info.device_ptr)));
      // close file descriptor
      HSA_CHECK(hsa_amd_portable_close_dmabuf(allocation_info.buffer_fd));
      // deallocate memory
      HSA_CHECK(hsa_amd_memory_pool_free(allocation_info.buffer));
    } break;
    case test_mode_t::Vmem:
      // unmap buffer
      HSA_CHECK(hsa_amd_vmem_unmap(allocation_info.buffer, allocation_info.buffer_byte_size));
      // release virtual address
      HSA_CHECK(
          hsa_amd_vmem_address_free(allocation_info.buffer, allocation_info.buffer_byte_size));
      // release memory handle
      HSA_CHECK(hsa_amd_vmem_handle_release(allocation_info.memory_handle));
      break;
  }
  code_object_free(code_object);
  HSA_CHECK(hsa_shut_down());

  return EXIT_SUCCESS;
}
