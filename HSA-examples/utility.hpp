/* Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved. */

#ifndef ROCR_EXAMPLES_UTILITY_HPP
#define ROCR_EXAMPLES_UTILITY_HPP

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "hsa/hsa.h"

// Checks if x is not HSA_STATUS_SUCCESS, prints an error message and aborts
// execution
#define HSA_CHECK(x)                                                           \
  do {                                                                         \
    hsa_status_t const check_status = (x);                                     \
    if (check_status != HSA_STATUS_SUCCESS) {                                  \
      const char *msg = NULL;                                                  \
      auto const string_status = hsa_status_string(check_status, &msg);        \
      if (string_status == HSA_STATUS_SUCCESS) {                               \
        std::printf("%s:%i: %s\n", __func__, __LINE__, msg);                   \
      } else {                                                                 \
        std::printf("%s:%i: HSA error but failed to find error message\n",     \
                    __func__, __LINE__);                                       \
      }                                                                        \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

// Returns the agent name as a string
inline std::string get_agent_name(hsa_agent_t agent) {
  char agent_name[64] = {};
  HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, &agent_name));
  return std::string(agent_name);
}

// Returns the minimum queue size
inline std::uint32_t get_agent_min_queue_size(hsa_agent_t agent) {
  std::uint32_t min_queue_size = 0;
  HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                               &min_queue_size));
  return min_queue_size;
}

#endif // ROCR_EXAMPLES_UTILITY_HPP
