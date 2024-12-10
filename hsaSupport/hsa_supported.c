#include <hsa/hsa.h>
#include <memory.h>
#include <stdio.h>

int main(int argc, char** argv) {
  hsa_status_t err;

  err = hsa_init();
  if (err != HSA_STATUS_SUCCESS) {
    return -1;
  }

  uint16_t version_major = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &version_major);
  uint16_t version_minor = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &version_minor);
  fprintf(stderr, "HSA version %u.%u\n", version_major, version_minor);

  bool svm_by_default = false;
  hsa_system_get_info(HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT,
                      &svm_by_default);
  fprintf(stderr, "HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT=%d\n",
          svm_by_default ? 1 : 0);

  bool virtual_mem_api_supported = false;
  hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED,
                      &virtual_mem_api_supported);
  fprintf(stderr, "Dezhi, HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED=%d\n",
          virtual_mem_api_supported ? 1 : 0);

  hsa_shut_down();

  return virtual_mem_api_supported ? 1 : 0;
}
