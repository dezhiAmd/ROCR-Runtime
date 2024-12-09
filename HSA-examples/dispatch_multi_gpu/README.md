# Multi GPU dispatch

This test stresses buffer sharing across multiple GPU agents in HSA.

This tests creates a buffer in the host, and initializes it with 0s. Then it iterates over all the GPU agents. Each agent will add 1 to the buffer. At the end we check that the numbers in the buffers all match the number of GPU agents.

There are two modes in this test. One is for exercising `hsa_amd_agents_allow_access`, and the other for DMA Buffer Export/Import operations.


## Run

Run in your build directory:
```bash
./add_one_multigpu -g add-one.hsaco -m 0 # Default, hsa_amd_agents_allow_access mode
./add_one_multigpu -g add-one.hsaco -m 1 # dma export/import mode
```
