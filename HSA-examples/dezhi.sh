g++ -std=c++20 -g -O0 dispatch_multi_gpu.cpp -I/home/dezhliao/develop/ROCR-debug/include -L/home/dezhliao/develop/ROCR-debug/lib/ -lhsa-runtime64 -lhsakmt -lnuma -lstdc++ -lm -lelf -ldrm -ldrm_amdgpu -o test