#ifndef GPU_WRAPPER_H
#define GPU_WRAPPER_H

#include "gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// Context creation
void* gpu_createContext();

// Tensor operations
void* gpu_createTensor(void* ctx, const void* shape, int type);
void gpu_toGPU_float(void* ctx, const float* data, void* tensor);
void gpu_toGPU_half(void* ctx, const void* data, void* tensor);
void gpu_toCPU(void* ctx, void* tensor, float* data, size_t size);

// Kernel operations
void* gpu_createKernel(void* ctx, const void* code, const void* dataBindings, 
                       size_t numTensors, const size_t* viewOffsets, 
                       const void* nWorkgroups, const void* params, size_t paramsSize);
void gpu_dispatchKernel(void* ctx, void* op, void* promise);

// Shape operations
void* gpu_createShape(const size_t* dims, size_t rank);
size_t gpu_getShapeElement(const void* shape, size_t index);
void gpu_setShapeElement(void* shape, size_t index, size_t value);

// Other utility functions
size_t gpu_cdiv(size_t a, size_t b);

#ifdef __cplusplus
}
#endif

#endif // GPU_WRAPPER_H