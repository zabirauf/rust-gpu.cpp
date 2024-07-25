#include "gpu_wrapper.h"
#include <vector>

extern "C" {

void* gpu_createContext() {
    return new gpu::Context(gpu::createContext());
}

void* gpu_createTensor(void* ctx, const void* shape, int type) {
    return new gpu::Tensor(gpu::createTensor(
        *static_cast<gpu::Context*>(ctx),
        *static_cast<const gpu::Shape*>(shape),
        static_cast<gpu::NumType>(type)
    ));
}

void gpu_toGPU_float(void* ctx, const float* data, void* tensor) {
    gpu::toGPU(*static_cast<gpu::Context*>(ctx), data, *static_cast<gpu::Tensor*>(tensor));
}

void gpu_toGPU_half(void* ctx, const void* data, void* tensor) {
    gpu::toGPU(*static_cast<gpu::Context*>(ctx), static_cast<const half*>(data), *static_cast<gpu::Tensor*>(tensor));
}

void gpu_toCPU(void* ctx, void* tensor, float* data, size_t size) {
    gpu::toCPU(*static_cast<gpu::Context*>(ctx), *static_cast<gpu::Tensor*>(tensor), data, size);
}

void* gpu_createKernel(void* ctx, const void* code, const void* dataBindings, 
                       size_t numTensors, const size_t* viewOffsets, 
                       const void* nWorkgroups, const void* params, size_t paramsSize) {
    return new gpu::Kernel(gpu::createKernel(
        *static_cast<gpu::Context*>(ctx),
        *static_cast<const gpu::KernelCode*>(code),
        static_cast<const gpu::Tensor*>(dataBindings),
        numTensors,
        viewOffsets,
        *static_cast<const gpu::Shape*>(nWorkgroups),
        params,
        paramsSize
    ));
}


void gpu_dispatchKernel(void* ctx, void* op, void* promise) {
    gpu::dispatchKernel(
        *static_cast<gpu::Context*>(ctx),
        *static_cast<gpu::Kernel*>(op),
        *static_cast<std::promise<void>*>(promise)
    );
}

void* gpu_createShape(const size_t* dims, size_t rank) {
    gpu::Shape* shape = new gpu::Shape();
    shape->rank = rank;
    for (size_t i = 0; i < rank && i < gpu::Shape::kMaxRank; ++i) {
        shape->data[i] = dims[i];
    }
    return shape;
}

size_t gpu_getShapeElement(const void* shape, size_t index) {
    return (*static_cast<const gpu::Shape*>(shape))[index];
}

void gpu_setShapeElement(void* shape, size_t index, size_t value) {
    (*static_cast<gpu::Shape*>(shape))[index] = value;
}

size_t gpu_cdiv(size_t a, size_t b) {
    return gpu::cdiv(a, b);
}

}