use std::ffi::{c_void, CString};
use std::ptr;

// Import the bindings
use rust_gpu_cpp::root::{
    gpu_createContext, gpu_createTensor, gpu_createKernel, gpu_dispatchKernel,
    gpu_toCPU, gpu_cdiv, gpu_createShape, gpu_toGPU_float,
    gpu::NumType_kf32, gpu::KernelCode, gpu::Shape,
};

const K_GELU: &str = r#"
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dummy: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
                 * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
"#;

fn main() {
    println!("\x1B[2J\x1B[1;1H");
    println!("\nHello gpu.cpp!");
    println!("--------------\n");

    unsafe {
        let ctx = gpu_createContext();
        
        const N: usize = 10000;
        let mut input_arr = [0.0f32; N];
        let mut output_arr = [0.0f32; N];
        
        for i in 0..N {
            input_arr[i] = i as f32 / 10.0; // dummy input data
        }

        let shape = gpu_createShape(&N as *const usize, 1);
        let input = gpu_createTensor(ctx, shape, NumType_kf32 as i32);
        let output = gpu_createTensor(ctx, shape, NumType_kf32 as i32);

        gpu_toGPU_float(ctx, input_arr.as_ptr(), input);

        let code = KernelCode {
            _bindgen_opaque_blob: [
                K_GELU.as_ptr() as u64,
                K_GELU.len() as u64,
                256,
                NumType_kf32 as u64,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        };

        let bindings = [input, output];
        let n_workgroups_data = [gpu_cdiv(N, 256), 1, 1];
        let n_workgroups = gpu_createShape(n_workgroups_data.as_ptr(), 3);

        let op = gpu_createKernel(
            ctx,
            &code as *const _ as *const c_void,
            bindings.as_ptr() as *const c_void,
            2,
            ptr::null(),
            n_workgroups,
            ptr::null(),
            0,
        );

        let promise = Box::into_raw(Box::new(std::sync::Mutex::new(false))) as *mut c_void;
        gpu_dispatchKernel(ctx, op, promise);

        // Wait for the kernel to complete (this is a simplification)
        while !*(*(promise as *mut std::sync::Mutex<bool>)).lock().unwrap() {
            std::thread::yield_now();
        }

        gpu_toCPU(ctx, output, output_arr.as_mut_ptr(), std::mem::size_of::<[f32; N]>());

        for i in 0..12 {
            println!("  gelu({:.2}) = {:.2}", input_arr[i], output_arr[i]);
        }
        println!("  ...\n");
        println!("Computed {} values of GELU(x)\n", N);

        // Clean up (you should implement proper cleanup functions)
        // Free ctx, input, output, op, shape, n_workgroups, etc.
    }
}