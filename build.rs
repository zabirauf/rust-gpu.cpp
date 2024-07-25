use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Build the C++ library
    let cpp_lib_path = PathBuf::from("extern/gpu.cpp");
    let wrapper_dir = env::current_dir().unwrap().join("cpp_wrapper");
    
    // Create build directory
    std::fs::create_dir_all(cpp_lib_path.join("build")).unwrap();

    // Run CMake
    let status = Command::new("cmake")
        .current_dir(cpp_lib_path.clone())
        .arg(".")
        .status()
        .expect("Failed to execute CMake");
    assert!(status.success());

    // Run Make
    let status = Command::new("make")
        .current_dir(cpp_lib_path.clone())
        .status()
        .expect("Failed to execute Make");
    assert!(status.success());

    // Compile the wrapper
    let output = Command::new("clang++")
        .args(&[
            "-c",
            "-std=c++17",
            "-I", cpp_lib_path.to_str().unwrap(),
            "-I", cpp_lib_path.join("third_party/headers").to_str().unwrap(),
            "-I", wrapper_dir.to_str().unwrap(),
            "-o", "gpu_wrapper.o",
            wrapper_dir.join("gpu_wrapper.cpp").to_str().unwrap()
        ])
        .output()
        .expect("Failed to compile gpu_wrapper.cpp");

    if !output.status.success() {
        panic!("Failed to compile gpu_wrapper.cpp: {}", String::from_utf8_lossy(&output.stderr));
    }

    // Create a static library
    Command::new("ar")
        .args(&["crus", "libgpu_wrapper.a", "gpu_wrapper.o"])
        .output()
        .expect("Failed to create static library");

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=native={}", cpp_lib_path.display());
    println!("cargo:rustc-link-search=native={}", cpp_lib_path.join("third_party/lib").display());
    println!("cargo:rustc-link-search=./");

    println!("cargo:rustc-link-lib=static=gpu_wrapper");
    println!("cargo:rustc-link-lib=dylib=dawn");
    println!("cargo:rustc-link-lib=dylib=dl");

    // Tell cargo to tell rustc to link the library
    println!("cargo:rustc-link-lib=dylib=gpud");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=cpp_wrapper/gpu_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp_wrapper/gpu_wrapper.h");

    let mut builder = bindgen::Builder::default()
    .header("wrapper.h")
    .clang_arg(format!("-I{}", cpp_lib_path.display()))
    .clang_arg(format!("-I{}", cpp_lib_path.join("third_party/headers").display()))
    .clang_arg(format!("-I{}", wrapper_dir.display()))
    .clang_arg("-std=c++17")
    .clang_arg("-x")
    .clang_arg("c++")
    // Disable generation of layout tests
    .layout_tests(false)
    // Block problematic types
    .blocklist_type("std::.*")
    .blocklist_type("__gnu_cxx::.*")
    // Allow list only the types and functions you need
    .allowlist_type("gpu::.*")
    .allowlist_function("gpu::.*")
    .allowlist_function("gpu_.*")
    // Treat all types as opaque to avoid conflicts
    .opaque_type(".*")
    // Generate smart pointers for C++ classes
    .enable_cxx_namespaces()
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    // Check if we need to add -stdlib=libc++
    let output = std::process::Command::new("clang++")
        .args(&["-std=c++17", "-x", "c++", "-E", "-include", "array", "-"])
        .stdin(std::process::Stdio::null())
        .output()
        .expect("Failed to execute clang++");

    if !output.status.success() {
        builder = builder.clang_arg("-stdlib=libc++");
    }

    let bindings = builder
        .generate()
        .expect("Unable to generate bindings");


    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}