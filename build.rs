use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Build the C++ library
    let cpp_lib_path = PathBuf::from("extern/gpu.cpp");
    
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

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={}", cpp_lib_path.display());

    // Tell cargo to tell rustc to link the library
    println!("cargo:rustc-link-lib=libgpud");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .clang_arg("-x")
        .clang_arg("c++")
        //.clang_arg("-std c++17")
        .clang_arg("-include array")
        .clang_arg("-I./extern/gpu.cpp")
        //.clang_arg("-stdlib=libc++")
        //.enable_cxx_namespaces()
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}