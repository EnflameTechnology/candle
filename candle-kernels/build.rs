use cudaforge::KernelBuilder;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/compatibility.cuh");
    println!("cargo:rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/binary_op_macros.cuh");

    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .build_ptx()
        .expect("Failed to compile CUDA kernels");

    bindings
        .write("src/lib.rs")
        .expect("Failed to write PTX bindings");
}
