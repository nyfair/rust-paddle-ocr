fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::copy("mnn_bindings.rs", format!("{}/mnn_bindings.rs", out_dir)).expect("no mnn_bindings.rs");
    cc::Build::new()
        .cpp(true)
        .file("cpp/src/mnn_wrapper.cpp")
        .include("include")
        .include("cpp/include")
        .flag("/std:c++14")
        .flag("/EHsc")
        .compile("mnn_wrapper");
    println!("cargo:rustc-link-search=native=lib");
    println!("cargo:rustc-link-lib=static=MNN");
}
