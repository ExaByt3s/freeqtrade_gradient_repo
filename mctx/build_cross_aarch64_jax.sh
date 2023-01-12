# git clone https://github.com/google/jax
cd jax/
# sudo apt install g++ python python3-dev
# pip install numpy wheel
# https://github.com/google/jax/issues/7097#issuecomment-1216826398
# sudo apt install crossbuild-essential-arm64
# mkdir toolchain/
# GCC_MAJOR_VERSION='9'
    # cxx_builtin_include_directories = [
        # "/usr/aarch64-linux-gnu/include",
        # "/usr/lib/gcc-cross/aarch64-linux-gnu/11/include",
        # "/usr/local/include",
        # "/usr/include",
        # "/usr/include/c++/11",
        # "/usr/include/c++/11/backward",
    # ],
echo '
load("@local_config_cc//:cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "k8|compiler": "@local_config_cc//:cc-compiler-k8",
        "k8": "@local_config_cc//:cc-compiler-k8",
        "aarch64": ":cc-compiler-aarch64",
    },
)

cc_toolchain(
    name = "cc-compiler-aarch64",
    all_files = "@local_config_cc//:compiler_deps",
    ar_files = "@local_config_cc//:compiler_deps",
    as_files = "@local_config_cc//:compiler_deps",
    compiler_files = "@local_config_cc//:compiler_deps",
    dwp_files = ":empty",
    linker_files = "@local_config_cc//:compiler_deps",
    module_map = None,
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":cross_aarch64",
    toolchain_identifier = "cross_aarch64",
)

cc_toolchain_config(
    name = "cross_aarch64",
    abi_libc_version = "local",
    abi_version = "local",
    compile_flags = [
        "-U_FORTIFY_SOURCE",
        "-fstack-protector",
        "-Wall",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer",
        "-I/usr",
        "-I/usr/include/python3.8",        
    ],
    compiler = "compiler",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "aarch64",
    cxx_builtin_include_directories = [
        "/usr",
        "/usr/aarch64-linux-gnu/include",
        "/usr/lib/gcc-cross/aarch64-linux-gnu/9/include",
        "/usr/local/include",
        "/usr/include",
        "/usr/include/c++/9",
        "/usr/include/c++/9/backward",
    ],
    cxx_flags = ["-std=c++0x"],
    dbg_compile_flags = ["-g"],
    host_system_name = "local",
    link_flags = [
        "-fuse-ld=gold",
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
        "-B/usr/bin/aarch64-linux-gnu-",
        "-pass-exit-codes",
    ],
    link_libs = [
        "-lstdc++",
        "-lm",
    ],
    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    supports_start_end_lib = True,
    target_libc = "local",
    target_system_name = "local",
    tool_paths = {
        "ar": "/usr/bin/ar",
        "ld": "/usr/bin/aarch64-linux-gnu-ld",
        "llvm-cov": "/usr/bin/llvm-cov",
        "cpp": "/usr/bin/aarch64-linux-gnu-cpp",
        "gcc": "/usr/bin/aarch64-linux-gnu-gcc",
        "dwp": "/usr/bin/aarch64-linux-gnu-dwp",
        "gcov": "/usr/bin/aarch64-linux-gnu-gcov",
        "nm": "/usr/bin/aarch64-linux-gnu-nm",
        "objcopy": "/usr/bin/aarch64-linux-gnu-objcopy",
        "objdump": "/usr/bin/aarch64-linux-gnu-objdump",
        "strip": "/usr/bin/aarch64-linux-gnu-strip",
    },
    toolchain_identifier = "cross_aarch64",
    unfiltered_compile_flags = [
        "-fno-canonical-system-headers",
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
    ],
)' > toolchain/BUILD
python build/build.py  --bazel_option=--crosstool_top=//toolchain:toolchain --target_cpu=aarch64
# pip install dist/*.whl
# ln -sfvn /usr/include/python3.8 /usr/aarch64-linux-gnu/include/python3.8
# ln -sfvn /usr/aarch64-linux-gnu/include/python3.8 /usr/aarch64-linux-gnu/python3.8
# python3.8-config --includes
