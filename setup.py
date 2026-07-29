from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    packages=["minidist"],
    ext_modules=[
        CUDAExtension(
            name="minidist._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/process_group_nccl.cpp",
                "csrc/tensor_utils.cpp",
                "csrc/work_nccl.cpp",
            ],
            include_dirs=["csrc"],
            libraries=["nccl"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)