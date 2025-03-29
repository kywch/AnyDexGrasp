## Installling Minkowski Engine with CUDA 12.

* After `pixi shell`, run `pixi run build_minko`. If the build is successful, test it with `pixi run test_minko`. You should see:

```
==========System==========
Linux-6.9.3-76060903-generic-x86_64-with-glibc2.35
DISTRIB_ID=Pop
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Pop!_OS 22.04 LTS"
3.10.16 | packaged by conda-forge | (main, Dec  5 2024, 14:16:10) [GCC 13.3.0]
==========Pytorch==========
2.3.1
torch.cuda.is_available(): True
==========NVIDIA-SMI==========
/usr/bin/nvidia-smi
Driver Version 560.35.03
CUDA Version 12.6
VBIOS Version 94.02.59.00.D6
Image Version G001.0000.03.03
GSP Firmware Version 560.35.03
==========NVCC==========
/workspace/realdex/AnyDexGrasp/.pixi/envs/default/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
==========CC==========
CC=/workspace/realdex/AnyDexGrasp/.pixi/envs/default/bin/x86_64-conda-linux-gnu-c++
/workspace/realdex/AnyDexGrasp/.pixi/envs/default/bin/x86_64-conda-linux-gnu-c++
x86_64-conda-linux-gnu-c++ (conda-forge gcc 12.4.0-2) 12.4.0
Copyright (C) 2022 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

==========MinkowskiEngine==========
0.5.4
MinkowskiEngine compiled with CUDA Support: True
NVCC version MinkowskiEngine is compiled: 12010
CUDART version MinkowskiEngine is compiled: 12010
```

* The official NVIDIA Minkowski Engine repo does not support CUDA 12. So I used this repo, which is added here as a submodule: 
https://github.com/EthenJ/MinkowskiEngine
See the issue: https://github.com/NVIDIA/MinkowskiEngine/issues/601

* I also encountered the below error during compile, which was mentioned in https://github.com/NVIDIA/MinkowskiEngine/issues/596

    /workspace/realdex/AnyDexGrasp/.pixi/envs/default/lib/gcc/x86_64-conda-linux-gnu/12.4.0/include/c++/bits/shared_ptr_base.h(1561): error: more than one instance of overloaded function "std::__to_address" matches the argument list:
                function template "_Tp *cuda::std::__4::__to_address(_Tp *) noexcept" (declared at line 277 of /workspace/realdex/AnyDexGrasp/.pixi/envs/default/include/cuda/std/detail/libcxx/include/__memory/pointer_traits.h)
                function template "_Tp *std::__to_address(_Tp *) noexcept" (declared at line 209 of /workspace/realdex/AnyDexGrasp/.pixi/envs/default/lib/gcc/x86_64-conda-linux-gnu/12.4.0/include/c++/bits/ptr_traits.h)
                argument types are: (concurrent_unordered_map<minkowski::coordinate<int32_t>, uint32_t, minkowski::detail::coordinate_murmur3<int32_t>, minkowski::detail::coordinate_equal_to<int32_t>, minkowski::detail::default_allocator<cuda::std::__4::pair<minkowski::coordinate<int32_t>, uint32_t>>> *)
         auto __raw = __to_address(__r.get());

To fix this, in the `.pixi/envs/default/lib/gcc/x86_64-conda-linux-gnu/12.4.0/include/c++/bits/shared_ptr_base.h` file, line 1561:
change `auto __raw = __to_address(__r.get());` to `auto __raw = std::__to_address(__r.get());`
