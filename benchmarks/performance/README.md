# Various performance benchmark apps.

## NCU Profiling notes

To profile with NVIDIA's `ncu` CLI tool you can run, e.g.,

```
ncu --kernel-name-base demangled --kernel-name regex:.*Laplace.* --set full -o some_outfile.out -f ./benchmark_operators
```

where `--kernel-name` gets a regex that must match the demangled Kokkos kernel name. Requires a bit of trial and error.
For instance the demangled name of the a Laplace operator (functor - so a class with operator() overload) is

```
void Kokkos::Impl::cuda_parallel_launch_constant_memory<Kokkos::Impl::ParallelFor<terra::fe::wedge::operators::shell::LaplaceSimple<double>, Kokkos::MDRangePolicy<Kokkos::Rank<(unsigned int)4, (Kokkos::Iterate)0, (Kokkos::Iterate)0>>, Kokkos::Cuda>>()
```

You can then e.g. import the written file with the NVIDIA NSight Compute app.