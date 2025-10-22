**TerraNeoX** is a mantle convection code based on [Kokkos](https://github.com/kokkos/kokkos) for performance portability.

\if NOT_DOXYGEN_GENERATED

> It seems you are viewing this in plain text or with a standard Markdown renderer (e.g., on GitHub).
> 
> ❗️This file is best read in HTML format after generating the documentation via running `doxygen` inside of `doc/src/`.

\endif

> Looking for the [general framework documentation](framework/framework.md)? 

# Features

🏗️

\docseplarge

# Building

## Dependencies

Mandatory:

* MPI (e.g. OpenMPI)

Optional:

* CUDA (for GPU support)

## Compiling on the LMU systems (`cachemiss`, `memoryleak`) for usage with CUDA:

```
$ module load mpi.ompi
$ module load nvidia-hpc

$ mkdir terraneox-build

$ ll
terraneox/               # <== the cloned source code
terraneox-build/

$ cd terraneox-build

$ cmake ../terraneox/ -Kokkos_ENABLE_CUDA=ON

# Build tests
$ cd tests
$ make -j16
```

Note the capitalization: it must be `Kokkos_ENABLE_CUDA=ON`, NOT `KOKKOS_ENABLE_CUDA=ON`.

\docseplarge

## Compiling on the NHR systems, e.g. helma (NVIDIA H100 GPUs):

Access form: https://hpc.fau.de/access-to-helma/

SSH Connect and other info on helma: https://doc.nhr.fau.de/clusters/helma/

```
$ module load openmpi/5.0.5-nvhpc24.11-cuda
$ module load cmake
$ mkdir terra-kokkos-build
$ cd terra-kokkos-build

# give parallel backend and architecture via cmake (Kokkos may be unable to autodetect the arch)
$ cmake ../terra-kokkos -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON

# Build tests
$ cd tests
$ make -j 16
```

\docseplarge

# Project Structure

```
terraneox/
├── apps/                     # Applications (benchmarks, tools, ...) using the framework.
│   ├── benchmarks/           # Benchmarks ...
│   │   ├── performance/      # ... run time
│   │   └── physics/          # ... physics
│   └── tools/                # Tools (e.g., for visualization of meshes)
├── data/                     # Stuff that is not exactly library/framework source code
│   └── scripts/              # Scripts, e.g., for post-processing
│       └── plotting/         # Scripts for visualization, e.g., timing, radial profiles, ... 
├── doc/                      # Documentation
│   ├── documents/            # Documents that are not part of the generated documentation (e.g., Latex documents)
│   └── src/                  # Doxygen documentation
│       └── documentation/    # Doxygen documentation source (this page is likely here)
├── extern/                   # External libraries shipped with this repository
├── src/                      # Source code
└── tests/                    # Tests
```

\docseplarge

# Todo Items

## Big features (definitely required - order not clear)

- [x] advection-diffusion discretization / solver
- [x] advection-diffusion boundary handling
- [ ] advection-diffusion source term handling (must add SUPG term in linear form)
- [x] ~~GMRES~~ BiCGStab(l)
- [x] BDF2 (not yet in a dedicated function, see test_heat_eq)
- [x] multigrid (some notes: a) we need higher operator quad degree than constant (not sure where exactly: diagonal,
  fine-level, everywhere?), b) two-grid V(10, 10) looks ok, otherwise with multigrid we do not get perfectly h-ind. conv
  rates., I
  suppose we need Galerkin coarse grid operators maybe)
- [x] MPI parallel execution (multi-GPU, multi-node CPU)
- [x] intra-diamond subdomain communication (then also test/fix boundary handling in operators/tests - subdomain
  boundaries are sometimes treated as domain boundaries even if they are not)
- [x] variable viscosity
- [x] full stokes
- [ ] plates
- [ ] free-slip
- [ ] compressible Stokes
- [x] FGMRES (BiCGStab works well mostly - but seems to randomly produce NaNs occasionally (not 100% sure if related to
  the solver, but it is very likely))
- [ ] Galerkin coarsening
- [ ] iterative refinement
- [ ] spherical harmonics helper
- [ ] radial profiles loader
- [ ] checkpoints (re-use XDMF bin files!)
- [ ] return unmanaged views from SubdomainNeighborhoodSendRecvBuffer that point to contiguous memory per rank, add
  another getter to the pointer of that array and then pass that to MPI_Send/Recv instead
- [x] radial layer data assimilation
- [x] timing(tree?)
- [x] ~~compress VTK(?)~~ XDMF output (binary, actual float data, and ~~/or~~ ~~HDF5/ADIOS2~~ with a single mesh file)
- [x] CLI interface / parameter files

## Small features / improvements (not necessarily / maybe required)

- [ ] cube-like test case (this may require some new FE features)
- [ ] performance engineering
- [x] ~~curved wedges~~ the wedges are curved (unlike I assumed when writing this)
- [ ] particles(?)
- [x] matrix export / assembly (implemented for debugging - not for actual use)
- [ ] CPU SIMD kernels
- [ ] adapt solver ctor like in FGMRES (I think that is the best design)

## Documentation / cleanup / refactoring

- [ ] Github page
- [x] Doxygen
- [ ] Doxygen page
- [x] move mask stuff that generalizes away from shell namespace
- [ ] sort out what is spherical shell specific and what is not