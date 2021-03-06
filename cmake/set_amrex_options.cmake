#Set amrex options
set(USE_XSDK_DEFAULTS OFF)
set(DIM 3)
set(ENABLE_ACC OFF)
set(ENABLE_AMRDATA OFF)
set(ENABLE_ASSERTIONS OFF)
set(ENABLE_BACKTRACE OFF)
set(ENABLE_BASE_PROFILE OFF)
set(ENABLE_COMM_PROFILE OFF)
set(ENABLE_CONDUIT OFF)
set(ENABLE_CUDA ${AMR_WIND_ENABLE_CUDA})
set(ENABLE_DP ON)
set(ENABLE_DPCPP ${AMR_WIND_ENABLE_DPCPP})
set(ENABLE_EB OFF)
set(ENABLE_FORTRAN ${AMR_WIND_ENABLE_FORTRAN})
set(ENABLE_FORTRAN_INTERFACES ${AMR_WIND_ENABLE_FORTRAN})
set(ENABLE_FPE OFF)
set(ENABLE_HYPRE ${AMR_WIND_ENABLE_HYPRE})
set(ENABLE_LINEAR_SOLVERS ON)
set(ENABLE_MEM_PROFILE OFF)
set(ENABLE_MPI ${AMR_WIND_ENABLE_MPI})
set(ENABLE_OMP ${AMR_WIND_ENABLE_OPENMP})
set(ENABLE_PARTICLES ON)
set(ENABLE_PIC ON)
set(ENABLE_PLOTFILE_TOOLS ${AMR_WIND_ENABLE_FCOMPARE})
set(ENABLE_PROFPARSER OFF)
set(ENABLE_SENSEI_INSITU OFF)
set(ENABLE_SUNDIALS OFF)
set(ENABLE_TINY_PROFILE OFF)
set(ENABLE_TRACE_PROFILE OFF)
