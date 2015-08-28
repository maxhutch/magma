#!/bin/tcsh
#
# Prints vital stats about environment.
#
# @author Mark Gates

echo "============================== environment"
echo "USER=$USER"

echo "HOST="`hostname`

if ( $?OMP_NUM_THREADS ) then
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
else
    echo "OMP_NUM_THREADS undefined"
endif

if ( $?MKL_NUM_THREADS ) then
    echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
else
    echo "MKL_NUM_THREADS undefined"
endif

if ( $?MKLROOT ) then
    echo "MKLROOT=$MKLROOT"
else
    echo "MKLROOT undefined"
endif

if ( $?CUDADIR ) then
    echo "CUDADIR=$CUDADIR"
else
    echo "CUDADIR undefined"
endif

if ( $?GPU_TARGET ) then
    echo "GPU_TARGET=$GPU_TARGET"
else
    echo "GPU_TARGET undefined"
endif

echo
echo
echo "============================== magma"
pwd

echo
if ( -d .svn || -d ../.svn ) then
    echo "svn info"
    svn info
else
    echo "no svn info"
endif

echo
ls -l ../make.inc

echo
echo
echo "============================== compilers"
which icc
icc --version

echo
which gcc
gcc --version

echo
which gfortran
gfortran --version

echo
which nvcc
nvcc --version

echo
echo
echo "============================== processors"
grep -m 4 'model name|MHz|cache size|siblings' /proc/cpuinfo

echo
cpuid -1 | grep '\(synth\)'

echo
nvidia-smi
