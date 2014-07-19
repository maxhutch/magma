MacOS veclib has a bug where some single precision functions return
a double precision result, for instance slange.
This is observed with -m64, but oddly not with -m32.
The easiest fix is to replace those routines with correct ones from LAPACK (3.5.0).
See BLAS_FIX in make.inc.macos

Note that these are Level 1 and Level 2 BLAS and BLAS-like functions,
primarily norms and dot products, which are not generally performance critical.
