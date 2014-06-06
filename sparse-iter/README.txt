#==============================================================================#
#                       README FILE FOR MAGMA-SPARSE                           #
#==============================================================================#


==============
 INSTALLATION 
==============
1) unzip the tar.gz file into some folder

2) "cd magma-x.y.z-lll/"

3) set a symbolic link via 
        "ln -s make.inc.xxx-yyy make.inc" 
   to the most suitable configuration. You may still have to adapt some paths 
   inside.

3) compile magma using "make -j8" (in case you want to use 8 cores...)

4) "cd sparse-iter"

5) compile magma-sparse using "make -j8" (in case you want to use 8 cores...)

6) If no errors show up, everything has worked out fine. 
   In the sparse-iter/testing folder you find example runfiles - for proper
   usage see section below.


==============
 USAGE     
==============

In the sparse-iter/testing folder are simple programs to run the different 
solvers. They all need a matrix in MatrixMarket format (.mtx) as input. 
Test matrices of this type can be downloaded at the University of Florida Matrix 
Collection: http://www.cise.ufl.edu/research/sparse/matrices/ 

A suitable test matrix is given by Trefethen_2000:
http://www.cise.ufl.edu/research/sparse/matrices/JGD_Trefethen/Trefethen_2000.html

To run a example type


            ./run_xsolver [ --options ] matrices


Every solver (replace "solver" by cg, gmres, bicgstab, iterref, jacobi, bcsrlu)
exists in single ("x"=s), double ("x"=d), single-complex and double-complex
version ("x"=c or z, respectively).


For different solvers there exist different options, which are printed 
when executing "./run_xsolver".

 Some options are:

"--verbose k"
    k = 0 : solver is run in production mode, no additional characteristics 
            monitored
    k > 0 : solver is run in verbose mode, residual and runtime is monitored
            every k iterations

"--format k"
    k = 0 : CSR
    k = 1 : ELLPACK
    k = 2 : ELLPACKT
    k = 3 : ELLPACKRT

"--blocksize k"
    for Magma_ELLPACKRT: denotes the number of rows in one slice of the matrix
                      and the number of rows assigned to one multiprocessor
    for Magma_SELLC: k denotes the number of rows in one slice of the matrix
                      and the number of rows assigned to one multiprocessor
    for Magma_SELLCM: k denotes the number of rows in one slice of the matrix
                      and the number of rows assigned to one multiprocessor

"--alignment k"
    for Magma_ELLPACKRT: k denotes the number of threads assigned to one row
                      notice: blocksize * alignment needs to fit into shared mem
    for Magma_SELLCM: k denotes the number of threads assigned to one row
                      notice: blocksize * alignment needs to fit into shared mem

"--ortho k"
    k = 0 : orthogonalization via classical Gram-Schmidt (CGS)
    k = 1 : orthogonalization via modified Gram-Schmidt (MGS)
    k = 2 : orthogonalization via fused classical Gram-Schmidt (CGS_FUSED)

"--maxiter k"
    k : upper bound for iterations

"--tol k"
    k : relative residual stopping criterion (machine precision as default)

"--restart k"
    k : Krylov subspace restart number for GMRES-(k)

"--version k"
    k : for some solvers (CG, BiCGStab) there exist accelerated versions merging
        multiple numerical operations into one kernel operation. These can be
        selected via the version number. The default implementations are the
        classical ones. 
        For BCSRLU, version allows to select either CUBLAS-batched GEMM or
        a custom-designed kernel suitable for block-size 64.

"--preconditioner k"
    k = 0 : Jacobi
    k = 1 : CG
    k = 2 : BiCGStab
    k = 3 : GMRES

"--precond-maxiter/tol/restart" similar like above. The preconditioner types for
    the iterative refinement are set to the default classical implementations. 
    If the merged variants are preferred, they have to be called explicitly in 
    the runfile.

The last argument is the traget matrices. These should be stored in MatrixMarket
format, see http://math.nist.gov/MatrixMarket/formats.html.
