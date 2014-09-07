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

To run a solver:


            ./run_xsolver [ --options ] matrices


Every solver exists in single ("x"=s), double ("x"=d), single-complex and 
double-complex version ("x"=c or z, respectively).


For different solvers there exist different options, which are printed 
when executing "./run_xsolver -h".

 Some options are:

 --solver      Possibility to choose a solver
               0   CG
               1   merged CG
               2   preconditioned CG
               3   BiCGSTAB
               4   merged BiCGSTAB
               5   preconditioned BiCGSTAB
               6   GMRES
               7   preconditioned GMRES
               8   LOBPCG
               9   Iterative Refinement
               10  Jacobi
               11  Block-asynchronous Iteration

"--verbose k"
    k = 0 : solver is run in production mode, no additional characteristics 
            monitored
    k > 0 : solver is run in verbose mode, residual and runtime is monitored
            every k iterations

"--format k"
    k = 0 : CSR
    k = 1 : ELLPACK
    k = 2 : SELLP

"--blocksize k"
    for Magma_ELLPACKRT: denotes the number of rows in one slice of the matrix
                      and the number of rows assigned to one multiprocessor
    for Magma_SELLP: k denotes the number of rows in one slice of the matrix
                      and the number of rows assigned to one multiprocessor
    for Magma_SELLP: k denotes the number of rows in one slice of the matrix
                      and the number of rows assigned to one multiprocessor

"--alignment k"
    for Magma_ELLPACKRT: k denotes the number of threads assigned to one row
                      notice: blocksize * alignment needs to fit into shared mem
    for Magma_SELLCM: k denotes the number of threads assigned to one row
                      notice: blocksize * alignment needs to fit into shared mem

"--maxiter k"
    k : upper bound for iterations

"--tol k"
    k : relative residual stopping criterion (machine precision as default)

"--restart k"
    k : Krylov subspace restart number for GMRES-(k)


"--mscale k"
   k = 0 no scaling
   k = 1 scale symmetrically to unit diagonal
   k = 2 scale to unit rownorm

"--preconditioner k"
    k = 0 : Jacobi
    k = 1 : ILU/IC
    k = 2 : iterative ILU(0)/IC(0)
    Other preconditiners are only available for the Iterative Refinement.

"--ev k"
    k : number of egenvalue/eigenvectors to compute


The last argument is the traget matrices. These should be stored in MatrixMarket
format, see http://math.nist.gov/MatrixMarket/formats.html.
