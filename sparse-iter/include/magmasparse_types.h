/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014
*/

#ifndef MAGMASPARSE_TYPES_H
#define MAGMASPARSE_TYPES_H


#if defined(HAVE_PASTIX)
//PaStiX include
#include <stdint.h>
/* to access functions from the libpastix, respect this order */
#include <pastix.h>
#include <read_matrix.h>
#include <get_options.h>
#include <assert.h>
#endif

// includes CUDA
#include <cusparse_v2.h>



#ifdef __cplusplus
extern "C" {
#endif




typedef struct magma_z_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_symmetry_t   sym;
    magma_diagorder_t  diagorder_type;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    magma_int_t        diameter;
    magmaDoubleComplex *val;
    magmaDoubleComplex *diag;
    magma_index_t      *row; 
    magma_index_t      *rowidx; 
    magma_index_t      *col;
    magma_index_t      *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;
    magma_int_t        alignment;

}magma_z_sparse_matrix;

typedef struct magma_c_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_symmetry_t   sym;
    magma_diagorder_t  diagorder_type;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    magma_int_t        diameter;
    magmaFloatComplex  *val;
    magmaFloatComplex  *diag;
    magma_index_t      *row; 
    magma_index_t      *rowidx; 
    magma_index_t      *col;
    magma_index_t      *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;
    magma_int_t        alignment;

}magma_c_sparse_matrix;


typedef struct magma_d_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_symmetry_t   sym;
    magma_diagorder_t  diagorder_type;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    magma_int_t        diameter;
    double             *val;
    double             *diag;
    magma_index_t      *row; 
    magma_index_t      *rowidx; 
    magma_index_t      *col;
    magma_index_t      *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;
    magma_int_t        alignment;

}magma_d_sparse_matrix;


typedef struct magma_s_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_symmetry_t   sym;
    magma_diagorder_t  diagorder_type;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    magma_int_t        diameter;
    float              *val;
    float              *diag;
    magma_index_t      *row; 
    magma_index_t      *rowidx; 
    magma_index_t      *col;
    magma_index_t      *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;
    magma_int_t        alignment;

}magma_s_sparse_matrix;



typedef struct magma_z_vector{

    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        nnz; 
    magmaDoubleComplex *val;

}magma_z_vector;

typedef struct magma_c_vector{

    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        nnz; 
    magmaFloatComplex  *val;

}magma_c_vector;


typedef struct magma_d_vector{

    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        nnz; 
    double             *val;

}magma_d_vector;


typedef struct magma_s_vector{

    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        nnz; 
    float              *val;

}magma_s_vector;

//*****************     solver parameters     ********************************//

typedef struct magma_z_solver_par{

    magma_solver_type  solver;
    magma_int_t        version;
    double             epsilon;  
    magma_int_t        maxiter;
    magma_int_t        restart; 
    magma_ortho_t      ortho;
    magma_int_t        numiter;
    double             init_res;
    double             final_res;
    double             iter_res;
    real_Double_t      runtime;
    real_Double_t      *res_vec;
    real_Double_t      *timing;
    magma_int_t        verbose;
    magma_int_t        num_eigenvalues;
    magma_int_t        ev_length;
    double             *eigenvalues;
    magmaDoubleComplex *eigenvectors;
    magma_int_t        info;

//---------------------------------
// the input for verbose is:
// 0 = production mode
// k>0 = convergence and timing is monitored in *res_vec and *timeing every  
// k-th iteration 
//
// the output of info is:
//  0 = convergence (stopping criterion met)
// -1 = no convergence
// -2 = convergence but stopping criterion not met within maxiter
//--------------------------------

}magma_z_solver_par;



typedef struct magma_c_solver_par{

    magma_solver_type  solver;
    magma_int_t        version;
    float              epsilon;  
    magma_int_t        maxiter;
    magma_int_t        restart; 
    magma_ortho_t      ortho;
    magma_int_t        numiter;
    float              init_res;
    float              final_res;
    float              iter_res;
    real_Double_t      runtime;
    real_Double_t      *res_vec;
    real_Double_t      *timing;
    magma_int_t        verbose;
    magma_int_t        num_eigenvalues;
    magma_int_t        ev_length;
    float              *eigenvalues;
    magmaFloatComplex  *eigenvectors;
    magma_int_t        info;

//---------------------------------
// the input for verbose is:
// 0 = production mode
// k>0 = convergence and timing is monitored in *res_vec and *timeing every  
// k-th iteration 
//
// the output of info is:
//  0 = convergence (stopping criterion met)
// -1 = no convergence
// -2 = convergence but stopping criterion not met within maxiter
//--------------------------------

}magma_c_solver_par;



typedef struct magma_d_solver_par{

    magma_solver_type  solver;
    magma_int_t        version;
    double             epsilon;  
    magma_int_t        maxiter;
    magma_int_t        restart; 
    magma_ortho_t      ortho;
    magma_int_t        numiter;
    double             init_res;
    double             final_res;
    double             iter_res;
    real_Double_t      runtime;
    real_Double_t      *res_vec;
    real_Double_t      *timing;
    magma_int_t        verbose;
    magma_int_t        num_eigenvalues;
    magma_int_t        ev_length;
    double             *eigenvalues;
    double             *eigenvectors;
    magma_int_t        info;

//---------------------------------
// the input for verbose is:
// 0 = production mode
// k>0 = convergence and timing is monitored in *res_vec and *timeing every  
// k-th iteration 
//
// the output of info is:
//  0 = convergence (stopping criterion met)
// -1 = no convergence
// -2 = convergence but stopping criterion not met within maxiter
//--------------------------------

}magma_d_solver_par;



typedef struct magma_s_solver_par{

    magma_solver_type  solver;
    magma_int_t        version;
    float              epsilon;  
    magma_int_t        maxiter;
    magma_int_t        restart; 
    magma_ortho_t      ortho;
    magma_int_t        numiter;
    float              init_res;
    float              final_res;
    float              iter_res;
    real_Double_t      runtime;
    real_Double_t      *res_vec;
    real_Double_t      *timing;
    magma_int_t        verbose;
    magma_int_t        num_eigenvalues;
    magma_int_t        ev_length;
    float              *eigenvalues;
    float              *eigenvectors;
    magma_int_t        info;

//---------------------------------
// the input for verbose is:
// 0 = production mode
// k>0 = convergence and timing is monitored in *res_vec and *timeing every  
// k-th iteration 
//
// the output of info is:
//  0 = convergence (stopping criterion met)
// -1 = no convergence
// -2 = convergence but stopping criterion not met within maxiter
//--------------------------------

}magma_s_solver_par;



//************            preconditioner parameters       ********************//

typedef struct magma_z_preconditioner{

    magma_solver_type       solver;
    magma_int_t             levels;
    magma_int_t             sweeps;
    magma_precision         format;
    double                  epsilon;  
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    double                  init_res;
    double                  final_res;
    magma_z_sparse_matrix   M;
    magma_z_sparse_matrix   L;
    magma_z_sparse_matrix   U;
    magma_z_sparse_matrix   LD;
    magma_z_sparse_matrix   UD;
    magma_z_vector          d;
    magma_z_vector          work1;
    magma_z_vector          work2;
    magma_int_t*            int_array_1;
    magma_int_t*            int_array_2;
    cusparseSolveAnalysisInfo_t cuinfo;
    cusparseSolveAnalysisInfo_t cuinfoL;
    cusparseSolveAnalysisInfo_t cuinfoU;
#if defined(HAVE_PASTIX)
    pastix_data_t*          pastix_data;
    magma_int_t*            iparm;
    double*                 dparm;
#endif

}magma_z_preconditioner;

typedef struct magma_c_preconditioner{

    magma_solver_type       solver;
    magma_int_t             levels;
    magma_int_t             sweeps;
    magma_precision         format;
    float                   epsilon;  
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    float                   init_res;
    float                   final_res;
    magma_c_sparse_matrix   M;
    magma_c_sparse_matrix   L;
    magma_c_sparse_matrix   U;
    magma_c_sparse_matrix   LD;
    magma_c_sparse_matrix   UD;
    magma_c_vector          d;
    magma_c_vector          work1;
    magma_c_vector          work2;
    magma_int_t*            int_array_1;
    magma_int_t*            int_array_2;
    cusparseSolveAnalysisInfo_t cuinfo;
    cusparseSolveAnalysisInfo_t cuinfoL;
    cusparseSolveAnalysisInfo_t cuinfoU;
#if defined(HAVE_PASTIX)
    pastix_data_t*          pastix_data;
    magma_int_t*            iparm;
    float*                  dparm;
#endif

}magma_c_preconditioner;


typedef struct magma_d_preconditioner{

    magma_solver_type       solver;
    magma_int_t             levels;
    magma_int_t             sweeps;
    magma_precision         format;
    double                  epsilon;  
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    double                  init_res;
    double                  final_res;
    magma_d_sparse_matrix   M;
    magma_d_sparse_matrix   L;
    magma_d_sparse_matrix   U;
    magma_d_sparse_matrix   LD;
    magma_d_sparse_matrix   UD;
    magma_d_vector          d;
    magma_d_vector          work1;
    magma_d_vector          work2;
    magma_int_t*            int_array_1;
    magma_int_t*            int_array_2;
    cusparseSolveAnalysisInfo_t cuinfo;
    cusparseSolveAnalysisInfo_t cuinfoL;
    cusparseSolveAnalysisInfo_t cuinfoU;
#if defined(HAVE_PASTIX)
    pastix_data_t*          pastix_data;
    magma_int_t*            iparm;
    double*                 dparm;
#endif

}magma_d_preconditioner;


typedef struct magma_s_preconditioner{

    magma_solver_type       solver;
    magma_int_t             levels;
    magma_int_t             sweeps;
    magma_precision         format;
    float                   epsilon;  
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    float                   init_res;
    float                   final_res;
    magma_s_sparse_matrix   M;
    magma_s_sparse_matrix   L;
    magma_s_sparse_matrix   U;
    magma_s_sparse_matrix   LD;
    magma_s_sparse_matrix   UD;
    magma_s_vector          d;
    magma_s_vector          work1;
    magma_s_vector          work2;
    magma_int_t*            int_array_1;
    magma_int_t*            int_array_2;
    cusparseSolveAnalysisInfo_t cuinfo;
    cusparseSolveAnalysisInfo_t cuinfoL;
    cusparseSolveAnalysisInfo_t cuinfoU;
#if defined(HAVE_PASTIX)
    pastix_data_t*          pastix_data;
    magma_int_t*            iparm;
    float*                  dparm;
#endif

}magma_s_preconditioner;


//##############################################################################
//
//              opts for the testers
//
//##############################################################################

typedef struct magma_zopts{

    magma_z_solver_par      solver_par;
    magma_z_preconditioner  precond_par;
    magma_storage_t         input_format;
    int                     blocksize;
    int                     alignment;
    magma_storage_t         output_format;
    magma_location_t        input_location;
    magma_location_t        output_location;
    magma_scale_t           scaling;

}magma_zopts;

typedef struct magma_copts{

    magma_c_solver_par      solver_par;
    magma_c_preconditioner  precond_par;
    magma_storage_t         input_format;
    int                     blocksize;
    int                     alignment;
    magma_storage_t         output_format;
    magma_location_t        input_location;
    magma_location_t        output_location;
    magma_scale_t           scaling;

}magma_copts;

typedef struct magma_dopts{

    magma_d_solver_par      solver_par;
    magma_d_preconditioner  precond_par;
    magma_storage_t         input_format;
    int                     blocksize;
    int                     alignment;
    magma_storage_t         output_format;
    magma_location_t        input_location;
    magma_location_t        output_location;
    magma_scale_t           scaling;

}magma_dopts;

typedef struct magma_sopts{

    magma_s_solver_par      solver_par;
    magma_s_preconditioner  precond_par;
    magma_storage_t         input_format;
    int                     blocksize;
    int                     alignment;
    magma_storage_t         output_format;
    magma_location_t        input_location;
    magma_location_t        output_location;
    magma_scale_t           scaling;

}magma_sopts;

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMASPARSE_TYPES_H
