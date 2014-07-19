/**
    ------------------------------------------------------------
    @defgroup magma_init   Initialization
    @defgroup magma_util   Utilities

    ------------------------------------------------------------
    @defgroup solvers   Linear Systems
    @brief    Solve \f$ Ax = b \f$
    @{
        @defgroup magma_gesv   LU solve
        @brief    Solve \f$ Ax = b \f$, using LU factorization for general \f$ A \f$
        @{
            @defgroup magma_gesv_driver   LU solve: driver
            @brief    Whole \f$ Ax=b \f$ problem
            @{
                @defgroup magma_sgesv_driver single precision
                @defgroup magma_dgesv_driver double precision
                @defgroup magma_cgesv_driver single-complex precision
                @defgroup magma_zgesv_driver double-complex precision
            @}

            @defgroup magma_gesv_comp     LU solve: computational
            @brief    Major computational phases of solving \f$ Ax=b \f$
            @{
                @defgroup magma_sgesv_comp single precision
                @defgroup magma_dgesv_comp double precision
                @defgroup magma_cgesv_comp single-complex precision
                @defgroup magma_zgesv_comp double-complex precision
            @}

            @defgroup magma_gesv_aux      LU solve: auxiliary
            @brief    Low-level functions
            @{
                @defgroup magma_sgesv_aux single precision
                @defgroup magma_dgesv_aux double precision
                @defgroup magma_cgesv_aux single-complex precision
                @defgroup magma_zgesv_aux double-complex precision
            @}

            @defgroup magma_gesv_tile     Tiled LU
            @brief    Functions for tiled algorithms (incremental pivoting)
            @{
                @defgroup magma_sgesv_tile single precision
                @defgroup magma_dgesv_tile double precision
                @defgroup magma_cgesv_tile single-complex precision
                @defgroup magma_zgesv_tile double-complex precision
            @}
        @}

        @defgroup magma_posv   Cholesky solve
        @brief    Solve \f$ Ax = b \f$, using Cholesky factorization
                  for symmetric positive definite (SPD) \f$ A \f$
        @{
            @defgroup magma_posv_driver   Cholesky solve: driver
            @brief    Whole \f$ Ax=b \f$ (SPD) problem
            @{
                @defgroup magma_sposv_driver single precision
                @defgroup magma_dposv_driver double precision
                @defgroup magma_cposv_driver single-complex precision
                @defgroup magma_zposv_driver double-complex precision
            @}

            @defgroup magma_posv_comp     Cholesky solve: computational
            @brief    Major computational phases of solving \f$ Ax=b \f$ (SPD)
            @{
                @defgroup magma_sposv_comp single precision
                @defgroup magma_dposv_comp double precision
                @defgroup magma_cposv_comp single-complex precision
                @defgroup magma_zposv_comp double-complex precision
            @}

            @defgroup magma_posv_aux      Cholesky solve: auxiliary
            @brief    Low-level functions
            @{
                @defgroup magma_sposv_aux single precision
                @defgroup magma_dposv_aux double precision
                @defgroup magma_cposv_aux single-complex precision
                @defgroup magma_zposv_aux double-complex precision
            @}
        @}

        @defgroup magma_gels   Least Squares
        @brief    Solve over- or under-determined \f$ Ax = b \f$
        @{
            @defgroup magma_gels_driver   Least Squares solve: driver
            @brief    Whole \f$ Ax=b \f$ (least squares) problem
            @{
                @defgroup magma_sgels_driver single precision
                @defgroup magma_dgels_driver double precision
                @defgroup magma_cgels_driver single-complex precision
                @defgroup magma_zgels_driver double-complex precision
            @}

            @defgroup magma_gels_comp     Least Squares solve: computational
            @brief    Major computational phases of solving \f$ Ax=b \f$ (least squares); @sa orthogonal
            @{
                @defgroup magma_sgels_comp single precision
                @defgroup magma_dgels_comp double precision
                @defgroup magma_cgels_comp single-complex precision
                @defgroup magma_zgels_comp double-complex precision
            @}
        @}
    @}

    ------------------------------------------------------------
    @defgroup orthogonal   Orthogonal factorizations
    @brief    Factor \f$ A \f$, using QR, RQ, QL, LQ
    @{
        @defgroup magma_geqrf   QR factorization
        @brief    Factor \f$ A = QR \f$
        @{
            @defgroup magma_geqrf_comp   QR factorization: computational
            @brief    Major computational phase of least squares and SVD problems
            @{
                @defgroup magma_sgeqrf_comp single precision
                @defgroup magma_dgeqrf_comp double precision
                @defgroup magma_cgeqrf_comp single-complex precision
                @defgroup magma_zgeqrf_comp double-complex precision
            @}

            @defgroup magma_geqp3_comp   QR with pivoting
            @brief    Slower but more stable QR, especially for rank-deficient matrices
            @{
                @defgroup magma_sgeqp3_comp single precision
                @defgroup magma_dgeqp3_comp double precision
                @defgroup magma_cgeqp3_comp single-complex precision
                @defgroup magma_zgeqp3_comp double-complex precision
            @}

            @defgroup magma_geqrf_tile   Tiled QR factorization
            @brief    Functions for tiled algorithms
            @{
                @defgroup magma_sgeqrf_tile single precision
                @defgroup magma_dgeqrf_tile double precision
                @defgroup magma_cgeqrf_tile single-complex precision
                @defgroup magma_zgeqrf_tile double-complex precision
            @}
        @}

        @defgroup magma_gerqf_comp   RQ factorization
        @brief    Factor \f$ A = RQ \f$
            @{
                @defgroup magma_sgerqf_comp single precision
                @defgroup magma_dgerqf_comp double precision
                @defgroup magma_cgerqf_comp single-complex precision
                @defgroup magma_zgerqf_comp double-complex precision
            @}

        @defgroup magma_geqlf_comp   QL factorization
        @brief    Factor \f$ A = QL \f$
            @{
                @defgroup magma_sgeqlf_comp single precision
                @defgroup magma_dgeqlf_comp double precision
                @defgroup magma_cgeqlf_comp single-complex precision
                @defgroup magma_zgeqlf_comp double-complex precision
            @}

        @defgroup magma_gelqf_comp   LQ factorization
        @brief    Factor \f$ A = LQ \f$
            @{
                @defgroup magma_sgelqf_comp single precision
                @defgroup magma_dgelqf_comp double precision
                @defgroup magma_cgelqf_comp single-complex precision
                @defgroup magma_zgelqf_comp double-complex precision
            @}
    @}

    ------------------------------------------------------------
    @defgroup eigenvalue   Eigenvalue
    @brief    Solve \f$ Ax = \lambda x \f$
    @{
        @defgroup magma_geev   Non-symmetric eigenvalue
        @brief    Solve \f$ Ax = \lambda x \f$ for non-symmetric \f$ A \f$
        @{
            @defgroup magma_geev_driver   Non-symmetric eigenvalue: driver
            @brief    Whole \f$ Ax = \lambda x \f$ non-symmetric eigenvalue problem
            @{
                @defgroup magma_sgeev_driver single precision
                @defgroup magma_dgeev_driver double precision
                @defgroup magma_cgeev_driver single-complex precision
                @defgroup magma_zgeev_driver double-complex precision
            @}

            @defgroup magma_geev_comp     Non-symmetric eigenvalue: computational
            @brief    Major computational phases of non-symmetric eigenvalue problem
            @{
                @defgroup magma_sgeev_comp single precision
                @defgroup magma_dgeev_comp double precision
                @defgroup magma_cgeev_comp single-complex precision
                @defgroup magma_zgeev_comp double-complex precision
            @}

            @defgroup magma_geev_aux      Non-symmetric eigenvalue: auxiliary
            @brief    Low-level functions
            @{
                @defgroup magma_sgeev_aux single precision
                @defgroup magma_dgeev_aux double precision
                @defgroup magma_cgeev_aux single-complex precision
                @defgroup magma_zgeev_aux double-complex precision
            @}
        @}

        @defgroup magma_syev   Symmetric eigenvalue
        @brief    Solve \f$ Ax = \lambda x \f$ for symmetric \f$ A \f$
        @{
            @defgroup magma_syev_driver   Symmetric eigenvalue: driver
            @brief    Whole \f$ Ax = \lambda x \f$ eigenvalue problem
            @{
                @defgroup magma_ssyev_driver single precision
                @defgroup magma_dsyev_driver double precision
                @defgroup magma_cheev_driver single-complex precision
                @defgroup magma_zheev_driver double-complex precision
            @}

            @defgroup magma_sygv_driver   Generalized symmetric eigenvalue: driver
            @brief    Whole \f$ Ax = \lambda Bx \f$, or \f$ ABx = \lambda x \f$, or \f$ BAx = \lambda x \f$ generalized symmetric eigenvalue problem
            @{
                @defgroup magma_ssygv_driver single precision
                @defgroup magma_dsygv_driver double precision
                @defgroup magma_chegv_driver single-complex precision
                @defgroup magma_zhegv_driver double-complex precision
            @}


            @defgroup magma_syev_comp     Symmetric eigenvalue: computational
            @brief    Major computational phases of eigenvalue problem, 1-stage algorithm
            @{
                @defgroup magma_ssyev_comp single precision
                @defgroup magma_dsyev_comp double precision
                @defgroup magma_cheev_comp single-complex precision
                @defgroup magma_zheev_comp double-complex precision
            @}


            @defgroup magma_syev_2stage   Symmetric eigenvalue: computational, 2-stage
            @brief    Major computational phases of eigenvalue problem, 2-stage algorithm
            @{
                @defgroup magma_ssyev_2stage single precision
                @defgroup magma_dsyev_2stage double precision
                @defgroup magma_cheev_2stage single-complex precision
                @defgroup magma_zheev_2stage double-complex precision
            @}


            @defgroup magma_syev_aux      Symmetric eigenvalue: auxiliary
            @brief    Low-level functions
            @{
                @defgroup magma_ssyev_aux single precision
                @defgroup magma_dsyev_aux double precision
                @defgroup magma_cheev_aux single-complex precision
                @defgroup magma_zheev_aux double-complex precision
            @}
        @}
    @}

    ------------------------------------------------------------
    @defgroup magma_gesvd   Singular Value Decomposition (SVD)
    @brief    Compute SVD, \f$ A = U \Sigma V^T \f$
    @{
            @defgroup magma_gesvd_driver   SVD: driver
            @brief    Whole SVD problem
            @{
                @defgroup magma_sgesvd_driver single precision
                @defgroup magma_dgesvd_driver double precision
                @defgroup magma_cgesvd_driver single-complex precision
                @defgroup magma_zgesvd_driver double-complex precision
            @}

            @defgroup magma_gesvd_comp     SVD: computational
            @brief    Major computational phases of SVD problem
            @{
                @defgroup magma_sgesvd_comp single precision
                @defgroup magma_dgesvd_comp double precision
                @defgroup magma_cgesvd_comp single-complex precision
                @defgroup magma_zgesvd_comp double-complex precision
            @}

            @defgroup magma_gesvd_aux      SVD: auxiliary
            @brief    Low-level functions
            @{
                @defgroup magma_sgesvd_aux single precision
                @defgroup magma_dgesvd_aux double precision
                @defgroup magma_cgesvd_aux single-complex precision
                @defgroup magma_zgesvd_aux double-complex precision
            @}
    @}

    ------------------------------------------------------------
    @defgroup BLAS   BLAS and auxiliary
    @{
        @defgroup magma_blas1   Level-1 BLAS
        @brief    Level-1, vector operations: \f$ O(n) \f$ operations on \f$ O(n) \f$ data; memory bound
        @{
            @defgroup magma_sblas1 single precision
            @defgroup magma_dblas1 double precision
            @defgroup magma_cblas1 single-complex precision
            @defgroup magma_zblas1 double-complex precision
        @}

        @defgroup magma_blas2   Level-2 BLAS
        @brief    Level-2, matrix–vector operations: \f$ O(n^2) \f$ operations on \f$ O(n^2) \f$ data; memory bound
        @{
            @defgroup magma_sblas2 single precision
            @defgroup magma_dblas2 double precision
            @defgroup magma_cblas2 single-complex precision
            @defgroup magma_zblas2 double-complex precision
        @}

        @defgroup magma_blas3   Level-3 BLAS
        @brief    Level-3, matrix–matrix operations: \f$ O(n^3) \f$ operations on \f$ O(n^2) \f$ data; compute bound
        @{
            @defgroup magma_sblas3 single precision
            @defgroup magma_dblas3 double precision
            @defgroup magma_cblas3 single-complex precision
            @defgroup magma_zblas3 double-complex precision
        @}

        @defgroup magma_aux0    Math auxiliary
        @brief    Element operations, \f$ O(1) \f$ operations on \f$ O(1) \f$ data
        @{
            @defgroup magma_saux0 single precision
            @defgroup magma_daux0 double precision
            @defgroup magma_caux0 single-complex precision
            @defgroup magma_zaux0 double-complex precision
        @}

        @defgroup magma_aux1    Level-1 auxiliary
        @brief    Additional auxiliary Level-1 functions
        @{
            @defgroup magma_saux1 single precision
            @defgroup magma_daux1 double precision
            @defgroup magma_caux1 single-complex precision
            @defgroup magma_zaux1 double-complex precision
        @}

        @defgroup magma_aux2    Level-2 auxiliary
        @brief    Additional auxiliary Level-2 functions
        @{
            @defgroup magma_saux2 single precision
            @defgroup magma_daux2 double precision
            @defgroup magma_caux2 single-complex precision
            @defgroup magma_zaux2 double-complex precision
        @}

        @defgroup magma_aux3    Level-3 auxiliary
        @brief    Additional auxiliary Level-3 functions
        @{
            @defgroup magma_saux3 single precision
            @defgroup magma_daux3 double precision
            @defgroup magma_caux3 single-complex precision
            @defgroup magma_zaux3 double-complex precision
        @}

        @defgroup magma_aux     General auxiliary
        @brief    General (non-matrix) auxiliary functions

        @defgroup magma_comm    Communication
        @brief    CPU to GPU communication
        @{
            @defgroup magma_scommunication single precision
            @defgroup magma_dcommunication double precision
            @defgroup magma_ccommunication single-complex precision
            @defgroup magma_zcommunication double-complex precision
        @}
    @}
    
    ------------------------------------------------------------
    @defgroup sparse  Sparse
    @brief
    @{
        @defgroup sparse_solvers       Sparse Linear Systems
        @brief    Solve \f$ Ax = b \f$
        @{
            @defgroup sparse_gesv      General solver
            @brief    Solve \f$ Ax = b \f$, for general \f$ A \f$
            @{
                @defgroup magmasparse_sgesv single precision
                @defgroup magmasparse_dgesv double precision
                @defgroup magmasparse_cgesv single-complex precision
                @defgroup magmasparse_zgesv double-complex precision
            @}
            
            @defgroup sparse_posv      Symmetric (SPD) solver
            @brief    Solve \f$ Ax = b \f$,
                      for symmetric positive definite (SPD) \f$ A \f$
            @{
                @defgroup magmasparse_sposv single precision
                @defgroup magmasparse_dposv double precision
                @defgroup magmasparse_cposv single-complex precision
                @defgroup magmasparse_zposv double-complex precision
            @}
        @}
        
        @defgroup sparse_eigenvalue    Sparse eigenvalue
        @brief    Solve \f$ Ax = \lambda x \f$
        @{
            @defgroup sparse_geev      Non-symmetric eigenvalue
            @brief    Solve \f$ Ax = \lambda x \f$ for non-symmetric \f$ A \f$
            @{
                @defgroup magmasparse_sgeev single precision
                @defgroup magmasparse_dgeev double precision
                @defgroup magmasparse_cgeev single-complex precision
                @defgroup magmasparse_zgeev double-complex precision
            @}
            
            @defgroup sparse_syev      Symmetric eigenvalue
            @brief    Solve \f$ Ax = \lambda x \f$ for symmetric \f$ A \f$
            @{
                @defgroup magmasparse_ssyev single precision
                @defgroup magmasparse_dsyev double precision
                @defgroup magmasparse_csyev single-complex precision
                @defgroup magmasparse_zsyev double-complex precision
            @}
        @}

        @defgroup sparse_precond    Sparse preconditioner
        @brief    Preconditioner for solving \f$ Ax = \lambda x \f$
        @{
            @defgroup sparse_gepr      Non-symmetric preconditioner
            @brief    Preconditioner for \f$ Ax = \lambda x \f$ for non-symmetric \f$ A \f$
            @{
                @defgroup magmasparse_sgepr single precision
                @defgroup magmasparse_dgepr double precision
                @defgroup magmasparse_cgepr single-complex precision
                @defgroup magmasparse_zgepr double-complex precision
            @}
            
            @defgroup sparse_sypr      Symmetric preconditioner
            @brief    Preconditioner for \f$ Ax = \lambda x \f$ for symmetric \f$ A \f$
            @{
                @defgroup magmasparse_ssypr single precision
                @defgroup magmasparse_dsypr double precision
                @defgroup magmasparse_csypr single-complex precision
                @defgroup magmasparse_zsypr double-complex precision
            @}
        @}

        @defgroup sparse_gpukernels    GPU kernels for sparse LA
        @brief    Preconditioner for solving \f$ Ax = \lambda x \f$
        @{
            @defgroup sparse_gegpuk  GPU kernels for non-symmetric sparse LA  
            @brief    GPU kernels for Non-symmetric sparse LA
            @{
                @defgroup magmasparse_sgegpuk single precision
                @defgroup magmasparse_dgegpuk double precision
                @defgroup magmasparse_cgegpuk single-complex precision
                @defgroup magmasparse_zgegpuk double-complex precision
            @}
            
            @defgroup sparse_sygpuk  GPU kernels for symmetric sparse LA
            @brief    GPU kernels for symmetric sparse LA
            @{
                @defgroup magmasparse_ssgpuk single precision
                @defgroup magmasparse_dsgpuk double precision
                @defgroup magmasparse_csgpuk single-complex precision
                @defgroup magmasparse_zsgpuk double-complex precision
            @}
        @}
        
        @defgroup sparse_blas          Sparse BLAS
            @{
                @defgroup magmasparse_sblas single precision
                @defgroup magmasparse_dblas double precision
                @defgroup magmasparse_cblas single-complex precision
                @defgroup magmasparse_zblas double-complex precision
            @}

        @defgroup sparse_aux           Sparse auxiliary
            @{
                @defgroup magmasparse_saux single precision
                @defgroup magmasparse_daux double precision
                @defgroup magmasparse_caux single-complex precision
                @defgroup magmasparse_zaux double-complex precision
            @}

        @defgroup unfiled              Sparse ** unfiled **
            @{
                @defgroup magmasparse_s single precision
                @defgroup magmasparse_d double precision
                @defgroup magmasparse_c single-complex precision
                @defgroup magmasparse_z double-complex precision
            @}
    @}
*/

/*
        NOT YET IMPLEMENTED

        @defgroup magma_sysv   Symmetric indefinite solve
        @brief    Solve \f$ Ax = b \f$, using symmetric indefinite factorization
                  for symmetric \f$ A \f$
        @{
            @defgroup magma_sysv_driver   Symmetric indefinite solve: driver
            @brief    Whole \f$ Ax=b \f$ (symmetric) problem
            @{
                @defgroup magma_ssysv_driver single precision
                @defgroup magma_dsysv_driver double precision
                @defgroup magma_csysv_driver single-complex precision
                @defgroup magma_zsysv_driver double-complex precision
            @}

            @defgroup magma_sysv_comp     Symmetric indefinite solve: computational
            @brief    Major computational phases of solving \f$ Ax=b \f$ (symmetric)
            @{
                @defgroup magma_ssysv_comp single precision
                @defgroup magma_dsysv_comp double precision
                @defgroup magma_csysv_comp single-complex precision
                @defgroup magma_zsysv_comp double-complex precision
            @}

            @defgroup magma_sysv_aux      Symmetric indefinite solve: auxiliary
            @brief    Low-level functions
            @{
                @defgroup magma_ssysv_aux single precision
                @defgroup magma_dsysv_aux double precision
                @defgroup magma_csysv_aux single-complex precision
                @defgroup magma_zsysv_aux double-complex precision
            @}
        @}
*/

/*
        EMPTY GROUPS

            @defgroup magma_gels_aux      Least Squares solve: auxiliary
            @brief    Low-level functions
            @{
                @defgroup magma_sgels_aux single precision
                @defgroup magma_dgels_aux double precision
                @defgroup magma_cgels_aux single-complex precision
                @defgroup magma_zgels_aux double-complex precision
            @}

            @defgroup magma_geqrf_aux    QR factorization: auxiliary
            @brief    Low-level functions
            @{
                @defgroup magma_sgeqrf_aux single precision
                @defgroup magma_dgeqrf_aux double precision
                @defgroup magma_cgeqrf_aux single-complex precision
                @defgroup magma_zgeqrf_aux double-complex precision
            @}
*/
