/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Azzam Haidar
       @author Stan Tomov
       @author Raffaele Solca

       @precisions normal z -> s d c

 */
#include "common_magma.h"
#include "magma_bulge.h"
#include "magma_zbulge.h"
#include <cblas.h>


#define PRECISION_z

extern "C" {
    void magma_zstedc_withZ(char JOBZ, magma_int_t n, double *D, double * E, magmaDoubleComplex *Z, magma_int_t LDZ);
    void magma_zstedx_withZ(magma_int_t n, magma_int_t ne, double *D, double * E, magmaDoubleComplex *Z, magma_int_t LDZ);
}

static void* parallel_section(void *arg);

static void tile_bulge_parallel(magma_int_t my_core_id, magma_int_t cores_num, magmaDoubleComplex *A, magma_int_t lda,
                                magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magma_int_t n, magma_int_t nb, magma_int_t nb_tiles,
                                magma_int_t band, magma_int_t grsiz, magma_int_t Vblksiz, volatile magma_int_t *prog);

static void tile_bulge_computeT_parallel(magma_int_t my_core_id, magma_int_t cores_num, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU,
                                         magmaDoubleComplex *T, magma_int_t ldt, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz);

static void *applyQ_parallel_section(void *arg);

static void tile_bulge_applyQ(magma_int_t wantz, char side, magma_int_t n_cpu, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
                              magmaDoubleComplex *E, magma_int_t lde, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magmaDoubleComplex *T, magma_int_t ldt);

//////////////////////////////////////////////////////////////////////////////////////////////////////////

class bulge_data {

public:

    bulge_data(magma_int_t threads_num_, magma_int_t n_, magma_int_t nb_, magma_int_t nbtiles_,
               magma_int_t band_, magma_int_t grsiz_, magma_int_t Vblksiz_, magma_int_t wantz_,
               magmaDoubleComplex *A_, magma_int_t lda_, magmaDoubleComplex *V_, magma_int_t ldv_, magmaDoubleComplex *TAU_,
               magmaDoubleComplex *T_, magma_int_t ldt_, magma_int_t computeQ1_, magmaDoubleComplex *dQ1_, magma_int_t lddq1_,
               magmaDoubleComplex *dT1_, volatile magma_int_t* prog_)
    :
    threads_num(threads_num_),
    n(n_),
    nb(nb_),
    nbtiles(nbtiles_),
    band(band_),
    grsiz(grsiz_),
    Vblksiz(Vblksiz_),
    wantz(wantz_),
    A(A_),
    lda(lda_),
    V(V_),
    ldv(ldv_),
    TAU(TAU_),
    T(T_),
    ldt(ldt_),
    computeQ1(computeQ1_),
    dQ1(dQ1_),
    lddq1(lddq1_),
    dT1(dT1_),
    prog(prog_)
    {
        magma_int_t count = threads_num;

        if(wantz > 0 && threads_num > 1 && computeQ1 == 1)
            --count;

        pthread_barrier_init(&barrier, NULL, count);
    }

    ~bulge_data()
    {
        pthread_barrier_destroy(&barrier);
    }

    const magma_int_t threads_num;
    const magma_int_t n;
    const magma_int_t nb;
    const magma_int_t nbtiles;
    const magma_int_t band;
    const magma_int_t grsiz;
    const magma_int_t Vblksiz;
    const magma_int_t wantz;
    magmaDoubleComplex* const A;
    const magma_int_t lda;
    magmaDoubleComplex* const V;
    const magma_int_t ldv;
    magmaDoubleComplex* const TAU;
    magmaDoubleComplex* const T;
    const magma_int_t ldt;
    const magma_int_t computeQ1;
    magmaDoubleComplex* const dQ1;
    const magma_int_t lddq1;
    magmaDoubleComplex* const dT1;
    volatile magma_int_t *prog;
    pthread_barrier_t barrier;

private:

    bulge_data(bulge_data& data); // disable copy

};

class bulge_id_data {

public:

    bulge_id_data()
    : id(-1), data(NULL)
    {}

    bulge_id_data(magma_int_t id_, bulge_data* data_)
    : id(id_), data(data_)
    {}

    magma_int_t id;
    bulge_data* data;

};

class applyQ_data {

public:

    applyQ_data(magma_int_t threads_num_, magma_int_t n_, magma_int_t ne_, magma_int_t n_gpu_,
                magma_int_t nb_, magma_int_t Vblksiz_, magma_int_t wantz_,
                magmaDoubleComplex *E_, magma_int_t lde_, magmaDoubleComplex *V_, magma_int_t ldv_, magmaDoubleComplex *TAU_,
                magmaDoubleComplex *T_, magma_int_t ldt_, magmaDoubleComplex *dE_, magma_int_t ldde_)
    :
    threads_num(threads_num_),
    n(n_),
    ne(ne_),
    n_gpu(n_gpu_),
    nb(nb_),
    Vblksiz(Vblksiz_),
    wantz(wantz_),
    E(E_),
    lde(lde_),
    V(V_),
    ldv(ldv_),
    TAU(TAU_),
    T(T_),
    ldt(ldt_),
    dE(dE_),
    ldde(ldde_)
    {
        magma_int_t count = threads_num;

        if(threads_num > 1)
            --count;

        pthread_barrier_init(&barrier, NULL, count);
    }

    ~applyQ_data()
    {
        pthread_barrier_destroy(&barrier);
    }

    const magma_int_t threads_num;
    const magma_int_t n;
    const magma_int_t ne;
    const magma_int_t n_gpu;
    const magma_int_t nb;
    const magma_int_t Vblksiz;
    const magma_int_t wantz;
    magmaDoubleComplex* const E;
    const magma_int_t lde;
    magmaDoubleComplex* const V;
    const magma_int_t ldv;
    magmaDoubleComplex* const TAU;
    magmaDoubleComplex* const T;
    const magma_int_t ldt;
    magmaDoubleComplex* const dE;
    const magma_int_t ldde;
    pthread_barrier_t barrier;

private:

    applyQ_data(applyQ_data& data); // disable copy

};

class applyQ_id_data {

public:

    applyQ_id_data()
    : id(-1), data(NULL)
    {}

    applyQ_id_data(magma_int_t id_, applyQ_data* data_)
    : id(id_), data(data_)
    {}

    magma_int_t id;
    applyQ_data* data;

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" 
magma_int_t magma_zhetrd_bhe2trc_v5(magma_int_t threads, magma_int_t wantz, char uplo, 
                                    magma_int_t ne, magma_int_t n, magma_int_t nb,
                                    magmaDoubleComplex *A, magma_int_t lda, 
                                    double *D, double *E,
                                    magmaDoubleComplex *dT1, magma_int_t ldt1)
{
    ////////////////////////////////////////////////////////////////
    magmaDoubleComplex *da, *Z;
    if((wantz>0)/*&&(wantz!=4)*/){
        if (MAGMA_SUCCESS != magma_zmalloc( &da, n*lda )) {
            return MAGMA_ERR_DEVICE_ALLOC;
        }

    }
    magmaDoubleComplex *dQ1 = da;
    magma_int_t lddq1=lda;
    ////////////////////////////////////////////////////////////////
    char uplo_[2] = {uplo, 0};
    double timelpk=0.0, timeaplQ2=0.0;
    double timeblg=0.0, timeeigen=0.0, timegemm =0.0;

    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    magma_int_t mklth = threads;
    magma_int_t computeQ1 = 0;

    magma_int_t band=6, INgrsiz=1;

#if (defined(PRECISION_s) || defined(PRECISION_d))
    magma_int_t Vblksiz = min(nb,48);
#else
    magma_int_t Vblksiz = min(nb,32);
#endif

    magma_int_t ldt = Vblksiz;
    magma_int_t ldv = nb + Vblksiz - 1;
    magma_int_t blkcnt = magma_bulge_get_blkcnt(n, nb, Vblksiz);

    magma_int_t nbtiles = magma_ceildiv(n, nb);

    /*************************************************
     *     INITIALIZING MATRICES
     * ***********************************************/
    /* A1 is equal to A2 */
    magma_int_t lda2 = nb+1+(nb-1);
    if(lda2>n){
        printf("error matrix too small N=%d NB=%d\n",n,nb);
        return -14;
    }
    if((n<=0)||(nb<=0)){
        printf("error N or NB <0 N=%d NB=%d\n",n,nb);
        return -14;
    }

    timelpk = magma_wtime();
    /* copy the input matrix into a matrix A2 with band storage */
    magmaDoubleComplex *A2;
    magma_zmalloc_cpu(&A2, n*lda2);

    memset(A2, 0, n*lda2*sizeof(magmaDoubleComplex));

    for (magma_int_t j = 0; j < n-nb; j++)
    {
        cblas_zcopy(nb+1, &A[j*(lda+1)], 1, &A2[j*lda2], 1);
        memset(&A[j*(lda+1)], 0, (nb+1)*sizeof(magmaDoubleComplex));
        A[nb + j*(lda+1)] = c_one;
    }
    for (magma_int_t j = 0; j < nb; j++)
    {
        cblas_zcopy(nb-j, &A[(j+n-nb)*(lda+1)], 1, &A2[(j+n-nb)*lda2], 1);
        memset(&A[(j+n-nb)*(lda+1)], 0, (nb-j)*sizeof(magmaDoubleComplex));
    }

    if(wantz>0)
       magma_zsetmatrix( n, n, A, lda, dQ1, lddq1 );

    timelpk = magma_wtime() - timelpk;
    printf("  Finish CONVERT timing= %f\n", timelpk);

    magmaDoubleComplex *T;
    magmaDoubleComplex *TAU;
    magmaDoubleComplex *V;

    magma_zmalloc_cpu(&T,   blkcnt*ldt*Vblksiz);
    magma_zmalloc_cpu(&TAU, blkcnt*Vblksiz    );
    magma_zmalloc_cpu(&V,   blkcnt*ldv*Vblksiz);

    memset(T,   0, blkcnt*ldt*Vblksiz*sizeof(magmaDoubleComplex));
    memset(TAU, 0, blkcnt*Vblksiz*sizeof(magmaDoubleComplex));
    memset(V,   0, blkcnt*ldv*Vblksiz*sizeof(magmaDoubleComplex));

    magma_int_t* prog;
    magma_malloc_cpu((void**) &prog, (2*nbtiles+threads+10)*sizeof(magma_int_t));
    memset(prog, 0, (2*nbtiles+threads+10)*sizeof(magma_int_t));

    bulge_id_data* arg;
    magma_malloc_cpu((void**) &arg, threads*sizeof(bulge_id_data));

    pthread_t* thread_id;
    magma_malloc_cpu((void**) &thread_id, threads*sizeof(pthread_t));

    pthread_attr_t thread_attr;

    magma_setlapack_numthreads(1);

    if (wantz==2 || wantz==3)
        computeQ1 = 1;

    bulge_data data_bulge(threads, n, nb, nbtiles, band, INgrsiz, Vblksiz, wantz,
                          A2, lda2, V, ldv, TAU, T, ldt, computeQ1, dQ1, lddq1, dT1, prog);

    // Set one thread per core
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
    pthread_setconcurrency(threads);

    //timing
    timeblg = magma_wtime();

    // Launch threads
    for (magma_int_t thread = 1; thread < threads; thread++)
    {
        arg[thread] = bulge_id_data(thread, &data_bulge);
        pthread_create(&thread_id[thread], &thread_attr, parallel_section, &arg[thread]);
    }
    arg[0] = bulge_id_data(0, &data_bulge);
    parallel_section(&arg[0]);

    // Wait for completion
    for (magma_int_t thread = 1; thread < threads; thread++)
    {
        void *exitcodep;
        pthread_join(thread_id[thread], &exitcodep);
    }

    // timing
    timeblg = magma_wtime()-timeblg;


    magma_free_cpu(thread_id);
    magma_free_cpu(arg);
    magma_free_cpu(prog);

    printf("  Finish BULGE+T timing= %f\n", timeblg);

    /*================================================
     *  store resulting diag and lower diag D and E
     *  note that D and E are always real
     *================================================*/

    /* Make diagonal and superdiagonal elements real,
     * storing them in D and E
     */
    /* In complex case, the off diagonal element are
     * not necessary real. we have to make off-diagonal
     * elements real and copy them to E.
     * When using HouseHolder elimination,
     * the ZLARFG give us a real as output so, all the
     * diagonal/off-diagonal element except the last one are already
     * real and thus we need only to take the abs of the last
     * one.
     *  */

#if defined(PRECISION_z) || defined(PRECISION_c)
    if(uplo==MagmaLower){
        for (magma_int_t i=0; i < n-1 ; i++)
        {
            D[i] = MAGMA_Z_REAL(A2[i*lda2  ]);
            E[i] = MAGMA_Z_REAL(A2[i*lda2+1]);
        }
        D[n-1] = MAGMA_Z_REAL(A2[(n-1)*lda2]);
    } else { /* MagmaUpper not tested yet */
        for (magma_int_t i=0; i<n-1; i++)
        {
            D[i]  =  MAGMA_Z_REAL(A2[i*lda2+nb]);
            E[i] = MAGMA_Z_REAL(A2[i*lda2+nb-1]);
        }
        D[n-1] = MAGMA_Z_REAL(A2[(n-1)*lda2+nb]);
    } /* end MagmaUpper */
#else
    if( uplo == MagmaLower ){
        for (magma_int_t i=0; i < n-1; i++) {
            D[i] = A2[i*lda2];   // diag
            E[i] = A2[i*lda2+1]; //lower diag
        }
        D[n-1] = A2[(n-1)*lda2];
    } else {
        for (magma_int_t i=0; i < n-1; i++) {
            D[i] = A2[i*lda2+nb];   // diag
            E[i] = A2[i*lda2+nb-1]; //lower diag
        }
        D[n-1] = A2[(n-1)*lda2+nb];
    }
#endif
/*
    if (wantz != 4)
        magma_free( dT1 );  //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
*/
    magma_int_t ldz=n;
    magmaDoubleComplex *dZ;
    magma_int_t info;

    // in case for tridiagonal testing
    if (wantz < 0){
        goto fin;
    }

    magma_setlapack_numthreads(mklth);

    timeeigen = magma_wtime();
    if(wantz==0){
        timelpk = magma_wtime();

        magma_zstedc_withZ(MagmaNoVec, n, D, E, NULL, 1);
        //magma_zstedc_withZ(MagmaNoVec, n, D, E, Z, ldz);

        timelpk = magma_wtime()-timelpk;
        printf("  Finish WANTZ %d  eigensolver 'N'    timing=%f  threads %d\n", wantz, timelpk, mklth);

    }
    else {

        timelpk = magma_wtime();

        magma_zmalloc_cpu (&Z, n*n);

        magma_zstedx_withZ(n, ne, D, E, Z, ldz);

        magma_setlapack_numthreads(1);

        timelpk = magma_wtime()-timelpk;

        if(wantz==2){
            /////////////////////////////////
            //  WANTZ 2
            /////////////////////////////////

            double f= 1.;
            magma_int_t n_gpu = n;

            if(threads>40){
                f = 0.5;
                n_gpu = (magma_int_t)(f*n)/64*64;
            }
            else if(threads>10){
#if (defined(PRECISION_s) || defined(PRECISION_d))
                f = 0.68;
#else
                f = 0.72;
#endif
                n_gpu = (magma_int_t)(f*n)/64*64;
            }
            else if(threads>5){
#if (defined(PRECISION_s) || defined(PRECISION_d))
                f = 0.84;
#else
                f = 0.86;
#endif
                n_gpu = (magma_int_t)(f*n)/64*64;
            }

            /****************************************************
             * apply V2 from Right to Q1. da = da*(I-V2*T2*V2')
             * **************************************************/
            timeaplQ2 = magma_wtime();
            /*============================
             *  use GPU+CPU's
             *==========================*/
            if(n_gpu < n)
            {

                // define the size of Q to be done on CPU's and the size on GPU's
                // note that GPU use Q(1:N_GPU) and CPU use Q(N_GPU+1:N)

                magma_zgetmatrix( n-n_gpu, n, &da[n_gpu], lda, &A[n_gpu], lda);

                printf("---> calling GPU + CPU to apply V2 to Q1 with N %d     N_GPU %d   N_CPU %d\n",n, n_gpu, n-n_gpu);

                applyQ_data data_applyQ(threads, n, ne, n_gpu, nb, Vblksiz, wantz, A, lda, V, ldv, TAU, T, ldt, da, lda);

                applyQ_id_data* arg;
                magma_malloc_cpu((void**) &arg, threads*sizeof(applyQ_id_data));

                pthread_t* thread_id;
                magma_malloc_cpu((void**) &thread_id, threads*sizeof(pthread_t));

                //pthread_attr_t thread_attr;

                // ===============================
                // relaunch thread to apply Q
                // ===============================
                // Set one thread per core
                pthread_attr_init(&thread_attr);
                pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
                pthread_setconcurrency(threads);

                // Launch threads
                for (magma_int_t thread = 1; thread < threads; thread++)
                {
                    arg[thread] = applyQ_id_data(thread, &data_applyQ);
                    pthread_create(&thread_id[thread], &thread_attr, applyQ_parallel_section, &arg[thread]);
                }
                arg[0] = applyQ_id_data(0, &data_applyQ);
                applyQ_parallel_section(&arg[0]);

                // Wait for completion
                for (magma_int_t thread = 1; thread < threads; thread++)
                {
                    void *exitcodep;
                    pthread_join(thread_id[thread], &exitcodep);
                }

                magma_free_cpu(thread_id);
                magma_free_cpu(arg);

                magma_zsetmatrix( n-n_gpu, n, &A[n_gpu], lda, &da[n_gpu], lda);

                /*============================
                 *  use only GPU
                 *==========================*/
            }else{
                magma_zbulge_applyQ_v2('R', n, n, nb, Vblksiz, dQ1, n, V, ldv, T, ldt, &info);
            }
            timeaplQ2 = magma_wtime()-timeaplQ2;

            /****************************************************
             * compute the GEMM of Q*Z
             * **************************************************/

            if(MAGMA_SUCCESS != magma_zmalloc( &dZ, n*ne )) {
                printf ("!!!! magma_alloc failed for: dZ\n" );
                exit(-1);
            }
            timegemm = magma_wtime();
            // copy the eigenvectors to GPU
            magma_zsetmatrix( n, ne, Z, ldz, dZ, n );
            //make a gemm of (Q1 * Q2) * Z = da * dZ --> dZ2
            magmaDoubleComplex *dZ2;
            if(MAGMA_SUCCESS != magma_zmalloc( &dZ2, n*ne )) {
                printf ("!!!! magma_alloc failed for: dZ2\n" );
                exit(-1);
            }
            magma_zgemm( MagmaNoTrans, MagmaNoTrans, n, ne, n, c_one, da, n, dZ, n, c_zero, dZ2, n);
            magma_zgetmatrix( n, ne, dZ2, n, A, lda );
            magma_free(dZ2);
            timegemm = magma_wtime()-timegemm;
        }
        else if(wantz==3 || wantz==4){

            /////////////////////////////////
            //  WANTZ 3 and 4
            /////////////////////////////////

            double f= 1.;
            magma_int_t n_gpu = ne;

            if(threads>40){
                f = 0.5;
                n_gpu = (magma_int_t)(f*ne)/64*64;
            }
            else if(threads>10){
#if (defined(PRECISION_s) || defined(PRECISION_d))
                f = 0.68;
#else
                f = 0.72;
#endif
                n_gpu = (magma_int_t)(f*ne)/64*64;
            }
            else if(threads>5){
#if (defined(PRECISION_s) || defined(PRECISION_d))
                f = 0.82;
#else
                f = 0.86;
#endif
                n_gpu = (magma_int_t)(f*ne)/64*64;
            }
            else if(threads>2){
#if (defined(PRECISION_s) || defined(PRECISION_d))
                f = 0.96;
#else
                f = 0.96;
#endif
                n_gpu = (magma_int_t)(f*ne)/64*64;
            }

            /****************************************************
             *  apply V2 from left to the eigenvectors Z. dZ = (I-V2*T2*V2')*Z
             * **************************************************/
            if(MAGMA_SUCCESS != magma_zmalloc( &dZ, n*ne )) {
                printf ("!!!! magma_alloc failed for: dZ\n" );
                exit(-1);
            }
            timeaplQ2 = magma_wtime();

            /*============================
             *  use GPU+CPU's
             *==========================*/
            if(n_gpu < ne)
            {

                // define the size of Q to be done on CPU's and the size on GPU's
                // note that GPU use Q(1:N_GPU) and CPU use Q(N_GPU+1:N)

                printf("---> calling GPU + CPU(if N_CPU>0) to apply V2 to Z with NE %d     N_GPU %d   N_CPU %d\n",ne, n_gpu, ne-n_gpu);

                applyQ_data data_applyQ(threads, n, ne, n_gpu, nb, Vblksiz, wantz, Z, ldz, V, ldv, TAU, T, ldt, dZ, n);

                applyQ_id_data* arg;
                magma_malloc_cpu((void**) &arg, threads*sizeof(applyQ_id_data));

                pthread_t* thread_id;
                magma_malloc_cpu((void**) &thread_id, threads*sizeof(pthread_t));

                pthread_attr_t thread_attr;

                // ===============================
                // relaunch thread to apply Q
                // ===============================
                // Set one thread per core
                pthread_attr_init(&thread_attr);
                pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
                pthread_setconcurrency(threads);

                // Launch threads
                for (magma_int_t thread = 1; thread < threads; thread++)
                {
                    arg[thread] = applyQ_id_data(thread, &data_applyQ);
                    pthread_create(&thread_id[thread], &thread_attr, applyQ_parallel_section, &arg[thread]);
                }
                arg[0] = applyQ_id_data(0, &data_applyQ);
                applyQ_parallel_section(&arg[0]);

                // Wait for completion
                for (magma_int_t thread = 1; thread < threads; thread++)
                {
                    void *exitcodep;
                    pthread_join(thread_id[thread], &exitcodep);
                }

                magma_free_cpu(thread_id);
                magma_free_cpu(arg);

                magma_zsetmatrix(n, ne-n_gpu, Z + n_gpu*ldz, ldz, dZ + n_gpu*ldz, n);

                /*============================
                 *  use only GPU
                 *==========================*/
            }else{
                magma_zsetmatrix(n, ne, Z, ldz, dZ, n);
                magma_zbulge_applyQ_v2('L', ne, n, nb, Vblksiz, dZ, n, V, ldv, T, ldt, &info);
                magma_device_sync();
            }
            timeaplQ2 = magma_wtime()-timeaplQ2;

            if (wantz==3){
                /****************************************************
                 * compute the GEMM of Q1 * (Q2*Z)
                 * **************************************************/
                printf("calling zgemm\n");
                timegemm = magma_wtime();
                //make a gemm of Q1 * (Q2 * Z) = Q1 * ((I-V2T2V2')*Z) = da * dZ --> dZ2
                magmaDoubleComplex *dZ2;
                if(MAGMA_SUCCESS != magma_zmalloc( &dZ2, n*ne )) {
                    printf ("!!!! magma_alloc failed for: dZ2\n" );
                    exit(-1);
                }
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, n, ne, n, c_one, dQ1, lddq1, dZ, n, c_zero, dZ2, n);
                magma_zgetmatrix( n, ne, dZ2, n, A, lda );
                magma_free(dZ2);
                timegemm = magma_wtime()-timegemm;
            }else{
                /****************************************************
                 * apply Q1 to (Q2*Z)
                 * **************************************************/
                printf("calling zunmqr_gpu_2stages\n");
                timegemm = magma_wtime();
                magma_zunmqr_gpu_2stages(MagmaLeft, MagmaNoTrans, n-nb, ne, n-nb, dQ1+nb, lddq1,
                                         dZ+nb, n, dT1, nb, &info);
                magma_free(dT1);
                magma_zgetmatrix( n, ne, dZ, n, A, lda );
                timegemm = magma_wtime()-timegemm;
            }
        }
        magma_setlapack_numthreads(mklth);

        magma_free_cpu(Z);
        magma_free(dZ);
        magma_free(dQ1);
    }
    timeeigen = magma_wtime()-timeeigen;


fin:

    magma_free_cpu(A2);
    magma_free_cpu(TAU);
    magma_free_cpu(V);
    magma_free_cpu(T);
    //magma_free_pinned(V);
    //magma_free_pinned(T);

    printf("============================================================================\n");
    printf("  Finish WANTZ %d  computing Q2       timing= %f\n", wantz, timeaplQ2);

    printf("  Finish WANTZ %d  gemm Q1 / apply Q1 timing= %f\n", wantz, timegemm);
    printf("  Finish WANTZ %d  eigensolver 'I'    timing= %f  threads %d   N %d    NE %d\n", wantz, timelpk, mklth, n, ne);

    printf("  Finish WANTZ %d  full Eigenvectros  timing= %f\n", wantz, timeeigen);
    printf("============================================================================\n");



    return MAGMA_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

static void *parallel_section(void *arg)
{
    magma_int_t my_core_id   = ((bulge_id_data*)arg) -> id;
    bulge_data* data = ((bulge_id_data*)arg) -> data;

    magma_int_t allcores_num   = data -> threads_num;
    magma_int_t n              = data -> n;
    magma_int_t nb             = data -> nb;
    magma_int_t nbtiles        = data -> nbtiles;
    magma_int_t band           = data -> band;
    magma_int_t grsiz          = data -> grsiz;
    magma_int_t Vblksiz        = data -> Vblksiz;
    magma_int_t wantz          = data -> wantz;
    magmaDoubleComplex *A         = data -> A;
    magma_int_t lda            = data -> lda;
    magmaDoubleComplex *V         = data -> V;
    magma_int_t ldv            = data -> ldv;
    magmaDoubleComplex *TAU       = data -> TAU;
    magmaDoubleComplex *T         = data -> T;
    magma_int_t ldt            = data -> ldt;
    magma_int_t computeQ1      = data -> computeQ1;
    magmaDoubleComplex *dQ1       = data -> dQ1;
    magma_int_t lddq1          = data -> lddq1;
    magmaDoubleComplex *dT1       = data -> dT1;
    volatile magma_int_t* prog = data -> prog;

    pthread_barrier_t* barrier = &(data -> barrier);

    magma_int_t sys_corenbr    = 1;

    magma_int_t info;

    double timeB=0.0, timeT=0.0, timeaplQ1=0.0;

    // with MKL and when using omp_set_num_threads instead of mkl_set_num_threads
    // it need that all threads setting it to 1.
    magma_setlapack_numthreads(1);

#if defined(MAGMA_SETAFFINITY)
    // bind threads
    cpu_set_t set;
    // bind threads
    CPU_ZERO( &set );
    CPU_SET( my_core_id, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif

    if((wantz>0))
    {
        /* compute the Q1 overlapped with the bulge chasing+T.
         * if all_cores_num=1 it call Q1 on GPU and then bulgechasing.
         * otherwise the first thread run Q1 on GPU and
         * the other threads run the bulgechasing.
         * */

        if(allcores_num==1)
        {
            //=========================
            //    compute Q1
            //=========================
            if(computeQ1==1){
                magma_device_sync();
                timeaplQ1 = magma_wtime();

                magma_zungqr_2stage_gpu(n, n, n, dQ1, lddq1, NULL, dT1, nb, &info);
                magma_device_sync();

                timeaplQ1 = magma_wtime()-timeaplQ1;
                printf("  Finish applyQ1 timing= %f\n", timeaplQ1);
            }

            //=========================
            //    bulge chasing
            //=========================
            timeB = magma_wtime();

            tile_bulge_parallel(0, 1, A, lda, V, ldv, TAU, n, nb, nbtiles, band, grsiz, Vblksiz, prog);

            timeB = magma_wtime()-timeB;
            printf("  Finish BULGE   timing= %f\n", timeB);


            //=========================
            // compute the T's to be used when applying Q2
            //=========================
            timeT = magma_wtime();
            tile_bulge_computeT_parallel(0, 1, V, ldv, TAU, T, ldt, n, nb, Vblksiz);

            timeT = magma_wtime()-timeT;
            printf("  Finish T's     timing= %f\n", timeT);

        }else{ // allcore_num > 1

            magma_int_t id  = -1;
            magma_int_t tot = -1;

            if(computeQ1==1){
                id = my_core_id-1;
                tot = allcores_num-1;
            }
            else {
                id = my_core_id;
                tot = allcores_num;
            }

            if(computeQ1 == 1 && my_core_id == 0)
            {
                //=============================================
                //    compute Q1 on last newcoreid
                //=============================================
                magma_device_sync();
                timeaplQ1 = magma_wtime();

                magma_zungqr_2stage_gpu(n, n, n, dQ1, lddq1, NULL, dT1, nb, &info);
                magma_device_sync();

                timeaplQ1 = magma_wtime()-timeaplQ1;
                printf("  Finish applyQ1 timing= %f\n", timeaplQ1);

            }else{
                //=========================
                //    bulge chasing
                //=========================
                if(id == 0)timeB = magma_wtime();

                tile_bulge_parallel(id, tot, A, lda, V, ldv, TAU, n, nb, nbtiles, band, grsiz, Vblksiz, prog);
                pthread_barrier_wait(barrier);

                if(id == 0){
                    timeB = magma_wtime()-timeB;
                    printf("  Finish BULGE   timing= %f\n", timeB);
                }

                //=========================
                // compute the T's to be used when applying Q2
                //=========================
                if(id == 0)timeT = magma_wtime();

                tile_bulge_computeT_parallel(id, tot, V, ldv, TAU, T, ldt, n, nb, Vblksiz);
                pthread_barrier_wait(barrier);

                if (id == 0){
                    timeT = magma_wtime()-timeT;
                    printf("  Finish T's     timing= %f\n", timeT);
                }
            }

        } // allcore == 1

    }else{ // WANTZ = 0

        //=========================
        //    bulge chasing
        //=========================
        if(my_core_id == 0)
            timeB = magma_wtime();

        tile_bulge_parallel(my_core_id, allcores_num, A, lda, V, ldv, TAU, n, nb, nbtiles, band, grsiz, Vblksiz, prog);

        pthread_barrier_wait(barrier);

        if(my_core_id == 0){
            timeB = magma_wtime()-timeB;
            printf("  Finish BULGE   timing= %f\n", timeB);
        }
    } // WANTZ > 0

#if defined(MAGMA_SETAFFINITY)
    // unbind threads
    sys_corenbr = sysconf(_SC_NPROCESSORS_ONLN);
    CPU_ZERO( &set );
    for(magma_int_t i=0; i<sys_corenbr; i++)
        CPU_SET( i, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void tile_bulge_parallel(magma_int_t my_core_id, magma_int_t cores_num, magmaDoubleComplex *A, magma_int_t lda,
                                magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magma_int_t n, magma_int_t nb, magma_int_t nbtiles,
                                magma_int_t band, magma_int_t grsiz, magma_int_t Vblksiz, volatile magma_int_t *prog)
{
    magma_int_t sweepid, myid, shift, stt, st, ed, stind, edind;
    magma_int_t blklastind, colpt;
    magma_int_t stepercol;
    magma_int_t i,j,m,k;
    magma_int_t thgrsiz, thgrnb, thgrid, thed;
    magma_int_t coreid;
    magma_int_t colblktile,maxrequiredcores,colpercore,mycoresnb;
    magma_int_t fin;
    magmaDoubleComplex *work;

    if(n<=0)
        return ;

    //printf("=================> my core id %d of %d \n",my_core_id, cores_num);

    if((band!=0) && (band!=6) && (band!=62) && (band!=63)){
        if(my_core_id==0)printf(" ===============================================================================\n");
        if(my_core_id==0)printf("         ATTENTION ========> BAND is required to be 0 6 62 63, program will exit\n");
        if(my_core_id==0)printf(" ===============================================================================\n");
        return;
    }
    /* As I store V in the V vector there are overlap between
     * tasks so shift is now 4 where group need to be always
     * multiple of 2, because as example if grs=1 task 2 from
     * sweep 2 can run with task 6 sweep 1., but task 2 sweep 2
     * will overwrite the V of tasks 5 sweep 1 which are used by
     * task 6, so keep in mind that group need to be multiple of 2,
     * and thus tasks 2 sweep 2 will never run with task 6 sweep 1.
     * However, when storing V in A, shift could be back to 3.
     * */

    magma_zmalloc_cpu(&work, n);

    mycoresnb = cores_num;

    shift   = 5;
    if(grsiz==1)
        colblktile=1;
    else
        colblktile=grsiz/2;

    maxrequiredcores = nbtiles/colblktile;
    if(maxrequiredcores<1)maxrequiredcores=1;
    colpercore  = colblktile*nb;
    if(mycoresnb > maxrequiredcores)
    {
        if(my_core_id==0)printf("==================================================================================\n");
        if(my_core_id==0)printf("  WARNING only %3d threads are required to run this test optimizing cache reuse\n",maxrequiredcores);
        if(my_core_id==0)printf("==================================================================================\n");
        mycoresnb = maxrequiredcores;
    }
    thgrsiz = n;

    if(my_core_id==0) printf("  Static bulgechasing version v9_9col threads  %4d      N %5d      NB %5d    grs %4d thgrsiz %4d  BAND %4d\n",cores_num, n, nb, grsiz,thgrsiz,band);

    stepercol = magma_ceildiv(shift, grsiz);

    thgrnb  = magma_ceildiv(n-1, thgrsiz);

    for (thgrid = 1; thgrid<=thgrnb; thgrid++){
        stt  = (thgrid-1)*thgrsiz+1;
        thed = min( (stt + thgrsiz -1), (n-1));
        for (i = stt; i <= n-1; i++){
            ed=min(i,thed);
            if(stt>ed)break;
            for (m = 1; m <=stepercol; m++){
                st=stt;
                for (sweepid = st; sweepid <=ed; sweepid++){

                    for (k = 1; k <=grsiz; k++){
                        myid = (i-sweepid)*(stepercol*grsiz) +(m-1)*grsiz + k;
                        if(myid%2 ==0){
                            colpt      = (myid/2)*nb+1+sweepid-1;
                            stind      = colpt-nb+1;
                            edind      = min(colpt,n);
                            blklastind = colpt;
                            if(stind>=edind){
                                printf("ERROR---------> st>=ed  %d  %d \n\n",stind, edind);
                                exit(-10);
                            }
                        }else{
                            colpt      = ((myid+1)/2)*nb + 1 +sweepid -1 ;
                            stind      = colpt-nb+1;
                            edind      = min(colpt,n);
                            if( (stind>=edind-1) && (edind==n) )
                                blklastind=n;
                            else
                                blklastind=0;
                            if(stind>edind){
                                printf("ERROR---------> st>=ed  %d  %d \n\n",stind, edind);
                                exit(-10);
                            }
                        }

                        coreid = (stind/colpercore)%mycoresnb;

                        if(my_core_id==coreid)
                        {

                            fin=0;
                            while(fin==0)
                            {
                                if(myid==1)
                                {
                                    if( (prog[myid+shift-1]== (sweepid-1)) )
                                    {
                                        magma_ztrdtype1cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, stind, edind, sweepid, Vblksiz, work);

                                        fin=1;
                                        prog[myid]= sweepid;
                                        if(blklastind >= (n-1))
                                        {
                                            for (j = 1; j <= shift; j++)
                                                prog[myid+j]=sweepid;
                                        }
                                    } // END progress condition

                                }else{
                                    if( (prog[myid-1]==sweepid) && (prog[myid+shift-1]== (sweepid-1)) )
                                    {
                                        if(myid%2 == 0)
                                            magma_ztrdtype2cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, stind, edind, sweepid, Vblksiz, work);
                                        else
                                            magma_ztrdtype3cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, stind, edind, sweepid, Vblksiz, work);

                                        fin=1;
                                        prog[myid]= sweepid;
                                        if(blklastind >= (n-1))
                                        {
                                            for (j = 1; j <= shift+mycoresnb; j++)
                                                prog[myid+j]=sweepid;
                                        }
                                    } // END progress condition
                                } // END if myid==1
                            } // END while loop

                        } // END if my_core_id==coreid

                        if(blklastind >= (n-1))
                        {
                            stt=stt+1;
                            break;
                        }
                    }   // END for k=1:grsiz
                } // END for sweepid=st:ed
            } // END for m=1:stepercol
        } // END for i=1:n-1
    } // END for thgrid=1:thgrnb

    magma_free_cpu(work);

} // END FUNCTION
////////////////////////////////////////////////////////////////////////////////////////////////////

#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)   &(T[(m)])
static void tile_bulge_computeT_parallel(magma_int_t my_core_id, magma_int_t cores_num, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU,
                                         magmaDoubleComplex *T, magma_int_t ldt, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz)
{
    //%===========================
    //%   local variables
    //%===========================
    magma_int_t firstcolj;
    magma_int_t rownbm;
    magma_int_t st,ed,fst,vlen,vnb,colj;
    magma_int_t blkid,vpos,taupos,tpos;
    magma_int_t blkpercore, myid;

    if(n<=0)
        return ;
    magma_int_t blkcnt = magma_bulge_get_blkcnt(n, nb, Vblksiz);

    blkpercore = blkcnt/cores_num;

    magma_int_t nbGblk  = magma_ceildiv(n-1, Vblksiz);

    if(my_core_id==0) printf("  COMPUTE T parallel threads %d with  N %d   NB %d   Vblksiz %d \n",cores_num,n,nb,Vblksiz);

    for (magma_int_t bg = nbGblk; bg>0; bg--)
    {
        firstcolj = (bg-1)*Vblksiz + 1;
        rownbm    = magma_ceildiv(n-(firstcolj+1), nb);
        if(bg==nbGblk)
            rownbm    = magma_ceildiv(n-firstcolj, nb);  // last blk has size=1 used for complex to handle A(N,N-1)

        for (magma_int_t m = rownbm; m>0; m--)
        {
            vlen = 0;
            vnb  = 0;
            colj      = (bg-1)*Vblksiz; // for k=0;I compute the fst and then can remove it from the loop
            fst       = (rownbm -m)*nb+colj +1;
            for (magma_int_t k=0; k<Vblksiz; k++)
            {
                colj     = (bg-1)*Vblksiz + k;
                st       = (rownbm -m)*nb+colj +1;
                ed       = min(st+nb-1,n-1);
                if(st>ed)
                    break;
                if((st==ed)&&(colj!=n-2))
                    break;

                vlen=ed-fst+1;
                vnb=k+1;
            }
            colj     = (bg-1)*Vblksiz;
            magma_bulge_findVTAUTpos(n, nb, Vblksiz, colj, fst, ldv, ldt, &vpos, &taupos, &tpos, &blkid);
            myid = blkid/blkpercore;
            if(my_core_id==(myid%cores_num)){
                if((vlen>0)&&(vnb>0))
                    lapackf77_zlarft( "F", "C", &vlen, &vnb, V(vpos), &ldv, TAU(taupos), T(tpos), &ldt);
            }
        }
    }
}
#undef V
#undef TAU
#undef T
////////////////////////////////////////////////////////////////////////////////////////////////////

//##################################################################################################
static void *applyQ_parallel_section(void *arg)
{

    magma_int_t my_core_id   = ((applyQ_id_data*)arg) -> id;
    applyQ_data* data = ((applyQ_id_data*)arg) -> data;

    magma_int_t allcores_num   = data -> threads_num;
    magma_int_t n              = data -> n;
    magma_int_t ne             = data -> ne;
    magma_int_t n_gpu          = data -> n_gpu;
    magma_int_t nb             = data -> nb;
    magma_int_t Vblksiz        = data -> Vblksiz;
    magma_int_t wantz          = data -> wantz;
    magmaDoubleComplex *E         = data -> E;
    magma_int_t lde            = data -> lde;
    magmaDoubleComplex *V         = data -> V;
    magma_int_t ldv            = data -> ldv;
    magmaDoubleComplex *TAU       = data -> TAU;
    magmaDoubleComplex *T         = data -> T;
    magma_int_t ldt            = data -> ldt;
    magmaDoubleComplex *dE        = data -> dE;
    magma_int_t ldde           = data -> ldde;
    pthread_barrier_t* barrier = &(data -> barrier);

    magma_int_t info;

    real_Double_t timeQcpu=0.0, timeQgpu=0.0;

    magma_int_t n_cpu = ne - n_gpu;

    if(wantz<=0)
        return 0;
    // with MKL and when using omp_set_num_threads instead of mkl_set_num_threads
    // it need that all threads setting it to 1.
    magma_setlapack_numthreads(1);

#if defined(MAGMA_SETAFFINITY)
    cpu_set_t set;
    CPU_ZERO( &set );
    CPU_SET( my_core_id, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif


    /*################################################
     *   WANTZ == 2
     *################################################*/
    if((wantz==2))
    {

        if(my_core_id == 0)
        {
            //=============================================
            //   on GPU on thread 0:
            //    - apply V2*Z(:,1:N_GPU)
            //=============================================
            timeQgpu = magma_wtime();
            magma_zbulge_applyQ_v2('R', n, n, nb, Vblksiz, dE, ldde, V, ldv, T, ldt, &info);
            magma_device_sync();
            timeQgpu = magma_wtime()-timeQgpu;
            printf("  Finish Q2_GPU GGG timing= %f\n", timeQgpu);
            /* I am one of the remaining cores*/
        }else{
            //=============================================
            //   on CPU on threads 1:allcores_num-1:
            //    - apply V2*Z(:,N_GPU+1:N)
            //=============================================
            if(my_core_id == 1)timeQcpu = magma_wtime();

            magma_int_t n_loc = magma_ceildiv(n_cpu, allcores_num-1);
            magmaDoubleComplex* E_loc = E + n_gpu+ n_loc * (my_core_id-1);
            n_loc = min(n_loc,n_cpu - n_loc * (my_core_id-1));

            tile_bulge_applyQ(wantz, 'R', n_loc, n, nb, Vblksiz, E_loc, lde, V, ldv, TAU, T, ldt);
            pthread_barrier_wait(barrier);
            if(my_core_id == 1){
                timeQcpu = magma_wtime()-timeQcpu;
                printf("  Finish Q2_CPU CCC timing= %f\n", timeQcpu);
            }

        } // END if my_core_id

    }// END of WANTZ==2


    /*################################################
     *   WANTZ == 3 || WANTZ == 4
     *################################################*/
    if((wantz==3) || (wantz==4))
    {
            /* I am the last core in the new indexing and the original core=0 */
            if(my_core_id==0)
            {
                //=============================================
                //   on GPU on thread 0:
                //    - apply V2*Z(:,1:N_GPU)
                //=============================================
                timeQgpu = magma_wtime();
                magma_zsetmatrix(n, n_gpu, E, lde, dE, ldde);
                magma_zbulge_applyQ_v2('L', n_gpu, n, nb, Vblksiz, dE, ldde, V, ldv, T, ldt, &info);
                magma_device_sync();
                timeQgpu = magma_wtime()-timeQgpu;
                printf("  Finish Q2_GPU GGG timing= %f\n", timeQgpu);
                /* I am one of the remaining cores*/
            }else{
                //=============================================
                //   on CPU on threads 1:allcores_num-1:
                //    - apply V2*Z(:,N_GPU+1:N)
                //=============================================
                if(my_core_id == 1)
                    timeQcpu = magma_wtime();

                magma_int_t n_loc = magma_ceildiv(n_cpu, allcores_num-1);
                magmaDoubleComplex* E_loc = E + (n_gpu+ n_loc * (my_core_id-1))*lde;
                n_loc = min(n_loc,n_cpu - n_loc * (my_core_id-1));

                tile_bulge_applyQ(wantz, 'L', n_loc, n, nb, Vblksiz, E_loc, lde, V, ldv, TAU, T, ldt);
                pthread_barrier_wait(barrier);
                if(my_core_id == 1){
                    timeQcpu = magma_wtime()-timeQcpu;
                    printf("  Finish Q2_CPU CCC timing= %f\n", timeQcpu);
                }

            } // END if my_core_id

    }// END of WANTZ==3 / 4

#if defined(MAGMA_SETAFFINITY)
    // unbind threads
    magma_int_t sys_corenbr = 1;
    sys_corenbr = sysconf(_SC_NPROCESSORS_ONLN);
    CPU_ZERO( &set );
    for(magma_int_t i=0; i<sys_corenbr; i++)
        CPU_SET( i, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define E(m,n)   &(E[(m) + lde*(n)])
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)     &(T[(m)])
static void tile_bulge_applyQ(magma_int_t wantz, char side, magma_int_t n_loc, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
                              magmaDoubleComplex *E, magma_int_t lde, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magmaDoubleComplex *T, magma_int_t ldt)//, magma_int_t* info)
{
    //%===========================
    //%   local variables
    //%===========================
    magma_int_t firstcolj;
    magma_int_t bg, rownbm;
    magma_int_t st,ed,fst,vlen,vnb,colj;
    magma_int_t vpos,tpos;
    magma_int_t cur_blksiz,avai_blksiz, ncolinvolvd;
    magma_int_t nbgr, colst, coled;

    if(n<=0)
        return ;
    if(n_loc<=0)
        return ;

    //info = 0;
    magma_int_t INFO=0;

    magma_int_t nbGblk  = magma_ceildiv(n-1, Vblksiz);

    /* use colpercore =  N/cores_num; :if i want to split E into
     * cores_num chunk so I will have large chunk for each core.
     * However I prefer to split E into chunk of small size where
     * I guarantee that blas3 occur and the size of chunk remain into
     * cache, I think it is better. than I will have different chunk of
     * small sizeper coreand i will do each chunk till the end and
     * then move tothe second one for data locality
     *
     * version v1: for each chunck it apply all the V's then move to
     *                    the other chunck. the locality here inside each
     *                    chunck meaning that thread t apply V_k then move
     *                    to V_k+1 which overlap with V_k meaning that the
     *                    E_k+1 overlap with E_k. so here is the
     *                    locality however thread t had to read V_k+1 and
     *                    T_k+1 at each apply. note that all thread if they
     *                    run at same speed they might reading the same V_k
     *                    and T_k at the same time.
     * */

    magma_int_t nb_loc = 128; //$$$$$$$$

    magma_int_t     lwork = 2*nb_loc*max(Vblksiz,64);
    magmaDoubleComplex *work, *work2;

    magma_zmalloc_cpu(&work, lwork);
    magma_zmalloc_cpu(&work2, lwork);

    magma_int_t nbchunk =  magma_ceildiv(n_loc, nb_loc);

    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            each q_i consist of applying V to a block of row E(row_i,:) and applies are overlapped meaning
     *            that q_i+1 overlap a portion of the E(row_i, :).
     *            IN parallel E is splitten in vertical block over the threads  */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n
     *            each q_i consist of applying V to a block of col E(:, col_i,:) and the applies are overlapped meaning
     *            that q_i+1 overlap a portion of the E(:, col_i).
     *            IN parallel E is splitten in horizontal block over the threads  */

    printf("  APPLY Q2   N %d  N_loc %d  nbchunk %d  NB %d  Vblksiz %d  SIDE %c  WANTZ %d \n", n, n_loc, nbchunk, nb, Vblksiz, side, wantz);
    for (magma_int_t i = 0; i<nbchunk; i++)
    {
        magma_int_t ib_loc = min(nb_loc, (n_loc - i*nb_loc));

        if(side=='L'){
            for (bg = nbGblk; bg>0; bg--)
            {
                firstcolj = (bg-1)*Vblksiz + 1;
                rownbm    = magma_ceildiv((n-(firstcolj+1)),nb);
                if(bg==nbGblk) rownbm    = magma_ceildiv((n-(firstcolj)),nb);  // last blk has size=1 used for complex to handle A(N,N-1)
                for (magma_int_t j = rownbm; j>0; j--)
                {
                    vlen = 0;
                    vnb  = 0;
                    colj      = (bg-1)*Vblksiz; // for k=0;I compute the fst and then can remove it from the loop
                    fst       = (rownbm -j)*nb+colj +1;
                    for (magma_int_t k=0; k<Vblksiz; k++)
                    {
                        colj     = (bg-1)*Vblksiz + k;
                        st       = (rownbm -j)*nb+colj +1;
                        ed       = min(st+nb-1,n-1);
                        if(st>ed)
                            break;
                        if((st==ed)&&(colj!=n-2))
                            break;
                        vlen=ed-fst+1;
                        vnb=k+1;
                    }
                    colst     = (bg-1)*Vblksiz;
                    magma_bulge_findVTpos(n, nb, Vblksiz, colst, fst, ldv, ldt, &vpos, &tpos);

                    if((vlen>0)&&(vnb>0)){
                        lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &ib_loc, &vnb, V(vpos), &ldv, T(tpos), &ldt, E(fst,i*nb_loc), &lde, work, &ib_loc);
                    }
                    if(INFO!=0)
                        printf("ERROR ZUNMQR INFO %d \n",INFO);
                }
            }
        }else if (side=='R'){
            rownbm    = magma_ceildiv((n-1),nb);
            for (magma_int_t k = 1; k<=rownbm; k++)
            {
                ncolinvolvd = min(n-1, k*nb);
                avai_blksiz=min(Vblksiz,ncolinvolvd);
                nbgr = magma_ceildiv(ncolinvolvd,avai_blksiz);
                for (magma_int_t j = 1; j<=nbgr; j++)
                {
                    vlen = 0;
                    vnb  = 0;
                    cur_blksiz = min(ncolinvolvd-(j-1)*avai_blksiz, avai_blksiz);
                    colst = (j-1)*avai_blksiz;
                    coled = colst + cur_blksiz -1;
                    fst   = (rownbm -k)*nb+colst +1;
                    for (colj=colst; colj<=coled; colj++)
                    {
                        st       = (rownbm -k)*nb+colj +1;
                        ed       = min(st+nb-1,n-1);
                        if(st>ed)
                            break;
                        if((st==ed)&&(colj!=n-2))
                            break;
                        vlen=ed-fst+1;
                        vnb=vnb+1;
                    }
                    magma_bulge_findVTpos(n, nb, Vblksiz, colst, fst, ldv, ldt, &vpos, &tpos);
                    if((vlen>0)&&(vnb>0)){
                        lapackf77_zlarfb( "R", "N", "F", "C", &ib_loc, &vlen, &vnb, V(vpos), &ldv, T(tpos), &ldt, E(i*nb_loc,fst), &lde, work, &ib_loc);
                    }
                }
            }
        }else{
            printf("ERROR SIDE %d \n",side);
        }
    } // END loop over the chunks

    magma_free_cpu(work);
    magma_free_cpu(work2);

}
#undef E
#undef V
#undef TAU
#undef T
////////////////////////////////////////////////////////////////////////////////////////////////////

