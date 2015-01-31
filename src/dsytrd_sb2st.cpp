/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
       
       @author Azzam Haidar
       @author Stan Tomov
       @author Raffaele Solca

       @generated from zhetrd_hb2st.cpp normal z -> d, Fri Jan 30 19:00:18 2015

*/
#include "common_magma.h"
#include "magma_bulge.h"
#include "magma_dbulge.h"

#ifdef MAGMA_SETAFFINITY
#include "affinity.h"
#endif

#define PRECISION_d

static void *magma_dsytrd_sb2st_parallel_section(void *arg);

static void magma_dtile_bulge_parallel(
    magma_int_t my_core_id, magma_int_t cores_num,
    double *A, magma_int_t lda,
    double *V, magma_int_t ldv,
    double *TAU, magma_int_t n, magma_int_t nb, magma_int_t nbtiles,
    magma_int_t grsiz, magma_int_t Vblksiz, volatile magma_int_t *prog);

static void magma_dtile_bulge_computeT_parallel(
    magma_int_t my_core_id, magma_int_t cores_num,
    double *V, magma_int_t ldv, double *TAU,
    double *T, magma_int_t ldt,
    magma_int_t n, magma_int_t nb, magma_int_t Vblksiz);

//////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct magma_dbulge_data_s {
    magma_int_t threads_num;
    magma_int_t n;
    magma_int_t nb;
    magma_int_t nbtiles;
    magma_int_t grsiz;
    magma_int_t Vblksiz;
    magma_int_t compT;
    double* A;
    magma_int_t lda;
    double* V;
    magma_int_t ldv;
    double* TAU;
    double* T;
    magma_int_t ldt;
    volatile magma_int_t *prog;
    pthread_barrier_t barrier;
} magma_dbulge_data;

void magma_dbulge_data_init(
    magma_dbulge_data *dbulge_data_S,
    magma_int_t threads_num, magma_int_t n, magma_int_t nb, magma_int_t nbtiles,
    magma_int_t grsiz, magma_int_t Vblksiz, magma_int_t compT,
    double *A, magma_int_t lda,
    double *V, magma_int_t ldv, double *TAU,
    double *T, magma_int_t ldt,
    volatile magma_int_t* prog)
{
    dbulge_data_S->threads_num = threads_num;
    dbulge_data_S->n = n;
    dbulge_data_S->nb = nb;
    dbulge_data_S->nbtiles = nbtiles;
    dbulge_data_S->grsiz = grsiz;
    dbulge_data_S->Vblksiz = Vblksiz;
    dbulge_data_S->compT = compT;
    dbulge_data_S->A = A;
    dbulge_data_S->lda = lda;
    dbulge_data_S->V = V;
    dbulge_data_S->ldv= ldv;
    dbulge_data_S->TAU = TAU;
    dbulge_data_S->T = T;
    dbulge_data_S->ldt = ldt;
    dbulge_data_S->prog = prog;

    pthread_barrier_init(&(dbulge_data_S->barrier), NULL, dbulge_data_S->threads_num);
}
void magma_dbulge_data_destroy(magma_dbulge_data *dbulge_data_S)
{
    pthread_barrier_destroy(&(dbulge_data_S->barrier));
}
typedef struct magma_dbulge_id_data_s {
    magma_int_t id;
    magma_dbulge_data* data;
} magma_dbulge_id_data;

void magma_dbulge_id_data_init(magma_dbulge_id_data *id_data, magma_int_t id, magma_dbulge_data* data)
{
    id_data->id = id;
    id_data->data = data;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
    Purpose
    -------


    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangles of A is stored;
      -     = MagmaLower:  Lower triangles of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nb      INTEGER
            The order of the band matrix A.  N >= NB >= 0.

    @param[in]
    Vblksiz INTEGER
            The size of the block of householder vectors applied at once.

    @param[in]
    A       (workspace) DOUBLE_PRECISION array, dimension (LDA, N)
            On entry the band matrix stored in the following way:

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= 2*NB.

    @param[out]
    d       DOUBLE array, dimension (N)
            The diagonal elements of the tridiagonal matrix T:
            D(i) = A(i,i).

    @param[out]
    e       DOUBLE array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix T:
            E(i) = A(i,i+1) if UPLO = MagmaUpper, E(i) = A(i+1,i) if UPLO = MagmaLower.

    @param[out]
    V       DOUBLE_PRECISION array, dimension (BLKCNT, LDV, VBLKSIZ)
            On exit it contains the blocks of householder reflectors
            BLKCNT is the number of block and it is returned by the funtion MAGMA_BULGE_GET_BLKCNT.

    @param[in]
    ldv     INTEGER
            The leading dimension of V.
            LDV > NB + VBLKSIZ + 1

    @param[out]
    TAU     DOUBLE_PRECISION dimension(BLKCNT, VBLKSIZ)
            ???

    @param[in]
    compT   INTEGER
            if COMPT = 0 T is not computed
            if COMPT = 1 T is computed

    @param[out]
    T       DOUBLE_PRECISION dimension(LDT *)
            if COMPT = 1 on exit contains the matrices T needed for Q2
            if COMPT = 0 T is not referenced

    @param[in]
    ldt     INTEGER
            The leading dimension of T.
            LDT > Vblksiz

    @ingroup magma_dsyev_2stage
    ********************************************************************/
extern "C" magma_int_t
magma_dsytrd_sb2st(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
    double *A, magma_int_t lda, double *d, double *e,
    double *V, magma_int_t ldv, double *TAU,
    magma_int_t compT, double *T, magma_int_t ldt)
{
    #ifdef ENABLE_TIMER
    real_Double_t timeblg=0.0;
    #endif

    magma_int_t threads = magma_get_parallel_numthreads();
    magma_int_t mklth   = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);

    //const char* uplo_ = lapack_uplo_const( uplo );
    magma_int_t INgrsiz=1;
    magma_int_t blkcnt = magma_bulge_get_blkcnt(n, nb, Vblksiz);
    magma_int_t nbtiles = magma_ceildiv(n, nb);

    memset(T,   0, blkcnt*ldt*Vblksiz*sizeof(double));
    memset(TAU, 0, blkcnt*Vblksiz*sizeof(double));
    memset(V,   0, blkcnt*ldv*Vblksiz*sizeof(double));

    volatile magma_int_t* prog;
    magma_malloc_cpu((void**) &prog, (2*nbtiles+threads+10)*sizeof(magma_int_t));
    memset((void *) prog, 0, (2*nbtiles+threads+10)*sizeof(magma_int_t));

    magma_dbulge_id_data* arg;
    magma_malloc_cpu((void**) &arg, threads*sizeof(magma_dbulge_id_data));

    pthread_t* thread_id;
    magma_malloc_cpu((void**) &thread_id, threads*sizeof(pthread_t));
    pthread_attr_t thread_attr;

    magma_dbulge_data data_bulge;
    magma_dbulge_data_init(&data_bulge, threads, n, nb, nbtiles, INgrsiz, Vblksiz, compT,
                                 A, lda, V, ldv, TAU, T, ldt, prog);

    // Set one thread per core
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
    pthread_setconcurrency(threads);

    //timing
    #ifdef ENABLE_TIMER
    timeblg = magma_wtime();
    #endif

    // Launch threads
    for (magma_int_t thread = 1; thread < threads; thread++) {
        magma_dbulge_id_data_init(&(arg[thread]), thread, &data_bulge);
        pthread_create(&thread_id[thread], &thread_attr, magma_dsytrd_sb2st_parallel_section, &arg[thread]);
    }
    magma_dbulge_id_data_init(&(arg[0]), 0, &data_bulge);
    magma_dsytrd_sb2st_parallel_section(&arg[0]);

    // Wait for completion
    for (magma_int_t thread = 1; thread < threads; thread++) {
        void *exitcodep;
        pthread_join(thread_id[thread], &exitcodep);
    }

    // timing
    #ifdef ENABLE_TIMER
    timeblg = magma_wtime()-timeblg;
    printf("  time BULGE+T = %f\n", timeblg);
    #endif

    magma_free_cpu(thread_id);
    magma_free_cpu(arg);
    magma_free_cpu((void *) prog);
    magma_dbulge_data_destroy(&data_bulge);

    magma_set_lapack_numthreads(mklth);
    /*================================================
     *  store resulting diag and lower diag d and e
     *  note that d and e are always real
     *================================================*/

    /* Make diagonal and superdiagonal elements real,
     * storing them in d and e
     */
    /* In real case, the off diagonal element are
     * not necessary real. we have to make off-diagonal
     * elements real and copy them to e.
     * When using HouseHolder elimination,
     * the DLARFG give us a real as output so, all the
     * diagonal/off-diagonal element except the last one are already
     * real and thus we need only to take the abs of the last
     * one.
     *  */

#if defined(PRECISION_z) || defined(PRECISION_c)
    if (uplo == MagmaLower) {
        for (magma_int_t i=0; i < n-1; i++) {
            d[i] = MAGMA_D_REAL( A[i*lda  ] );
            e[i] = MAGMA_D_REAL( A[i*lda+1] );
        }
        d[n-1] = MAGMA_D_REAL(A[(n-1)*lda]);
    } else { /* MagmaUpper not tested yet */
        for (magma_int_t i=0; i < n-1; i++) {
            d[i] = MAGMA_D_REAL( A[i*lda+nb]   );
            e[i] = MAGMA_D_REAL( A[i*lda+nb-1] );
        }
        d[n-1] = MAGMA_D_REAL(A[(n-1)*lda+nb]);
    } /* end MagmaUpper */
#else
    if ( uplo == MagmaLower ) {
        for (magma_int_t i=0; i < n-1; i++) {
            d[i] = A[i*lda];   // diag
            e[i] = A[i*lda+1]; // lower diag
        }
        d[n-1] = A[(n-1)*lda];
    } else {
        for (magma_int_t i=0; i < n-1; i++) {
            d[i] = A[i*lda+nb];   // diag
            e[i] = A[i*lda+nb-1]; // lower diag
        }
        d[n-1] = A[(n-1)*lda+nb];
    }
#endif
    return MAGMA_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

static void *magma_dsytrd_sb2st_parallel_section(void *arg)
{
    magma_int_t my_core_id  = ((magma_dbulge_id_data*)arg) -> id;
    magma_dbulge_data* data = ((magma_dbulge_id_data*)arg) -> data;

    magma_int_t allcores_num   = data -> threads_num;
    magma_int_t n              = data -> n;
    magma_int_t nb             = data -> nb;
    magma_int_t nbtiles        = data -> nbtiles;
    magma_int_t grsiz          = data -> grsiz;
    magma_int_t Vblksiz        = data -> Vblksiz;
    magma_int_t compT          = data -> compT;
    double *A      = data -> A;
    magma_int_t lda            = data -> lda;
    double *V      = data -> V;
    magma_int_t ldv            = data -> ldv;
    double *TAU    = data -> TAU;
    double *T      = data -> T;
    magma_int_t ldt            = data -> ldt;
    volatile magma_int_t* prog = data -> prog;

    pthread_barrier_t* barrier = &(data -> barrier);

    //magma_int_t sys_corenbr    = 1;

    #ifdef ENABLE_TIMER
    real_Double_t timeB=0.0, timeT=0.0;
    #endif

    // with MKL and when using omp_set_num_threads instead of mkl_set_num_threads
    // it need that all threads setting it to 1.
    magma_set_lapack_numthreads(1);

#ifdef MAGMA_SETAFFINITY
//#define PRINTAFFINITY
#ifdef PRINTAFFINITY
    affinity_set print_set;
    print_set.print_affinity(my_core_id, "starting affinity");
#endif
    affinity_set original_set;
    affinity_set new_set(my_core_id);
    int check  = 0;
    int check2 = 0;
    // bind threads
    check = original_set.get_affinity();
    if (check == 0) {
        check2 = new_set.set_affinity();
        if (check2 != 0)
            printf("Error in sched_setaffinity (single cpu)\n");
    }
    else {
        printf("Error in sched_getaffinity\n");
    }
#ifdef PRINTAFFINITY
    print_set.print_affinity(my_core_id, "set affinity");
#endif
#endif

    if (compT == 1) {
        /* compute the Q1 overlapped with the bulge chasing+T.
         * if all_cores_num=1 it call Q1 on GPU and then bulgechasing.
         * otherwise the first thread run Q1 on GPU and
         * the other threads run the bulgechasing.
         * */

        if (allcores_num == 1) {
            //=========================
            //    bulge chasing
            //=========================
            #ifdef ENABLE_TIMER
            timeB = magma_wtime();
            #endif
            
            magma_dtile_bulge_parallel(0, 1, A, lda, V, ldv, TAU, n, nb, nbtiles, grsiz, Vblksiz, prog);

            #ifdef ENABLE_TIMER
            timeB = magma_wtime()-timeB;
            printf("  Finish BULGE   timing= %f\n", timeB);
            #endif
            //=========================
            // compute the T's to be used when applying Q2
            //=========================
            #ifdef ENABLE_TIMER
            timeT = magma_wtime();
            #endif

            magma_dtile_bulge_computeT_parallel(0, 1, V, ldv, TAU, T, ldt, n, nb, Vblksiz);

            #ifdef ENABLE_TIMER
            timeT = magma_wtime()-timeT;
            printf("  Finish T's     timing= %f\n", timeT);
            #endif
        } else { // allcore_num > 1
            magma_int_t id  = my_core_id;
            magma_int_t tot = allcores_num;


                //=========================
                //    bulge chasing
                //=========================
                #ifdef ENABLE_TIMER
                if (id == 0)
                    timeB = magma_wtime();
                #endif

                magma_dtile_bulge_parallel(id, tot, A, lda, V, ldv, TAU, n, nb, nbtiles, grsiz, Vblksiz, prog);
                pthread_barrier_wait(barrier);

                #ifdef ENABLE_TIMER
                if (id == 0) {
                    timeB = magma_wtime()-timeB;
                    printf("  Finish BULGE   timing= %f\n", timeB);
                }
                #endif

                //=========================
                // compute the T's to be used when applying Q2
                //=========================
                #ifdef ENABLE_TIMER
                if (id == 0)
                    timeT = magma_wtime();
                #endif

                magma_dtile_bulge_computeT_parallel(id, tot, V, ldv, TAU, T, ldt, n, nb, Vblksiz);
                pthread_barrier_wait(barrier);

                #ifdef ENABLE_TIMER
                if (id == 0) {
                    timeT = magma_wtime()-timeT;
                    printf("  Finish T's     timing= %f\n", timeT);
                }
                #endif
        } // allcore == 1
    } else { // WANTZ = 0
        //=========================
        //    bulge chasing
        //=========================
        #ifdef ENABLE_TIMER
        if (my_core_id == 0)
            timeB = magma_wtime();
        #endif

        magma_dtile_bulge_parallel(my_core_id, allcores_num, A, lda, V, ldv, TAU, n, nb, nbtiles, grsiz, Vblksiz, prog);
        pthread_barrier_wait(barrier);

        #ifdef ENABLE_TIMER
        if (my_core_id == 0) {
            timeB = magma_wtime()-timeB;
            printf("  Finish BULGE   timing= %f\n", timeB);
        }
        #endif
    } // WANTZ > 0

#ifdef MAGMA_SETAFFINITY
    // unbind threads
    if (check == 0) {
        check2 = original_set.set_affinity();
        if (check2 != 0)
            printf("Error in sched_setaffinity (restore cpu list)\n");
    }
#ifdef PRINTAFFINITY
    print_set.print_affinity(my_core_id, "restored_affinity");
#endif
#endif

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void magma_dtile_bulge_parallel(
    magma_int_t my_core_id, magma_int_t cores_num,
    double *A, magma_int_t lda,
    double *V, magma_int_t ldv,
    double *TAU, magma_int_t n, magma_int_t nb, magma_int_t nbtiles,
    magma_int_t grsiz, magma_int_t Vblksiz, volatile magma_int_t *prog)
{
    magma_int_t sweepid, myid, shift, stt, st, ed, stind, edind;
    magma_int_t blklastind, colpt;
    magma_int_t stepercol;
    magma_int_t i, j, m, k;
    magma_int_t thgrsiz, thgrnb, thgrid, thed;
    magma_int_t coreid;
    magma_int_t colblktile, maxrequiredcores, colpercore, mycoresnb;
    magma_int_t fin;
    double *work;

    if (n <= 0)
        return;
    if (grsiz <= 0)
        return;

    //printf("=================> my core id %d of %d \n",my_core_id, cores_num);

    /* As I store V in the V vector there are overlap between
     * tasks so shift is now 4 where group need to be always
     * multiple of 2, because as example if grs=1 task 2 from
     * sweep 2 can run with task 6 sweep 1., but task 2 sweep 2
     * will overwrite the V of tasks 5 sweep 1 which are used by
     * task 6, so keep in mind that group need to be multiple of 2,
     * and thus tasks 2 sweep 2 will never run with task 6 sweep 1.
     * However, when storing V in A, shift could be back to 3.
     * */

    magma_dmalloc_cpu(&work, n);
    mycoresnb = cores_num;

    shift   = 5;
    if (grsiz == 1)
        colblktile=1;
    else
        colblktile=grsiz/2;

    maxrequiredcores = nbtiles/colblktile;
    if (maxrequiredcores < 1)maxrequiredcores=1;
    colpercore  = colblktile*nb;
    if (mycoresnb > maxrequiredcores)
        mycoresnb = maxrequiredcores;
    thgrsiz = n;
    stepercol = magma_ceildiv(shift, grsiz);
    thgrnb  = magma_ceildiv(n-1, thgrsiz);

    #ifdef ENABLE_DEBUG
    if (my_core_id == 0) {
        if (cores_num > maxrequiredcores)    {
           printf("==================================================================================\n");
           printf("  WARNING only %3d threads are required to run this test optimizing cache reuse\n", maxrequiredcores);
           printf("==================================================================================\n");
        }
        printf("  Static bulgechasing version v9_9col threads  %4d      N %5d      NB %5d    grs %4d thgrsiz %4d \n", cores_num, n, nb, grsiz, thgrsiz);
    }
    #endif

    for (thgrid = 1; thgrid <= thgrnb; thgrid++) {
        stt  = (thgrid-1)*thgrsiz+1;
        thed = min( (stt + thgrsiz -1), (n-1));
        for (i = stt; i <= n-1; i++) {
            ed = min(i,thed);
            if (stt > ed) break;
            for (m = 1; m <= stepercol; m++) {
                st=stt;
                for (sweepid = st; sweepid <= ed; sweepid++) {
                    for (k = 1; k <= grsiz; k++) {
                        myid = (i-sweepid)*(stepercol*grsiz) +(m-1)*grsiz + k;
                        if (myid%2 == 0) {
                            colpt      = (myid/2)*nb+1+sweepid-1;
                            stind      = colpt-nb+1;
                            edind      = min(colpt,n);
                            blklastind = colpt;
                            if (stind >= edind) {
                                printf("ERROR---------> st >= ed  %d  %d \n\n", (int) stind, (int) edind);
                                exit(-10);
                            }
                        } else {
                            colpt      = ((myid+1)/2)*nb + 1 +sweepid -1;
                            stind      = colpt-nb+1;
                            edind      = min(colpt,n);
                            if ( (stind >= edind-1) && (edind == n) )
                                blklastind=n;
                            else
                                blklastind=0;
                            if (stind > edind) {
                                printf("ERROR---------> st >= ed  %d  %d \n\n", (int) stind, (int) edind);
                                exit(-10);
                            }
                        }

                        coreid = (stind/colpercore)%mycoresnb;

                        if (my_core_id == coreid) {
                            fin=0;
                            while(fin == 0) {
                                if (myid == 1) {
                                    if ( prog[myid+shift-1] == (sweepid-1) ) {
                                        magma_dtrdtype1cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, stind, edind, sweepid, Vblksiz, work);

                                        fin=1;
                                        prog[myid]= sweepid;
                                        if (blklastind >= (n-1)) {
                                            for (j = 1; j <= shift; j++)
                                                prog[myid+j]=sweepid;
                                        }
                                    } // END progress condition
                                } else {
                                    if ( (prog[myid-1] == sweepid) && (prog[myid+shift-1] == (sweepid-1)) ) {
                                        if (myid%2 == 0)
                                            magma_dtrdtype2cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, stind, edind, sweepid, Vblksiz, work);
                                        else
                                            magma_dtrdtype3cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, stind, edind, sweepid, Vblksiz, work);

                                        fin=1;
                                        prog[myid]= sweepid;
                                        if (blklastind >= (n-1)) {
                                            for (j = 1; j <= shift+mycoresnb; j++)
                                                prog[myid+j]=sweepid;
                                        }
                                    } // END progress condition
                                } // END if myid == 1
                            } // END while loop
                        } // END if my_core_id == coreid

                        if (blklastind >= (n-1)) {
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
static void magma_dtile_bulge_computeT_parallel(
    magma_int_t my_core_id, magma_int_t cores_num,
    double *V, magma_int_t ldv, double *TAU,
    double *T, magma_int_t ldt,
    magma_int_t n, magma_int_t nb, magma_int_t Vblksiz)
{
    //%===========================
    //%   local variables
    //%===========================
    magma_int_t Vm, Vn, mt, nt;
    magma_int_t myrow, mycol, blkj, blki, firstrow;
    magma_int_t blkid, vpos, taupos, tpos;
    magma_int_t blkpercore, myid;

    if (n <= 0)
        return;

    magma_int_t blkcnt = magma_bulge_get_blkcnt(n, nb, Vblksiz);
    blkpercore = blkcnt/cores_num;
    blkpercore = (blkpercore == 0 ? 1 : blkpercore);
    //magma_int_t nbGblk  = magma_ceildiv(n-1, Vblksiz);

    #ifdef ENABLE_DEBUG
    if (my_core_id == 0)
        printf("  COMPUTE T parallel threads %d with  N %d   NB %d   Vblksiz %d \n", cores_num, n, nb, Vblksiz);
    #endif



    /*========================================
     * compute the T's in parallel.
     * The Ts are independent so each core pick
     * a T and compute it. The loop is based on
     * the version 113 of the applyQ
     * which go over the losange block_column
     * by block column. but it is not important
     * here the order because Ts are independent.
     * ======================================== */
    nt  = magma_ceildiv((n-1), Vblksiz);
    for (blkj=nt-1; blkj >= 0; blkj--) {
        /* the index of the first row on the top of block (blkj) */
        firstrow = blkj * Vblksiz + 1;
        /*find the number of tile for this block */
        if ( blkj == nt-1 )
            mt = magma_ceildiv( n -  firstrow,    nb);
        else
            mt = magma_ceildiv( n - (firstrow+1), nb);
        /*loop over the tiles find the size of the Vs and apply it */
        for (blki=mt; blki > 0; blki--) {
            /*calculate the size of each losange of Vs= (Vm,Vn)*/
            myrow     = firstrow + (mt-blki)*nb;
            mycol     = blkj*Vblksiz;
            Vm = min( nb+Vblksiz-1, n-myrow);
            if ( ( blkj == nt-1 ) && ( blki == mt ) ) {
                Vn = min (Vblksiz, Vm);
            } else {
                Vn = min (Vblksiz, Vm-1);
            }
            /*calculate the pointer to the Vs and the Ts.
             * Note that Vs and Ts have special storage done
             * by the bulgechasing function*/
            magma_bulge_findVTAUTpos(n, nb, Vblksiz, mycol, myrow, ldv, ldt, &vpos, &taupos, &tpos, &blkid);
            myid = blkid/blkpercore;
            if ( my_core_id == (myid%cores_num) ) {
                if ( ( Vm > 0 ) && ( Vn > 0 ) ) {
                    lapackf77_dlarft( "F", "C", &Vm, &Vn, V(vpos), &ldv, TAU(taupos), T(tpos), &ldt);
                }
            }
        }
    }
}
#undef V
#undef TAU
#undef T
////////////////////////////////////////////////////////////////////////////////////////////////////
