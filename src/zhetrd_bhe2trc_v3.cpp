/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Azzam Haidar
       @author Stan Tomov

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "magma_zbulgeinc.h"

#define PRECISION_z

#define CHECK 0
#define LOGQ 0
#define LOG 0

#if defined(PRECISION_d)
extern "C" void    mydlarft_(const char *direct, const char *storev, magma_int_t *n, magma_int_t *k, 
                         double *v, magma_int_t *ldv, const double *tau, 
                         double *t, magma_int_t *ldt);
#endif

#if defined(PRECISION_z) || defined(PRECISION_d)
extern "C" void cmp_vals(magma_int_t n, double *wr1, double *wr2, double *nrmI, double *nrm1, double *nrm2);
extern "C" void zcheck_eig_(char *JOBZ, magma_int_t  *MATYPE, magma_int_t  *N, magma_int_t  *NB,
                       magmaDoubleComplex* A, magma_int_t  *LDA, double *AD, double *AE, double *D1, double *EIG,
                    magmaDoubleComplex *Z, magma_int_t  *LDZ, magmaDoubleComplex *WORK, double *RWORK, double *RESU);
#endif


extern "C"  magma_int_t plasma_ceildiv(magma_int_t a, magma_int_t b);
extern "C"  void tile_bulge_applyQ_cpu(magma_int_t WANTZ, char SIDE, magma_int_t N, magma_int_t NB, magma_int_t Vblksiz, magmaDoubleComplex *E, magma_int_t LDE, magmaDoubleComplex *V, magmaDoubleComplex *TAU, magmaDoubleComplex *T, magma_int_t *INFO);

static void barrier(int my_core_id, int cores_num);
static void *parallel_section(void *thread_id);
static void *applyQ_parallel_section(void *thread_id);
static void tile_bulge_parallel(int my_core_id);
static void tile_bulge_computeT_parallel(int my_core_id);
static void tile_bulge_applyQ_parallel(int my_core_id);
static void tile_bulge_applyQ_parallel2(int my_core_id);

/*
extern "C" void TRD_hbcelr_v62sym_withQ(int N, int NB, magmaDoubleComplex *A, int LDA, magmaDoubleComplex *V, magmaDoubleComplex *TAU, int st, int ed, int myid, int sweep, int Vblksiz);
extern "C" void TRD_type1cHLsym_withQ(int N, int NB, magmaDoubleComplex *A, int LDA, magmaDoubleComplex *V, magmaDoubleComplex *TAU, int st, int ed, int sweep, int Vblksiz);
extern "C" void TRD_type2cHLsym_withQ(int N, int NB, magmaDoubleComplex *A, int LDA, magmaDoubleComplex *V, magmaDoubleComplex *TAU, int st, int ed, int sweep, int Vblksiz);
extern "C" void TRD_type3cHLsym_withQ(int N, int NB, magmaDoubleComplex *A, int LDA, magmaDoubleComplex *V, magmaDoubleComplex *TAU, int st, int ed, int sweep, int Vblksiz);
*/
/*
#define magma_ztrdtype1cbHLsym_withQ                 magma_dtrdtype1cbHLsym_withQ
#define magma_ztrdtype2cbHLsym_withQ                 magma_dtrdtype2cbHLsym_withQ
#define magma_ztrdtype3cbHLsym_withQ                magma_dtrdtype3cbHLsym_withQ
#define magma_zbulge_applyQ                           magma_dbulge_applyQ
#define magma_zstedc_withZ                        magma_dstedc_withZ
*/

extern "C" void magma_ztrdtype1cbHLsym_withQ(magma_int_t N, magma_int_t NB, magmaDoubleComplex *A, magma_int_t LDA, magmaDoubleComplex *V, magmaDoubleComplex *TAU, magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz);
extern "C" void magma_ztrdtype2cbHLsym_withQ(magma_int_t N, magma_int_t NB, magmaDoubleComplex *A, magma_int_t LDA, magmaDoubleComplex *V, magmaDoubleComplex *TAU, magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz);
extern "C" void magma_ztrdtype3cbHLsym_withQ(magma_int_t N, magma_int_t NB, magmaDoubleComplex *A, magma_int_t LDA, magmaDoubleComplex *V, magmaDoubleComplex *TAU, magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz);
extern "C" void magma_zbulge_applyQ(magma_int_t WANTZ, char SIDE, magma_int_t NE, magma_int_t N, magma_int_t NB, magma_int_t Vblksiz, magmaDoubleComplex *E, magma_int_t LDE, magmaDoubleComplex *V, magmaDoubleComplex *TAU, magmaDoubleComplex *T, magma_int_t *INFO, magmaDoubleComplex *dV, magmaDoubleComplex *dT, magmaDoubleComplex *dE, magma_int_t copytype );
extern "C" void magma_zstedc_withZ(char JOBZ, magma_int_t N, double *D, double * E, magmaDoubleComplex *Z, magma_int_t LDZ);
extern "C" void magma_zstedx_withZ(magma_int_t N, magma_int_t NE, double *D, double * E, magmaDoubleComplex *Z, magma_int_t LDZ);


extern "C" magma_int_t
magma_zungqr_2stage_gpu(magma_int_t m, magma_int_t n, magma_int_t k,
                 magmaDoubleComplex *da, magma_int_t ldda,
                 magmaDoubleComplex *tau, magmaDoubleComplex *dT,
                 magma_int_t nb, magma_int_t *info);

         

////////////////////////////////////////////////////////////////////////////////////////////////////          
// define globals ONLY in the z file, and declare them extern in [sdc] files.
#if defined(PRECISION_z)
    volatile int barrier_in[MAX_THREADS_BLG];
    volatile int barrier_out[MAX_THREADS_BLG];
    volatile int *ss_prog;
    // rest are declared extern in [sdcz]bulge_inc.h header; define them here.
    struct gbstrct_blg core_in_all;
    int           event_numblg        [MAX_THREADS_BLG]                 __attribute__ ((aligned (128)));
    real_Double_t event_start_timeblg [MAX_THREADS_BLG]                 __attribute__ ((aligned (128)));
    real_Double_t event_end_timeblg   [MAX_THREADS_BLG]                 __attribute__ ((aligned (128)));
    real_Double_t event_logblg        [MAX_THREADS_BLG][MAX_EVENTSBLG]  __attribute__ ((aligned (128)));
    int           log_eventsblg;
#else
    extern volatile int barrier_in[MAX_THREADS_BLG];
    extern volatile int barrier_out[MAX_THREADS_BLG];
    extern volatile int *ss_prog;
#endif

static void barrier(int my_core_id, int cores_num);
static void barrier(int my_core_id, int cores_num)
{
    int core;
    
    if (my_core_id == 0)    {
        for (core = 1; core < cores_num; core++)
            while (barrier_in[core] == 0);

        for (core = 1; core < cores_num; core++)
            barrier_in[core] = 0;

        for (core = 1; core < cores_num; core++)
            barrier_out[core] = 1;
    }
    else
    {
        barrier_in[my_core_id] = 1;
        while (barrier_out[my_core_id] == 0);
        barrier_out[my_core_id] = 0;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////          
/* START CODE */
extern "C" magma_int_t magma_zhetrd_bhe2trc( magma_int_t THREADS, magma_int_t WANTZ, char uplo, magma_int_t NE, magma_int_t N, magma_int_t NB, magmaDoubleComplex *A1, magma_int_t LDA1, double *D2, double *E2, magmaDoubleComplex *dT1, magma_int_t LDT1)
{
    char uplo_[2] = {uplo, 0};
    real_Double_t timelpk=0.0,tconvert=0.0,timeaplQ1=0.0,timeaplQ2=0.0,timeblg=0.0, timeaplQ=0.0, timeeigen=0.0, timegemm=0.0;
    double nrmI=0.0, nrm1=0.0, nrm2=0.0;
    FILE *trace_file;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    
    magma_int_t mklth, thread, INFO;
    magma_int_t Vblksiz=-1, blkcnt=-1, LDV=-1, LDT =-1, INgrsiz=1, LDE=-1, BAND=6;
#if (defined(PRECISION_s) || defined(PRECISION_d))
    Vblksiz = min(NB,48);
#else
    Vblksiz = min(NB,32);
#endif
    LDT     = Vblksiz;
    findVTsiz(N, NB, Vblksiz, &blkcnt, &LDV);

    magma_int_t i,j;
    magma_int_t NBTILES, LDA2;
    i = N/NB;
    NBTILES = i*NB==N ? i:i+1;

    magma_int_t overlapQ1   = 1;
    magma_int_t usemulticpu = 1;
    magma_int_t N_CPU=0,N_GPU=N;

    //if(N<4000)
     //       usemulticpu =0;

    if((WANTZ!=3)&(WANTZ!=2))
           usemulticpu =0;

    core_in_all.usemulticpu = usemulticpu;
    core_in_all.overlapQ1   = overlapQ1;
    /************************************************* 
     *     INITIALIZING MATRICES 
     * ***********************************************/
    /* A1 is equal to A2 */
    LDA2 = NB+1+(NB-1);
    if(LDA2>N){
            printf("error matrix too small N=%d NB=%d\n", (int) N, (int) NB);
            return -14;
    }
    if((N<=0)||(NB<=0)){
            printf("error N or NB <0 N=%d NB=%d\n", (int) N, (int) NB);
            return -14;
    }


/*

       trace_file = fopen("AJETE/Abefore", "w");
    for (j = 0; j < (N-NB); j++)
    {
        for (i = 0; i < NB+1; i++)
        {
               fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+j+1,j+1,MAGMA_Z_REAL(A1[i+j*LDA1+j]) ,  MAGMA_Z_IMAG(A1[i+j*LDA1+j])  );
        }
    }        
    for (j = 0; j < NB; j++)
    {
        for (i = 0; i < NB-j; i++)
        {
               fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+(N-NB+j)+1,N-NB+j+1,MAGMA_Z_REAL(A1[i+(N-NB+j)*LDA1+(N-NB+j)]) ,  MAGMA_Z_IMAG(A1[i+(N-NB+j)*LDA1+(N-NB+j)])  );
        }
    }


       fclose(trace_file);
*/





    timelpk = magma_wtime();
    /* copy the input matrix into a matrix A2 with band storage */
    magmaDoubleComplex *A2    = (magmaDoubleComplex *) malloc (N*LDA2*sizeof(magmaDoubleComplex));
    memset(A2 , 0, N*LDA2*sizeof(magmaDoubleComplex));        
    for (j = 0; j < (N-NB); j++)
    {
        for (i = 0; i < NB+1; i++)
        {
            A2[i+j*LDA2]   = A1[i+j*LDA1+j];
            A1[i+j*LDA1+j] = c_zero;
        }
    }        
    for (j = 0; j < NB; j++)
    {
        for (i = 0; i < NB-j; i++)
        {
            A2[i+(N-NB+j)*LDA2] = A1[i+(N-NB+j)*LDA1+(N-NB+j)]; 
            A1[i+(N-NB+j)*LDA1+(N-NB+j)] = c_zero;
        }
    }

    for (j = 0; j < N-NB; j++)
       A1[NB+j*LDA1+j] = c_one;

    timelpk = magma_wtime() - timelpk;
    printf("  Finish CONVERT timing= %f\n", timelpk); 

    

    /************************************************* 
     *     compute Q1 
     * ***********************************************/
    magmaDoubleComplex *da, *NOTUSED;
    if((WANTZ>0)&(WANTZ!=4)){
       if (MAGMA_SUCCESS != magma_zmalloc( &da, N*LDA1 )) {
           //*info = MAGMA_ERR_DEVICE_ALLOC;
           return MAGMA_ERR_DEVICE_ALLOC;
       }
        
       magma_device_sync();
       magma_zsetmatrix( N, LDA1, A1, LDA1, da, LDA1 );
       if(overlapQ1==0){
           timeaplQ1 = magma_wtime();
           magma_zungqr_2stage_gpu(N, N, N, da, LDA1, NOTUSED, dT1, NB, &INFO);
           magma_device_sync();
           //cublasGetMatrix( N, LDA1, sizeof(magmaDoubleComplex), da, LDA1, A1, LDA1);
           timeaplQ1 = magma_wtime()-timeaplQ1;
           printf("  Finish applyQ1 timing= %f\n", timeaplQ1); 
       }
       /*            
       trace_file = fopen("AJETE/Q1", "w");
       for (j = 0; j < N ; j++) 
             for (i = 0; i < N ; i++) 
                        //fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,A1[j*LDA1+i]);
                         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+1,j+1,MAGMA_Z_REAL(A1[j*LDA1+i]) ,  MAGMA_Z_IMAG(A1[j*LDA1+i])  );
       fclose(trace_file);
       */
    }
    /***********************************************/
    

    /************************************************* 
     *    For local check use
     * ***********************************************/
    char JOBZ;
    double *D1;
    magmaDoubleComplex *AINIT;
    magma_int_t MM,NN,LDAINIT;
    if(CHECK)
    {
        MM=NB+1;
        NN=N;
        LDAINIT=NB+1;
        AINIT = (magmaDoubleComplex *) malloc(MM*NN*sizeof(magmaDoubleComplex) );
        memset(AINIT , 0, MM*NN*sizeof(magmaDoubleComplex));        
        lapackf77_zlacpy("A", &MM, &NN, A2, &LDA2, AINIT, &LDAINIT );
    }
    /***********************************************/


    magmaDoubleComplex *T     = (magmaDoubleComplex *) malloc (blkcnt*LDT*Vblksiz*sizeof(magmaDoubleComplex));
    magmaDoubleComplex *TAU   = (magmaDoubleComplex *) malloc (blkcnt*Vblksiz*sizeof(magmaDoubleComplex));
    magmaDoubleComplex *V     = (magmaDoubleComplex *) malloc (blkcnt*LDV*Vblksiz*sizeof(magmaDoubleComplex));
    memset(T,   0, blkcnt*LDT*Vblksiz*sizeof(magmaDoubleComplex));        
    memset(TAU, 0, blkcnt*Vblksiz*sizeof(magmaDoubleComplex));        
    memset(V,   0, blkcnt*LDV*Vblksiz*sizeof(magmaDoubleComplex));        





    // Set input parameters
    //    ssched_init(THREADS);
    ss_prog = (int*) malloc((2*NBTILES+THREADS+10)*sizeof(int));
    int   iamdone[MAX_THREADS_BLG]; 
    int   thread_num[MAX_THREADS_BLG];
    pthread_t      thread_id[MAX_THREADS_BLG];
    pthread_attr_t thread_attr;

    //goto ed;
    
    magma_setlapack_numthreads(1);
    core_in_all.cores_num = THREADS;
    core_in_all.dQ1       = da;
    core_in_all.dT1       = dT1;
    core_in_all.T         = T;
    core_in_all.A         = A2;
    core_in_all.TAU       = TAU;
    core_in_all.V         = V;
    core_in_all.NB        = NB;
    core_in_all.NBTILES   = NBTILES;
    core_in_all.N         = N;
    core_in_all.LDA       = LDA2;
    core_in_all.BAND      = BAND;
    core_in_all.grsiz     = INgrsiz;
    core_in_all.timeblg   = &timeblg;//0.0;
    core_in_all.timeaplQ  = &timeaplQ;//0.0;
    core_in_all.ss_prog   = ss_prog;
    core_in_all.WANTZ     = WANTZ;
    core_in_all.SIDE      = 'R';
    core_in_all.E         = A1;
    core_in_all.E_CPU     = A1;
    core_in_all.LDE       = LDA1;
    core_in_all.Vblksiz   = Vblksiz;

    if((overlapQ1==1)&&(THREADS>1)&&(WANTZ>0)){
       core_in_all.locores_num = THREADS-1;
    }else{
       core_in_all.locores_num = THREADS;
    }


    // Set one thread per core
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
    pthread_setconcurrency(THREADS);

    // Initializations
    for (thread = 0; thread < THREADS; thread++)
    {
        barrier_in[thread] = 0;
        barrier_out[thread] = 0;
        event_numblg[thread] = 0;
    }

   for (i = 0; i < 2*NBTILES+THREADS+10; i++)
       ss_prog[i] = 0;


    for (i = 0; i < THREADS; i++)
       iamdone[i] = 0;



    // Launch threads
    for (thread = 1; thread < THREADS; thread++)
    {
        thread_num[thread] = thread;
        pthread_create(&thread_id[thread], &thread_attr, parallel_section, &thread_num[thread]);
    }
    thread_num[0] = 0;
    parallel_section(&thread_num[0]);

    // Wait for completion
    for (thread = 1; thread < THREADS; thread++)
    {
        void *exitcodep;
        pthread_join(thread_id[thread], &exitcodep);
    }
    printf("  Finish BULGE+T timing= %f\n", *(core_in_all.timeblg));

    /*================================================
     *  store resulting diag and lower diag D2 and E2
     *  note that D and E are always real
     *================================================*/  
    /*
     * STORE THE RESULTING diagonal/off-diagonal in D AND E
     */
     memset(D2, 0,  N   *sizeof(double));
     memset(E2, 0, (N-1)*sizeof(double));
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


/*
       trace_file = fopen("AJETE/T", "w");
       for (j = 0; j < N-1 ; j++) {
         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",j+1,j+1,MAGMA_Z_REAL(A2[j*LDA2]) ,  MAGMA_Z_IMAG(A2[j*LDA2])  );
         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",j+2,j+1,MAGMA_Z_REAL(A2[j*LDA2+1]) ,  MAGMA_Z_IMAG(A2[j*LDA2+1])  );
       }
         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",N,N,MAGMA_Z_REAL(A2[(N-1)*LDA2]) ,  MAGMA_Z_IMAG(A2[(N-1)*LDA2])  );

       fclose(trace_file);
*/





#if defined(PRECISION_z) || defined(PRECISION_c)  
    if(uplo==MagmaLower){
       for (i=0; i < N-1 ; i++)
       {
          D2[i] = MAGMA_Z_REAL( A2[i*LDA2]);               
          E2[i] = MAGMA_Z_REAL(A2[i*LDA2+1]);
       }
       D2[N-1] = MAGMA_Z_REAL(A2[(N-1)*LDA2]);
    } else { /* MagmaUpper not tested yet */
        for (i=0; i<N-1; i++)
        {
            D2[i]  =  MAGMA_Z_REAL(A2[i*LDA2+NB]);               
            E2[i] = MAGMA_Z_REAL(A2[i*LDA2+NB-1]);
        }
        D2[N-1] = MAGMA_Z_REAL(A2[(N-1)*LDA2+NB]);
    } /* end MagmaUpper */
#else
    if( uplo == MagmaLower ){
        for (i=0; i < N-1; i++) {
            D2[i] = A2[i*LDA2];   // diag
            E2[i] = A2[i*LDA2+1]; //lower diag  
        }
        D2[N-1] = A2[(N-1)*LDA2];
    } else {
        for (i=0; i < N-1; i++) {
            D2[i] = A2[i*LDA2+NB];   // diag
            E2[i] = A2[i*LDA2+NB-1]; //lower diag  
        }
        D2[N-1] = A2[(N-1)*LDA2+NB];
    }
#endif






    /* APPLY V2 generated by the bulge to the matrix Q1 of the first stage Q2=Q2*(I-V_2*T_2*V_2') */
    magmaDoubleComplex *dV2, *dQ2, *dT2, *dZ, *Q2, *Z;
    int dVsize, LDQ2=N,LDZ=N;
    int parallel=0;


    if((usemulticpu==1)&&(THREADS>1)){
       core_in_all.locores_num = THREADS-1;
    }else{
       core_in_all.locores_num = THREADS;
    }

    //========================
    //  WANTZ =1 : compute Q1 and Q2 and the eigenvectors Z 
    //             then make 2 GEMM's Q1*Q2*Z 
    //  WANTZ =2 : compute Q1 then apply V2 to Q1 from right 
    //             generating the global Q, then compute
    //             eigenvectors Z and make GEMM with Z==> Q*Z on GPU
    //             WANTZ=5 is similar, where the GEMM is done 
    //             implicitly during the eigenvectors computation. 
    //             assumed to be bad in perf.
    //  WANTZ =3 : compute Q1, then compute the eigenvectors Z,
    //             then apply V2 to the Z from Left, then make 
    //             GEMM with Q1 ==> Q1 * resulting 
    //  WANTZ =4 : So compute the eigenvectors Z, 
    //             then apply V2, then apply V1 to the result. 
    //             similar to WANTZ=3 but does not compute Q1 
    //             instead it apply V1 (I guess to be good for 
    //             multicore because it avoid computing Q1, while 
    //             on GPU's Q1 is computed overlaped with the 
    //             bulgechasing so for free so it is better to do
    //             a GEMM with Q1 (WANTZ=3) then to do the apply).
    //  WANTZ =5 : compute Q1 then apply V2 to Q1 from right 
    //             generating the global Q, then compute
    //             eigenvectors Z with option "V" that make 
    //             GEMM with Z==> Q*Z implicitly during the 
    //             eigenvectors computation assumed to be bad in perf
    //             because a GEMM on GPU or optimized 
    //             for multithread will be faster. 
    //             Similar to WANTZ=2 where the GEMM is done out of the eigensolver
    //                 
    //========================
    
    timeeigen = magma_wtime();
    if(WANTZ==0){
        timelpk = magma_wtime();
        // compute the eigenvalues using lapack routine to be able to compare to it and used as ref 
        magma_setlapack_numthreads(THREADS);

        // call eigensolver for our resulting tridiag [D E] and form E=Q*Z
        magma_zstedc_withZ(MagmaNoVec, N, D2, E2, Z, LDZ);
        magma_setlapack_numthreads(1);
        timelpk = magma_wtime()-timelpk;
        printf("  Finish WANTZ %d  eigensolver 'N'    timing= %f  threads %d\n", (int) WANTZ, timelpk, (int) i);
        /*
        for(i=0;i<10;i++)
                printf(" voici D[%d] %e\n",i,D2[i]);*/
    }
    if(WANTZ>0){
        if(WANTZ==1){
            // compute Q2 by applying V2 on the Identity        
            LDZ=LDA1;        
            Z    = (magmaDoubleComplex *) malloc (N*N*sizeof(magmaDoubleComplex));
            memset(Z , 0, N*N*sizeof(magmaDoubleComplex));
            if(parallel==1){
                Q2    = (magmaDoubleComplex *) malloc (N*N*sizeof(magmaDoubleComplex));
                // no need to set the matrix to identity if we are using the GPU because it is done inside bulge_applyQ
                memset(Q2 , 0, N*N*sizeof(magmaDoubleComplex));        
                for (j = 0; j < N; j++)
                    Q2[j+j*LDQ2] = c_one;
            }
            core_in_all.SIDE      = 'L';
            core_in_all.E         = Q2;
            core_in_all.LDE       = LDQ2;
        }else if(WANTZ==2){
            // compute the Q by applying V2 to Q1 that has been computed 
            // from the Right. Q= Q1 * ( I-V2*T2*V2') = Q1*Q2        
            LDZ=LDA1;        
            Z    = (magmaDoubleComplex *) malloc (N*N*sizeof(magmaDoubleComplex));
            memset(Z , 0, N*N*sizeof(magmaDoubleComplex));
            core_in_all.SIDE      = 'R';
            core_in_all.E         = A1;
            core_in_all.LDE       = LDA1;
        }else if((WANTZ==3)||(WANTZ==4)){
            // wait till the Eigenvector are computed and the apply V2 from the Left, 
            // and then make gemm with Q1 (WANTZ=3) or apply V1 to the result (WANTZ=4). 
            LDZ=LDA1;        
            Z    = (magmaDoubleComplex *) malloc (N*N*sizeof(magmaDoubleComplex));
            memset(Z , 0, N*N*sizeof(magmaDoubleComplex));
            // variable for Q2
            core_in_all.SIDE      = 'L';
            core_in_all.E         = Z;
            core_in_all.LDE       = LDZ;
            if(WANTZ==4) {
                    printf("the implementation is not finished yet. code will exit\n");
                    exit(-1);
            }
        }


        if((WANTZ==1)||(WANTZ==2)||(WANTZ==3)||(WANTZ==4)){
            timelpk = magma_wtime();
            // compute the eigenvalues using lapack routine to be able to compare to it and used as ref 
            magma_setlapack_numthreads(THREADS);
            // call eigensolver for our resulting tridiag [D E] and form E=Q*Z
            //magma_zstedc_withZ('I', N, D2, E2, Z, LDZ);
            magma_zstedx_withZ(N, NE, D2, E2, Z, LDZ);
            magma_setlapack_numthreads(1);
            timelpk = magma_wtime()-timelpk;
        }

       /*
        trace_file = fopen("AJETE/Z", "w");
       for (j = 0; j < N ; j++) 
             for (i = 0; i < N ; i++) 
                        //fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,A1[j*LDA1+i]);
                         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+1,j+1,MAGMA_Z_REAL(Z[j*LDA1+i]) ,  MAGMA_Z_IMAG(Z[j*LDA1+i])  );
       fclose(trace_file);
       */



        // ************************************************
        // use GPU code to apply V2 or compute Q2
        // ************************************************
        if(parallel==0){
           // allocate space on GPU for dV2 and dT2        
           dVsize = max(N*N,blkcnt*LDV*Vblksiz);
           //printf("dvsize %f\n",(16.0*(real_Double_t)dVsize)*1e-9);
           if(MAGMA_SUCCESS != magma_zmalloc( &dV2, dVsize )) { 
               printf ("!!!! magma_alloc failed for: dV2\n" );       
               exit(-1);                                                           
           }
    
           if(MAGMA_SUCCESS != magma_zmalloc( &dT2, blkcnt*LDT*Vblksiz )) { 
              printf ("!!!! magma_alloc failed for: dT2\n" );       
              exit(-1);                                                           
           }

           //========================
           //  WANTZ =1 
           //  compute Q1  done
           //  compute Q2  here
           //  compute Z   done
           //  make 2 GEMM here
           //========================
           if(WANTZ==1){
               // to avoid allocating a 4 matrix on GPU there are 2 way:
               //   way 1: make Q = Q1*Q2 ==> dV2 = da * dZ
               //          then copy Z to GPU put it on dZ
               //          make Q*Z ==> da = dV2 * dZ
               //          copy the result to CPU, da-->A1                 
               //
               //   way 2: in case we need only 20% of Z so it is better
               //          to start multiplying by Q2*Z, thus to avoid a matrix on GPU
               //          I could copy Q1 (from da) back to CPU and later re-copy to GPU
               //          copy Q1 to CPU, so copy da --> A1
               //          copy Z to GPU, Z --> dZ
               //          make Q = Q2*Z ==> dV2 = da * dZ
               //          copy Q1 to GPU, A1 --> dZ
               //          make Q1*Q ==> da = dZ *dV2
               //          copy the result to CPU, da-->A1  
               //
               // way 2 is implemented because of Raffaele
               // NE is the number of eigenvectors we want.
               timeaplQ2 = magma_wtime();
               magma_free( dT1 );
               // copy Q1 to CPU
               magma_zgetmatrix( N, LDA1, da, N, A1, LDA1 );
               // compute Q2 by applying V2 to Identity and put it into da           
               magma_zbulge_applyQ(WANTZ, 'L', NE, N, NB, Vblksiz, Q2, N, V, TAU, T, &INFO, dV2, dT2, da, 2);
               // free dT2 and allocate dZ and copy Z to dZ
               magma_device_sync();
               timeaplQ2 = magma_wtime()-timeaplQ2;
               magma_free( dT2 );
               if(MAGMA_SUCCESS != magma_zmalloc( &dZ, N*N )) { 
                  printf ("!!!! magma_alloc failed for: dZ\n" );       
                  exit(-1);                                                           
               }

                /*
                Q2    = (magmaDoubleComplex *) malloc (N*N*sizeof(magmaDoubleComplex));
                memset(Q2 , 0, N*N*sizeof(magmaDoubleComplex));       
                magma_zgetmatrix( N, LDA1, da, N, Q2, N );
                trace_file = fopen("AJETE/Q2", "w");
                for (j = 0; j < N ; j++) 
                      for (i = 0; i < N ; i++) 
                                  fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+1,j+1,MAGMA_Z_REAL(Q2[j*N+i]) ,  MAGMA_Z_IMAG(Q2[j*N+i])  );
                fclose(trace_file);
                */

               timegemm = magma_wtime();
               // copy the eigenvectors to GPU
               magma_zsetmatrix( N, LDZ, Z, LDZ, dZ, N );
               // make GEMM Q2 * Z --> dV2 = da * dZ
               magma_zgemm( MagmaNoTrans, MagmaNoTrans, N, NE, N, c_one, da, N, dZ, N, c_zero, dV2, N);
               // copy Q1 to GPU --> dZ
               magma_zsetmatrix( N, LDA1, A1, LDA1, dZ, N );
               // make GEMM Q1 * (Q2 * Z) --> da = dZ * dV2
               magma_zgemm( MagmaNoTrans, MagmaNoTrans, N, NE, N, c_one, dZ, N, dV2, N, c_zero, da, N);
               magma_zgetmatrix( N, NE, da, N, A1, LDA1 );
               timegemm = magma_wtime()-timegemm;
           }
           if(WANTZ==2){
               /****************************************************
                * apply V2 from Right to Q1. da = da*(I-V2*T2*V2')
                * **************************************************/
               timeaplQ2 = magma_wtime();
               /*============================
                *  use GPU+CPU's
                *==========================*/             
               if((usemulticpu==1)&&(THREADS>1))
               {
                    
                   // define the size of Q to be done on CPU's and the size on GPU's
                   // note that GPU use Q(1:N_GPU) and CPU use Q(N_GPU+1:N)
                   if(THREADS>40){
                           N_GPU = (int) (0.5*(double)NE);
                           N_GPU = (N_GPU/64)*64;
                           N_CPU = NE-N_GPU;
                   }else if(THREADS>10){
#if (defined(PRECISION_s) || defined(PRECISION_d))
                           N_GPU = (int) (0.68*(double)NE);
#else
                           N_GPU = (int) (0.72*(double)NE);
#endif
                           N_GPU = (N_GPU/64)*64;
                           N_CPU = NE-N_GPU;
                   }else if(THREADS>5){
#if (defined(PRECISION_s) || defined(PRECISION_d))
                           N_GPU = (int) (0.84*(double)NE);
#else
                           N_GPU = (int) (0.86*(double)NE);
#endif
                           N_GPU = (N_GPU/64)*64;
                           N_CPU = NE-N_GPU;
                   }else{
                           N_GPU = NE;
                           N_CPU = 0;
                   }

                   magma_zgetmatrix( N_CPU, LDA1,
                                     da+(N_GPU), LDA1,
                                     A1+(N_GPU), LDA1 );
                   printf("---> calling GPU + CPU(if N_CPU>0) to apply V2 to Z with NE %d     N_GPU %d   N_CPU %d\n", (int) NE, (int) N_GPU, (int) N_CPU); 
                   core_in_all.SIDE      = 'R';
                   core_in_all.E         = A1;
                   core_in_all.E_CPU     = A1+(N_GPU);
                   core_in_all.LDE       = LDA1;
                   core_in_all.dE        = da;
                   core_in_all.dT2       = dT2;
                   core_in_all.dV2       = dV2;
                   core_in_all.N_CPU     = N_CPU;
                   core_in_all.N_GPU     = N_GPU;
                   core_in_all.NE        = NE;
                   core_in_all.T         = T;
                   core_in_all.TAU       = TAU;
                   core_in_all.V         = V;


                   // ===============================
                   // relaunch thread to apply Q
                   // ===============================
                   // Set one thread per core
                   pthread_attr_init(&thread_attr);
                   pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
                   pthread_setconcurrency(THREADS);
                  
                   // Initializations
                   for (thread = 0; thread < THREADS; thread++)
                   {
                       barrier_in[thread] = 0;
                       barrier_out[thread] = 0;
                       event_numblg[thread] = 0;
                   }
                   // Launch threads
                   for (thread = 1; thread < THREADS; thread++)
                   {
                       thread_num[thread] = thread;
                       pthread_create(&thread_id[thread], &thread_attr, applyQ_parallel_section, &thread_num[thread]);
                   }
                   thread_num[0] = 0;
                   applyQ_parallel_section(&thread_num[0]);
          
                   // Wait for completion
                   for (thread = 1; thread < THREADS; thread++)
                   {
                       void *exitcodep;
                       pthread_join(thread_id[thread], &exitcodep);
                   }
                   magma_zsetmatrix( N_CPU, LDA1,
                                     A1+(N_GPU), LDA1,
                                     da+(N_GPU), LDA1 );
                   





               /*============================
                *  use only GPU
                *==========================*/  
               }else{
                   magma_zbulge_applyQ(WANTZ, 'R', N, N, NB, Vblksiz, A1, LDA1, V, TAU, T, &INFO, dV2, dT2, da, 2);
                   magma_device_sync();
                   magma_free( dT2 );
               }
               timeaplQ2 = magma_wtime()-timeaplQ2;

               /****************************************************
                * compute the GEMM of Q*Z
                * **************************************************/
               magma_free( dT1 );
               if(MAGMA_SUCCESS != magma_zmalloc( &dZ, N*NE )) { 
                  printf ("!!!! magma_alloc failed for: dZ\n" );       
                  exit(-1);                                                           
               }
               timegemm = magma_wtime();
               // copy the eigenvectors to GPU
               magma_zsetmatrix( N, NE, Z, LDZ, dZ, N );
               //make a gemm of (Q1 * Q2) * Z = da * dZ --> dV2
               magma_zgemm( MagmaNoTrans, MagmaNoTrans, N, NE, N, c_one, da, N, dZ, N, c_zero, dV2, N);
               magma_zgetmatrix( N, NE, dV2, N, A1, LDA1 );
               timegemm = magma_wtime()-timegemm;
           }

           if(WANTZ==3){
               /****************************************************
                *  apply V2 from left to the eigenvectors Z. dZ = (I-V2*T2*V2')*Z
                * **************************************************/
               magma_free( dT1 );
               if(MAGMA_SUCCESS != magma_zmalloc( &dZ, N*NE )) { 
                  printf ("!!!! magma_alloc failed for: dZ\n" );       
                  exit(-1);                                                           
               }
               timeaplQ2 = magma_wtime();
               /*============================
                *  use GPU+CPU's
                *==========================*/             
               if((usemulticpu==1)&&(THREADS>1))
               {

                   // define the size of Q to be done on CPU's and the size on GPU's
                   // note that GPU use Q(1:N_GPU) and CPU use Q(N_GPU+1:N)
                   if(THREADS>40){
                           N_GPU = (int) (0.5*(double)NE);
                           N_GPU = (N_GPU/64)*64;
                           N_CPU = NE-N_GPU;
                   }else if(THREADS>10){
#if (defined(PRECISION_s) || defined(PRECISION_d))
                           N_GPU = (int) (0.68*(double)NE);
#else
                           N_GPU = (int) (0.72*(double)NE);
#endif
                           N_GPU = (N_GPU/64)*64;
                           N_CPU = NE-N_GPU;
                   }else if(THREADS>5){
#if (defined(PRECISION_s) || defined(PRECISION_d))
                           N_GPU = (int) (0.82*(double)NE);
#else
                           N_GPU = (int) (0.84*(double)NE);
#endif
                           N_GPU = (N_GPU/64)*64;
                           N_CPU = NE-N_GPU;
                   }else{
                           N_GPU = NE;
                           N_CPU = 0;
                   }
                   printf("---> calling GPU + CPU(if N_CPU>0) to apply V2 to Z with NE %d     N_GPU %d   N_CPU %d\n", (int) NE, (int) N_GPU, (int) N_CPU); 
                   core_in_all.SIDE      = 'L';
                   core_in_all.E         = Z;
                   core_in_all.E_CPU     = Z+(N_GPU*LDZ);
                   core_in_all.LDE       = LDZ;
                   core_in_all.dE        = dZ;
                   core_in_all.dT2       = dT2;
                   core_in_all.dV2       = dV2;
                   core_in_all.N_CPU     = N_CPU;
                   core_in_all.N_GPU     = N_GPU;
                   core_in_all.NE        = NE;
                   core_in_all.T         = T;
                   core_in_all.TAU       = TAU;
                   core_in_all.V         = V;


                   // ===============================
                   // relaunch thread to apply Q
                   // ===============================
                   // Set one thread per core
                   pthread_attr_init(&thread_attr);
                   pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
                   pthread_setconcurrency(THREADS);
                  
                   // Initializations
                   for (thread = 0; thread < THREADS; thread++)
                   {
                       barrier_in[thread] = 0;
                       barrier_out[thread] = 0;
                       event_numblg[thread] = 0;
                   }
                   // Launch threads
                   for (thread = 1; thread < THREADS; thread++)
                   {
                       thread_num[thread] = thread;
                       pthread_create(&thread_id[thread], &thread_attr, applyQ_parallel_section, &thread_num[thread]);
                   }
                   thread_num[0] = 0;
                   applyQ_parallel_section(&thread_num[0]);
          
                   // Wait for completion
                   for (thread = 1; thread < THREADS; thread++)
                   {
                       void *exitcodep;
                       pthread_join(thread_id[thread], &exitcodep);
                   }
                   magma_zsetmatrix( LDZ, N_CPU,
                                     Z+(N_GPU*LDZ),  LDZ,
                                     dZ+(N_GPU*LDZ), N );
                   
               /*============================
                *  use only GPU
                *==========================*/  
               }else{
                   magma_zbulge_applyQ(WANTZ, 'L', NE, N, NB, Vblksiz, Z, LDZ, V, TAU, T, &INFO, dV2, dT2, dZ, 3);
                   magma_device_sync();
               }

               timeaplQ2 = magma_wtime()-timeaplQ2;
               /****************************************************
                * compute the GEMM of Q1 * (Q2*Z)
                * **************************************************/
               magma_free( dT2 );
               printf("calling dgemm\n");
               timegemm = magma_wtime();
               //make a gemm of Q1 * (Q2 * Z) = Q1 * ((I-V2T2V2')*Z) = da * dZ --> dV2
               magma_zgemm( MagmaNoTrans, MagmaNoTrans, N, NE, N, c_one, da, N, dZ, N, c_zero, dV2, N);
               magma_zgetmatrix( N, NE, dV2, N, A1, LDA1 );
               timegemm = magma_wtime()-timegemm;
           }

           if(WANTZ==5){
               if(NE!=N){
                   printf("WANTZ=5 is not supported with NE=%d it compute all the eigenvectors meaning that NE=N\n", (int) NE);
                   exit(-2);
               }
               timeaplQ2 = magma_wtime();
               magma_zbulge_applyQ(WANTZ, 'R', NE, N, NB, Vblksiz, A1, LDA1, V, TAU, T, &INFO, dV2, dT2, da, 2);
               magma_zgetmatrix( N, LDA1, da, N, A1, LDA1 );
               timeaplQ2 = magma_wtime()-timeaplQ2;
               timelpk = magma_wtime();
               // compute the eigenvalues using lapack routine to be able to compare to it and used as ref 
               magma_setlapack_numthreads(THREADS);
               // call eigensolver for our resulting tridiag [D E] and form E=Q*Z
               magma_zstedc_withZ(MagmaVec, N, D2, E2, A1, LDA1);
               magma_setlapack_numthreads(1);
               timelpk = magma_wtime()-timelpk;
           }

        }
        // ************************************************


        // ************************************************
        // use parallel cpu code to apply V2 or compute Q2
        // need to be continued for Z
        // ************************************************
        if(parallel==1){
           // copy the matrix Q1 generated up from GPU to CPU because we are using the CPU's.        
           if(WANTZ==2) {
               magma_zgetmatrix( N, LDA1, da, LDA1, A1, LDA1);
           }
           // ===============================
           // relaunch thread to apply Q
           // ===============================
           // Set one thread per core
           pthread_attr_init(&thread_attr);
           pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
           pthread_setconcurrency(THREADS);
        
           // Initializations
           for (thread = 0; thread < THREADS; thread++)
           {
               barrier_in[thread] = 0;
               barrier_out[thread] = 0;
               event_numblg[thread] = 0;
           }
           // Launch threads
           for (thread = 1; thread < THREADS; thread++)
           {
               thread_num[thread] = thread;
               pthread_create(&thread_id[thread], &thread_attr, applyQ_parallel_section, &thread_num[thread]);
           }
           thread_num[0] = 0;
           applyQ_parallel_section(&thread_num[0]);
          
           // Wait for completion
           for (thread = 1; thread < THREADS; thread++)
           {
               void *exitcodep;
               pthread_join(thread_id[thread], &exitcodep);
           }
        }
        // ************************************************

       timeeigen = magma_wtime()-timeeigen;
       printf("============================================================================\n");
       printf("  Finish WANTZ %d  computing Q2       timing= %f\n", (int) WANTZ, timeaplQ2);
       if(WANTZ!=5){
           printf("  Finish WANTZ %d  making gemm        timing= %f\n", (int) WANTZ, timegemm);
           printf("  Finish WANTZ %d  eigensolver 'I'    timing= %f  threads %d   N %d    NE %d\n", (int) WANTZ, timelpk, (int) mklth, (int) N, (int) NE);
       }else{
           printf("  Finish WANTZ %d  eigensolver 'V'    timing= %f  threads %d   N %d    NE %d\n", (int) WANTZ, timelpk, (int) mklth, (int) N, (int) NE);
       }
       printf("  Finish WANTZ %d  full Eigenvectros  timing= %f\n",(int) WANTZ, timeeigen);
       printf("============================================================================\n");
    }

             



       /*
       trace_file = fopen("AJETE/U", "w");
       for (j = 0; j < N ; j++) 
             for (i = 0; i < N ; i++) 
                        //fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,A1[j*LDA1+i]);
                         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+1,j+1,MAGMA_Z_REAL(A1[j*LDA1+i]) ,  MAGMA_Z_IMAG(A1[j*LDA1+i])  );
       fclose(trace_file);

       trace_file = fopen("AJETE/D", "w");
       for (j = 0; j < N ; j++) 
                fprintf(trace_file,"%10d%10d%40.30e\n",j+1,1,D2[j]);
       fclose(trace_file);
       */



/*
            magmaDoubleComplex mydz=c_zero,mydo=c_one;
            magmaDoubleComplex *Z = (magmaDoubleComplex *) malloc(N*N*sizeof(magmaDoubleComplex));
   dgemm_("N","N",&N,&N,&N,&mydo,A1,&LDA1,Q1,&LDQ1,&mydz,Z,&LDA1);
   memcpy(A1,Z,N*N*sizeof(magmaDoubleComplex));
   memcpy(Q1,Z,N*N*sizeof(magmaDoubleComplex));
*/
    /*
       trace_file = fopen("AJETE/Qf", "w");
       for (j = 0; j < N ; j++) 
             for (i = 0; i < N ; i++) 
                            fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,A1[j*LDA1+i]);
       fclose(trace_file);
*/

    if(CHECK)
    {
       if(WANTZ!=1) JOBZ='N';
       if(WANTZ==1) JOBZ='V';
       magma_zstedc_withZ(JOBZ, N, D2, E2, Q2, LDQ2);            
       //magma_zstedc_withZ(MagmaNoVec, N, D2, E2, Q1, LDQ1);            
    }
    



#if defined(PRECISION_d)  
#if defined(CHECKEIG)
    if(CHECK)
    {
        /************************************************* 
        *     CALLING LAPACK on original matrix
        * ***********************************************/
        int    LWORK=256*N;
        magmaDoubleComplex *WORK    = (magmaDoubleComplex *) malloc( LWORK * sizeof(magmaDoubleComplex) );
        D1      = (double *) malloc( N * sizeof(double) );
        double *E1      = (double *) malloc( N * sizeof(double) );
        magmaDoubleComplex *ALPK    = (magmaDoubleComplex *) malloc (N*LDAINIT*sizeof(magmaDoubleComplex));
        memcpy(ALPK, AINIT, N*LDAINIT*sizeof(magmaDoubleComplex));

        timelpk = magma_wtime();
        lapackf77_zhbtrd("N", "L", &N, &NB, ALPK, &LDAINIT, D1, E1, WORK, &N, WORK, &INFO); 
        timelpk = magma_wtime() - timelpk;
        printf("\n");                
        printf("  Time ZHBTRD-MKL-LAPACK                      :   %f    N : %10d    NB : %10d\n\n\n",timelpk, N, NB );
        /* call eigensolver */
        lapackf77_dsterf(&N, D1, E1, &INFO);             
       /* ***********************************************/ 
       // call eigensolver 
       //dsterf_( &N, D2, E2, &INFO); 
       //magma_zstedc_withZ(JOBZ, N, D2, E2, Q1, LDQ1);
       // compare result 
       cmp_vals(N, D1, D2, &nrmI, &nrm1, &nrm2);

       magmaDoubleComplex *WORKAJETER;
       double *RWORKAJETER, *RESU;
       WORKAJETER  = (magmaDoubleComplex *) malloc( (2* N * N + N) * sizeof(magmaDoubleComplex) );
       RWORKAJETER = (double *) malloc( N * sizeof(double) );
       RESU        = (double *) malloc(10*sizeof(double));
       int MATYPE;
       memset(RESU,0,10*sizeof(double));

 
       MATYPE=2;
       magmaDoubleComplex NOTHING=c_zero;
       zcheck_eig_(&JOBZ, &MATYPE, &N, &NB, AINIT, &LDAINIT, &NOTHING, &NOTHING, D2 , D1, Q2, &LDQ2, WORKAJETER, RWORKAJETER, RESU );



 



        printf("\n");
        printf(" ==================================================================================================================================\n");
        printf("   ==> INFO voici  threads=%d    N=%d    NB=%d   BAND=%d WANTZ=%d\n",thread,N, NB, BAND, WANTZ);
        printf(" ==================================================================================================================================\n");
        printf("            ZHBTRD                : %15s\n", "STATblgv9withQ    ");
        printf(" ==================================================================================================================================\n");
        if(WANTZ==1)
           printf(" | A - U S U' | / ( |A| n ulp )   : %15.3E\n",        RESU[0]); 
        if(WANTZ==1)
           printf(" | I - U U' | / ( n ulp )         : %15.3E\n", RESU[1]);
        printf(" | D1 - EVEIGS | / (|D| ulp)      : %15.3E\n",  RESU[2]);
        printf(" max | D1 - EVEIGS |              : %15.3E\n",  RESU[6]);
        printf(" ==================================================================================================================================\n\n\n");

        printf(" ***********************************************************************************************************************************\n");
        printf(" Hello here are the norm  Infinite (max)=%e  norm one (sum)=%e   norm2(sqrt)=%e\n",nrmI, nrm1, nrm2);
        printf(" ***********************************************************************************************************************************\n\n");


    }
#endif
#endif


    magma_free( dV2 );
    magma_free( da );
    magma_free_cpu(A2);
    magma_free_cpu(TAU);
    magma_free_cpu(V);
    magma_free_cpu(T);

    return 0;
} /* END ZHETRD_BHE2TRC */







//##################################################################################################
static void *parallel_section(void *thread_id)
{
    int my_core_id       = *((int*)thread_id);
    int allcores_num     = core_in_all.cores_num;
    int locores_num      = core_in_all.locores_num;
    int overlapQ1        = core_in_all.overlapQ1;
    magmaDoubleComplex *dQ1 = core_in_all.dQ1;
    magmaDoubleComplex *dT1 = core_in_all.dT1;
    int N                = core_in_all.N;
    int NB               = core_in_all.NB;
    int WANTZ = core_in_all.WANTZ;
    static volatile int sys_corenbr = 1;
    magma_int_t INFO, i, my_newcore_id, lastcoreid;
    magmaDoubleComplex *NOTUSED;
    real_Double_t timeB=0.0, timeT=0.0, timeall=0.0, timeaplQ1=0.0;

#if defined(MAGMA_SETAFFINITY)    
    // bind threads 
    cpu_set_t set;
    // bind threads 
    CPU_ZERO( &set );
    CPU_SET( my_core_id, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif

    log_eventsblg = 1;
    core_event_startblg(my_core_id);
    barrier(my_core_id, allcores_num);
    core_event_endblg(my_core_id);
    core_log_eventblg(0x000000, my_core_id);

    if((my_core_id == 0)&&(overlapQ1==0)){
            if(locores_num!=allcores_num)
                    printf("\n\n\n WARNING **** not all cpu used on a version where overlapQ1 is disabled\n\n\n"); 
    }

    // timing
    if (my_core_id == 0){
       timeall = magma_wtime();
    }


    /*################################################
     *   WANTZ > 0 
     *################################################*/
    if((WANTZ>0))
    {
        /* compute the Q1 overlapped with the bulge chasing+T.
         * if cores_cum=1 meaning that only one thread is working 
         * meaning the whole code is sequential so nothing special 
         * to be done, i will call GPU then i will call bulgechasing.
         * if not the GPU is choosed to run on last coreid to avoid change 
         * in the barrier and the original bulge code. However we discover 
         * that the performance of GPU when going to last core is bad.  
         * it look like the cuda is reinitializing the GPU with the new core.
         * so we want to keep the GPU with the main core =0 and try to change id's.
         * However to avoid a lot of change in the code because the bulge 
         * and the T and the barrier are also based on core number "0" 
         * so we will cheat giving the bulge the remaining cores[2-allcores_num] 
         * saying that they start at core "0" which is core "1" so making shift
         * and give the GPU the main original core "0" saying that this is last_core.
         * so I will create a newcoreid and give original core "0" and id=allcores_num
         * and shift the remaining cores[2-allcores_num] back by one. 
         * */ 
         /************************************************
          *  only one core is running ==> code is sequential
          ************************************************/   
         if(allcores_num==1)
         {
             my_newcore_id = my_core_id;
             //=========================
             //    compute Q1
             //=========================
             if(overlapQ1==1){
                 magma_device_sync();
                 timeaplQ1 = magma_wtime();
                 magma_zungqr_2stage_gpu(N, N, N, dQ1, N, NOTUSED, dT1, NB, &INFO);
                 magma_device_sync();
                 //cublasGetMatrix( N, LDA1, sizeof(magmaDoubleComplex), da, LDA1, A1, LDA1);
                 timeaplQ1 = magma_wtime()-timeaplQ1;
                 printf("  Finish applyQ1 timing= %f\n", timeaplQ1); 
             }
           
             //=========================
             //    bulge chasing
             //=========================
             if(my_newcore_id == 0)timeB = magma_wtime();
             tile_bulge_parallel(my_newcore_id);
             barrier(my_newcore_id, locores_num);
             if(my_newcore_id == 0){
                 timeB = magma_wtime()-timeB;
                 printf("  Finish BULGE   timing= %f\n", timeB);
             }
           
             //=========================
             // compute the T's to be used when applying Q2
             //=========================
             if(my_newcore_id == 0)timeT = magma_wtime();
             tile_bulge_computeT_parallel(my_newcore_id);
             barrier(my_newcore_id, locores_num);
             // timing
             if (my_newcore_id == 0){
                timeT = magma_wtime()-timeT;
                printf("  Finish T's     timing= %f\n", timeT);
             }
         /************************************************
          *   more than one core
          ************************************************/   
         }else{
            if(overlapQ1==1)
            {
                lastcoreid =  allcores_num-1;
                // the the coreid "0" becomes last_one allcores_num-1
                // and "1" becomes "0" and so on            
                if(my_core_id==0){
                    my_newcore_id = lastcoreid; // giving it last core id
                }else{
                    my_newcore_id = my_core_id-1;
                }
            }else{
                lastcoreid = -1;
                my_newcore_id = my_core_id;
            }
       
       
            /* I am the last core in the new indexing and the original core=0 */
            if(my_newcore_id==lastcoreid)
            {
                //=============================================
                //    compute Q1 on last newcoreid
                //=============================================
                magma_device_sync();
                timeaplQ1 = magma_wtime();
                magma_zungqr_2stage_gpu(N, N, N, dQ1, N, NOTUSED, dT1, NB, &INFO);
                magma_device_sync();
                //cublasGetMatrix( N, LDA1, sizeof(magmaDoubleComplex), da, LDA1, A1, LDA1);
                timeaplQ1 = magma_wtime()-timeaplQ1;
                printf("  Finish applyQ1 timing= %f\n", timeaplQ1); 
            /* I am one of the remaining cores*/
            }else{
                //=========================
                //    bulge chasing
                //=========================
                if(my_newcore_id == 0)timeB = magma_wtime();
                tile_bulge_parallel(my_newcore_id);
                barrier(my_newcore_id, locores_num);
                if(my_newcore_id == 0){
                    timeB = magma_wtime()-timeB;
                    printf("  Finish BULGE   timing= %f\n", timeB);
                }
               
                //=========================
                // compute the T's to be used when applying Q2
                //=========================
                if(my_newcore_id == 0)timeT = magma_wtime();
                tile_bulge_computeT_parallel(my_newcore_id);
                barrier(my_newcore_id, locores_num);
                // timing
                if (my_newcore_id == 0){
                   timeT = magma_wtime()-timeT;
                   printf("  Finish T's     timing= %f\n", timeT);
                }
            } // END if my_newcore_id==allcores_num-1
       
         } // END of more than one core

    /*################################################
     *   WANTZ = 0  
     *################################################*/
    }else{
        my_newcore_id = my_core_id;
        //=========================
        //    bulge chasing
        //=========================
        if(my_newcore_id == 0)timeB = magma_wtime();
        tile_bulge_parallel(my_newcore_id);
        barrier(my_newcore_id, locores_num);
        if(my_newcore_id == 0){
            timeB = magma_wtime()-timeB;
            printf("  Finish BULGE   timing= %f\n", timeB);
        }
    }
    /*################################################
     *   END of WANTZ  
     *################################################*/




    // put a barrier on all the cores to be sure that 
    // both [1:cores_num-1] working for bulge+T and cores_num
    // working for Q1 have finish.
    barrier(my_core_id, allcores_num);


    // timing
    if (my_core_id == 0){
        timeall = magma_wtime()-timeall;
        *(core_in_all.timeblg) = timeall;     
    }

#if defined(MAGMA_SETAFFINITY)    
    // unbind threads 
    sys_corenbr = sysconf(_SC_NPROCESSORS_ONLN);
    CPU_ZERO( &set );
    for(i=0; i<sys_corenbr; i++)
            CPU_SET( i, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif

    return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

//##################################################################################################
static void *applyQ_parallel_section(void *thread_id)
{
    int my_core_id       = *((int*)thread_id);
    int allcores_num     = core_in_all.cores_num;
    int locores_num      = core_in_all.locores_num;
    int usemulticpu      = core_in_all.usemulticpu;
    magmaDoubleComplex *dZ  = core_in_all.dE;
    magmaDoubleComplex *dT2 = core_in_all.dT2;
    magmaDoubleComplex *dV2 = core_in_all.dV2;
    magmaDoubleComplex *Z   = core_in_all.E;
    magmaDoubleComplex *T2  = core_in_all.T;
    magmaDoubleComplex *V2  = core_in_all.V;
    magmaDoubleComplex *TAU2= core_in_all.TAU;
    int LDZ              = core_in_all.LDE;
    int NE               = core_in_all.NE;
    int N_CPU            = core_in_all.N_CPU;
    int N_GPU            = core_in_all.N_GPU;
    int N                = core_in_all.N;
    int NB               = core_in_all.NB;
    int Vblksiz          = core_in_all.Vblksiz;
    int WANTZ            = core_in_all.WANTZ;
    static volatile int sys_corenbr = 1;
    magma_int_t INFO, i, my_newcore_id, lastcoreid;

    real_Double_t timeQcpu=0.0, timeQgpu=0.0;
    magmaDoubleComplex *NOTUSED;
    int INOTUSED=0;


    if(WANTZ<=0) 
            return 0;


#if defined(MAGMA_SETAFFINITY)    
    cpu_set_t set;
    CPU_ZERO( &set );
    CPU_SET( my_core_id, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif


    log_eventsblg = 1;
    core_event_startblg(my_core_id);
    barrier(my_core_id, allcores_num);
    core_event_endblg(my_core_id);
    core_log_eventblg(0x000000, my_core_id);


/*################################################
     *   WANTZ == 2
     *################################################*/
    if((WANTZ==2))
    {
         /************************************************
          *  only one core is running ==> code is sequential
          *  usually code should not come here it is better 
          *  when we have only 1 cpu to just run the gpu 
          *  from the main function
          ************************************************/   
         if(allcores_num==1)
         {
             my_newcore_id = my_core_id;
             //=========================
             //    apply Q2
             //=========================
             if(usemulticpu==1){
                 timeQgpu = magma_wtime();
                 // here dZ is da and Z=Q1
                 magma_zbulge_applyQ(WANTZ, 'R', NE, N, NB, Vblksiz, NOTUSED, N, V2, TAU2, T2, &INFO, dV2, dT2, dZ, 2);
                 magma_device_sync();
                 timeQgpu = magma_wtime()-timeQgpu;
                 printf("  Finish Q2_GPU GGG timing= %f\n", timeQgpu);
             }
         /************************************************
          *   more than one core
          ************************************************/   
         }else if(usemulticpu==1){
            lastcoreid =  allcores_num-1;
            // the the coreid "0" becomes last_one allcores_num-1
            // and "1" becomes "0" and so on            
            if(my_core_id==0){
                my_newcore_id = lastcoreid; // giving it last core id
            }else{
                my_newcore_id = my_core_id-1;
            }
       
       
            /* I am the last core in the new indexing and the original core=0 */
            if(my_newcore_id==lastcoreid)
            {
                //=============================================
                //   on GPU on last_newcoreid:
                //    - apply V2*Z(:,1:N_GPU)
                //=============================================
                 timeQgpu = magma_wtime();
                 magma_zbulge_applyQ(WANTZ, 'R', N_GPU, N, NB, Vblksiz, NOTUSED, N, V2, TAU2, T2, &INFO, dV2, dT2, dZ, 2);
                 magma_device_sync();
                 timeQgpu = magma_wtime()-timeQgpu;
                 printf("  Finish Q2_GPU GGG timing= %f\n", timeQgpu);
            /* I am one of the remaining cores*/
            }else if(N_CPU>0){
                //=============================================
                //   on CPU on core 1:last_newcoreid-1
                //    - apply V2*Z(:,N_GPU+1:N)
                //=============================================
                if(my_newcore_id == 0)timeQcpu = magma_wtime();
                tile_bulge_applyQ_parallel(my_newcore_id);
                barrier(my_newcore_id, locores_num);
                if(my_newcore_id == 0){
                    timeQcpu = magma_wtime()-timeQcpu;
                    printf("  Finish Q2_CPU CCC timing= %f\n", timeQcpu);
                }

            } // END if my_newcore_id==allcores_num-1
       
         } // END of more than one core

    }// END of WANTZ==3


    /*################################################
     *   WANTZ > 0 
     *################################################*/
    if((WANTZ==3))
    {
         /************************************************
          *  only one core is running ==> code is sequential
          *  usually code should not come here it is better 
          *  when we have only 1 cpu to just run the gpu 
          *  from the main function
          ************************************************/   
         if(allcores_num==1)
         {
             my_newcore_id = my_core_id;
             //=========================
             //    apply Q2
             //=========================
             if(usemulticpu==1){
                 timeQgpu = magma_wtime();
                 magma_zbulge_applyQ(WANTZ, 'L', NE, N, NB, Vblksiz, Z, LDZ, V2, TAU2, T2, &INFO, dV2, dT2, dZ, 3);
                 magma_device_sync();
                 timeQgpu = magma_wtime()-timeQgpu;
                 printf("  Finish Q2_GPU GGG timing= %f\n", timeQgpu);
             }
         /************************************************
          *   more than one core
          ************************************************/   
         }else if(usemulticpu==1){
            lastcoreid =  allcores_num-1;
            // the the coreid "0" becomes last_one allcores_num-1
            // and "1" becomes "0" and so on            
            if(my_core_id==0){
                my_newcore_id = lastcoreid; // giving it last core id
            }else{
                my_newcore_id = my_core_id-1;
            }
       
       
            /* I am the last core in the new indexing and the original core=0 */
            if(my_newcore_id==lastcoreid)
            {
                //=============================================
                //   on GPU on last_newcoreid:
                //    - apply V2*Z(:,1:N_GPU)
                //=============================================
                 timeQgpu = magma_wtime();
                 magma_zbulge_applyQ(WANTZ, 'L', N_GPU, N, NB, Vblksiz, Z, LDZ, V2, TAU2, T2, &INFO, dV2, dT2, dZ, 3);
                 magma_device_sync();
                 timeQgpu = magma_wtime()-timeQgpu;
                 printf("  Finish Q2_GPU GGG timing= %f\n", timeQgpu);
            /* I am one of the remaining cores*/
            }else if(N_CPU>0){
                //=============================================
                //   on CPU on core 1:last_newcoreid-1
                //    - apply V2*Z(:,N_GPU+1:N)
                //=============================================
                if(my_newcore_id == 0)timeQcpu = magma_wtime();
                tile_bulge_applyQ_parallel(my_newcore_id);
                barrier(my_newcore_id, locores_num);
                if(my_newcore_id == 0){
                    timeQcpu = magma_wtime()-timeQcpu;
                    printf("  Finish Q2_CPU CCC timing= %f\n", timeQcpu);
                }

            } // END if my_newcore_id==allcores_num-1
       
         } // END of more than one core

    }// END of WANTZ==3



    // put a barrier on all the cores to be sure that 
    // both [1:cores_num-1] working with CPU_code and 
    // last_cores_num working with GPU have finish.
    barrier(my_core_id, allcores_num);



#if defined(MAGMA_SETAFFINITY)    
    // unbind threads 
    sys_corenbr = sysconf(_SC_NPROCESSORS_ONLN);
    CPU_ZERO( &set );
    for(i=0; i<sys_corenbr; i++)
            CPU_SET( i, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////////////////////////////
static void tile_bulge_parallel(int my_core_id)
{
    int core;
    int INFO;
    /* CHANGE HERE to SWITCH FROM local_cores_num to cores_num
     * or do it at the startup by putting core_in_all.es_num=cores_num*/
    int cores_num = core_in_all.locores_num;
    magmaDoubleComplex *A = core_in_all.A;
    magmaDoubleComplex *V = core_in_all.V;
    magmaDoubleComplex *TAU = core_in_all.TAU;
    int N = core_in_all.N;
    int NB = core_in_all.NB;
    int NBTILES = core_in_all.NBTILES;
    int LDA= core_in_all.LDA;
    int BAND= core_in_all.BAND;
    int grsiz= core_in_all.grsiz;
    int Vblksiz = core_in_all.Vblksiz;
    volatile int *prog = core_in_all.ss_prog;
    //%===========================
    //%   local variables
    //%===========================
    int sweepid, myid, shift, stt, st, ed, stind, edind;
    int blklastind, colpt, algtype;
    int stepercol,mylastid,grnb,grid;
    int i,j,m,k;
    int thgrsiz, thgrnb, thgrid, thed;
    int coreinit,loopfirsttask,coreid;
    int colblktile,maxrequiredcores,colpercore,mycoresnb;
    int fin;


    if(N<=0)
        return ;


    //printf("=================> my core id %d of %d\n",my_core_id, cores_num);

    if((BAND!=0) && (BAND!=6) && (BAND!=62) && (BAND!=63)){
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

    mycoresnb = cores_num;
    //grsiz   = 2;
    shift   = 5;
    if(grsiz==1) 
        colblktile=1;
    else
        colblktile=grsiz/2;

    maxrequiredcores = NBTILES/colblktile;
    if(maxrequiredcores<1)maxrequiredcores=1;
    colpercore  = colblktile*NB;
    if(mycoresnb > maxrequiredcores)
    {
        if(my_core_id==0)printf("==================================================================================\n");
        if(my_core_id==0)printf("  WARNING only %3d threads are required to run this test optimizing cache reuse\n",maxrequiredcores);
        if(my_core_id==0)printf("==================================================================================\n");
        mycoresnb = maxrequiredcores;
    }
    thgrsiz = N;//mycoresnb;

  if(my_core_id==0) printf("  Static bulgechasing version v9_9col threads  %4d      N %5d      NB %5d    grs %4d thgrsiz %4d  BAND %4d\n",cores_num, N, NB, grsiz,thgrsiz,BAND);





  i = shift/grsiz;
  stepercol =  i*grsiz == shift ? i:i+1;

  i       = (N-1)/thgrsiz;
  thgrnb  = i*thgrsiz == (N-1) ? i:i+1;

  for (thgrid = 1; thgrid<=thgrnb; thgrid++){
     stt  = (thgrid-1)*thgrsiz+1;
     thed = min( (stt + thgrsiz -1), (N-1));
     for (i = stt; i <= N-1; i++){
        ed=min(i,thed);
        if(stt>ed)break;
        for (m = 1; m <=stepercol; m++){ 
            st=stt;
            for (sweepid = st; sweepid <=ed; sweepid++){

                for (k = 1; k <=grsiz; k++){ 
                    myid = (i-sweepid)*(stepercol*grsiz) +(m-1)*grsiz + k;        
                    if(myid%2 ==0){
                         colpt      = (myid/2)*NB+1+sweepid-1;
                         stind      = colpt-NB+1;
                         edind      = min(colpt,N);
                         blklastind = colpt;
                         if(stind>=edind){
                             printf("ERROR---------> st>=ed  %d  %d\n\n",stind, edind);
                             exit(-10);
                         }
                    }else{
                         colpt      = ((myid+1)/2)*NB + 1 +sweepid -1 ;
                         stind      = colpt-NB+1;
                         edind      = min(colpt,N);
                         if( (stind>=edind-1) && (edind==N) )
                             blklastind=N;
                         else
                             blklastind=0;
                         if(stind>edind){
                             printf("ERROR---------> st>=ed  %d  %d\n\n",stind, edind);
                             exit(-10);
                         }
                    }

                    coreid = (stind/colpercore)%mycoresnb;


   //printf("    current col %3d sweep %3d myid %3d  coreid %7d my_core_id %3d ---------------------- st %2d  ed %2d\n",i,sweepid, myid, coreid,my_core_id, stind, edind); 
 //printf("MYID %2d prog  %3d %3d %3d %3d %3d %3d %3d\n",my_core_id,prog[0],prog[1],prog[2],prog[3],prog[4],prog[5],prog[6]);

                    if(my_core_id==coreid)
                    {
                        //printf("--> current col %3d sweep %3d myid %3d  my_core_id %3d   prog[myid-1] %3d    prog[myid+shiftd] %3d\n",i,sweepid, myid,my_core_id,prog[myid-1], prog[myid+shift]);
                        //__sync_synchronize();
                        //fflush(stdout);


                        fin=0;
                        while(fin==0)
                        { 
                            if(myid==1)
                            {
                            if( (prog[myid+shift-1]== (sweepid-1)) )
                                {
                                    
                                    if(LOG) core_event_startblg(my_core_id);
                                    /*
                                    if(BAND==0)
                                            TRD_type1cHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);
                                    else if(BAND==6)*/
                                            magma_ztrdtype1cbHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);/*
                                    else if(BAND==62)
                                            TRD_hbcelr_v62sym_withQ(N, NB, A, LDA, V, TAU, stind, edind, myid, sweepid, Vblksiz);*/
                                    if(LOG) {
                                        core_event_endblg(my_core_id);
                                        core_log_eventblg(0x006680, my_core_id);
                                    }
                                    
                                    fin=1;
                                    prog[myid]= sweepid;
                                    if(blklastind >= (N-1))
                                    {
                                        for (j = 1; j <= shift; j++) 
                                            prog[myid+j]=sweepid;
                                    }                                
                                    } // END progress condition
                            }else{
                                if( (prog[myid-1]==sweepid) && (prog[myid+shift-1]== (sweepid-1)) )
                                    {
                                        
                                    if(LOG) core_event_startblg(my_core_id);
                                    /*
                                    if(BAND==0){
                                        if(myid%2 == 0)
                                            TRD_type2cHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);
                                        else
                                            TRD_type3cHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);
                                    }else if(BAND==6){*/
                                        if(myid%2 == 0)
                                            magma_ztrdtype2cbHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);
                                        else
                                            magma_ztrdtype3cbHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);/*
                                    }else if(BAND==62){
                                            TRD_hbcelr_v62sym_withQ(N, NB, A, LDA, V, TAU, stind, edind, myid, sweepid, Vblksiz);
                                    }*/
                                    if(LOG) {
                                        core_event_endblg(my_core_id);
                                        core_log_eventblg(0xff0000, my_core_id);
                                    }        
                                    
                                    fin=1;
                                    prog[myid]= sweepid;
                                    if(blklastind >= (N-1))
                                    {
                                        for (j = 1; j <= shift+mycoresnb; j++) 
                                            prog[myid+j]=sweepid;
                                    }                                
                                    } // END progress condition
                            } // END if myid==1
                        } // END while loop
                        
                    } // END if my_core_id==coreid
                    
                    if(blklastind >= (N-1))
                    {
                          stt=stt+1;
                          break;
                    }
                }   // END for k=1:grsiz
               } // END for sweepid=st:ed
        } // END for m=1:stepercol
     } // END for i=1:N-1
//barrier(my_core_id, cores_num);   
  } // END for thgrid=1:thgrnb




// momentary solution for complex version when A(N,N-1) is complex and to avoid making it real out of this function 
// which will require to multiply the col/row of the Eigenvectors or Q by the scalar that make A(N,N-1) to real.
//       barrier(my_core_id, cores_num);
//       if(my_core_id == 0)magma_ztrdtype1cbHLsym_withQ(N, NB, A, LDA, V, TAU, N, N, N-1, Vblksiz);

} // END FUNCTION
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)   &(T[(m)])
static void tile_bulge_computeT_parallel(int my_core_id)
{
    magma_int_t INFO;
    /* CHANGE HERE to SWITCH FROM local_cores_num to cores_num
     * or do it at the startup by putting core_in_all.es_num=cores_num*/
    magma_int_t cores_num = core_in_all.locores_num;
    magmaDoubleComplex *T     = core_in_all.T;
    magmaDoubleComplex *V     = core_in_all.V;
    magmaDoubleComplex *TAU   = core_in_all.TAU;
    magma_int_t  N         = core_in_all.N;
    magma_int_t  NB        = core_in_all.NB;
    magma_int_t  Vblksiz   = core_in_all.Vblksiz;

    //%===========================
    //%   local variables
    //%===========================
    magma_int_t LDT, LDV,blklen,firstcolj;
    magma_int_t bg, nbGblk,rownbm, k, m, n;
    magma_int_t st,ed,fst,vlen,vnb,colj;
    magma_int_t blkid,vpos,taupos,tpos;
    magma_int_t cur_blksiz,avai_blksiz, ncolinvolvd;
    magma_int_t nbgr, colst, coled, version;
    magma_int_t blkpercore,blkcnt, myid;


    if(N<=0)
        return ;

    findVTsiz(N, NB, Vblksiz, &blkcnt, &LDV);
    blkpercore = blkcnt/cores_num;

    LDT     = Vblksiz;    
    LDV     = NB+Vblksiz-1;
    blklen  = LDV*Vblksiz;
    nbGblk  = plasma_ceildiv((N-1),Vblksiz);

    if(my_core_id==0) {
        printf("  COMPUTE T parallel threads %d with  N %d   NB %d   Vblksiz %d\n",
               (int) cores_num, (int) N, (int) NB, (int) Vblksiz);
    }
    
        for (bg = nbGblk; bg>0; bg--)
        {
           firstcolj = (bg-1)*Vblksiz + 1;
           rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
           if(bg==nbGblk) rownbm    = plasma_ceildiv((N-(firstcolj)),NB);  // last blk has size=1 used for complex to handle A(N,N-1)
           for (m = rownbm; m>0; m--)
           {
               vlen = 0;
               vnb  = 0;
               colj      = (bg-1)*Vblksiz; // for k=0;I compute the fst and then can remove it from the loop
               fst       = (rownbm -m)*NB+colj +1;
               for (k=0; k<Vblksiz; k++)
               {
                   colj     = (bg-1)*Vblksiz + k;
                   st       = (rownbm -m)*NB+colj +1;
                   ed       = min(st+NB-1,N-1);
                   if(st>ed)break;
                   if((st==ed)&&(colj!=N-2))break;
                   vlen=ed-fst+1;
                   vnb=k+1;
               }        
               colj     = (bg-1)*Vblksiz;
               findVTpos(N,NB,Vblksiz,colj,fst, &vpos, &taupos, &tpos, &blkid);
               myid = blkid/blkpercore;
               if(my_core_id==(myid%cores_num)){
                  if((vlen>0)&&(vnb>0))
                      lapackf77_zlarft( "F", "C", &vlen, &vnb, V(vpos), &LDV, TAU(taupos), T(tpos), &LDT);
               }
           }
        }
}
#undef V
#undef TAU
#undef T
////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////
#define E(m,n)   &(E[(m) + LDE*(n)])
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)     &(T[(m)])
static void tile_bulge_applyQ_parallel(int my_core_id)
{
    magma_int_t INFO;
    /* CHANGE HERE to SWITCH FROM local_cores_num to cores_num
     * or do it at the startup by putting core_in_all.es_num=cores_num*/
    magma_int_t cores_num = core_in_all.locores_num;
    magmaDoubleComplex *E      = core_in_all.E_CPU;

    magmaDoubleComplex *T      = core_in_all.T;
    magmaDoubleComplex *V      = core_in_all.V;
    magmaDoubleComplex *TAU    = core_in_all.TAU;
    magma_int_t  N_CPU     = core_in_all.N_CPU;
    magma_int_t  NE        = core_in_all.NE;
    magma_int_t  N         = core_in_all.N;
    magma_int_t  NB        = core_in_all.NB;
    magma_int_t  Vblksiz   = core_in_all.Vblksiz;
    magma_int_t  LDE       = core_in_all.LDE;
    char SIDE      = core_in_all.SIDE;
    magma_int_t  WANTZ     = core_in_all.WANTZ;

    //%===========================
    //%   local variables
    //%===========================
    magma_int_t LDT,LDV,blklen,firstcolj;
    magma_int_t bg, nbGblk,rownbm, k, m, n;
    magma_int_t st,ed,fst,vlen,vnb,colj,len;
    magma_int_t blkid, vpos,taupos,tpos;
    magmaDoubleComplex *WORK;
    magma_int_t LWORK;
    magma_int_t  cur_blksiz,avai_blksiz, ncolinvolvd;
    magma_int_t  nbgr, colst, coled, versionL,versionR;
    magma_int_t chunkid,nbchunk,colpercore,corest,corelen;
    magma_int_t coreid,mycoresnb,maxrequiredcores;


    if(N<=0)
        return ;
    if(NE<=0)
        return ;
    if(N_CPU<=0)
        return ;



    INFO=0;    
    versionL = 114;
    versionR = 92;
    LDT      = Vblksiz;
    LDV      = NB+Vblksiz-1;    
    blklen   = LDV*Vblksiz;
    nbGblk   = plasma_ceildiv((N-1),Vblksiz);
    //LWORK   = 2*N*max(Vblksiz,64);
    //WORK    = (magmaDoubleComplex *) malloc (LWORK*sizeof(magmaDoubleComplex));


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
     *
     * version v2: for each V_k and T_k thread t apply those V_k and 
     *                    T_k to E_k for all its chunck, then move to V_k+1 
     *                    T_k+1 and apply them to E_k+1 on all the chunck, 
     *                    and so on. the difference is that, thread keep V_k
     *                    and T_K while move over the E_k.
     *                    both version look like similar in perf.                   
     * */
    colpercore = min(NB,120); //colpercore = N make the code sequential running on thread=0;
    //colpercore =  N/cores_num; 

    colpercore =  plasma_ceildiv(N_CPU,cores_num);
    if(colpercore>1000)
        colpercore =  plasma_ceildiv(colpercore,10);
    else if(colpercore>800)
        colpercore =  plasma_ceildiv(colpercore,8);
    else if(colpercore>600)
        colpercore =  plasma_ceildiv(colpercore,6);
    else if(colpercore>400)
        colpercore =  plasma_ceildiv(colpercore,4);
    else if(colpercore>200)
        colpercore =  plasma_ceildiv(colpercore,2);
    if(colpercore>200)colpercore=120;

    LWORK   = 2*colpercore*max(Vblksiz,64);
    //LWORK   = 2*N_CPU*max(Vblksiz,64);
    WORK    = (magmaDoubleComplex *) malloc (LWORK*sizeof(magmaDoubleComplex));


    nbchunk    =  plasma_ceildiv(N_CPU,colpercore);

    mycoresnb = cores_num;
    maxrequiredcores = nbchunk;
    if(maxrequiredcores<1)maxrequiredcores=1;
    if(mycoresnb > maxrequiredcores)
    {
        if(my_core_id==0)printf("==================================================================================\n");
        if(my_core_id==0)printf("  WARNING only %3d threads are required to run this test optimizing cache reuse\n", (int) maxrequiredcores);
        if(my_core_id==0)printf("==================================================================================\n");
        mycoresnb = maxrequiredcores;
    }


    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            each q_i consist of applying V to a block of row E(row_i,:) and applies are overlapped meaning 
     *            that q_i+1 overlap a portion of the E(row_i, :).
     *            IN parallel E is splitten in vertical block over the threads  */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n 
     *            each q_i consist of applying V to a block of col E(:, col_i,:) and the applies are overlapped meaning 
     *            that q_i+1 overlap a portion of the E(:, col_i).
     *            IN parallel E is splitten in horizontal block over the threads  */

     /* WANTZ = 1 meaning E is IDENTITY so form Q using optimized update. 
      *         So we use the reverse order from small q to large one, 
      *         so from q_n to q_1 so Left update to Identity.
      *         Use versionL 113 because in 114 we need to update the whole matrix and not in icreasing order.
      * WANTZ = 2 meaning E is a full matrix and need to be updated from Left or Right so use normal update
      * */
    if(WANTZ==1) 
    {
        versionL=113;
        SIDE='L';
        printf("\n\n\nWARNING THIS OPTION CUOLD NOT RUN ON BOTH GPU AND CPU. PROG WILL EXIT\n\n\n");
    }

    //printf("  ENTERING FUNCTION APPLY Q_v115: same as 113(L) or 114(L) or 93(R)"\n);
    if(my_core_id==0) {
        printf("  APPLY Q_v1   parallel with threads %d   nbchunk %d  colpercore %d  N %d  N_CPU %d   NB %d   Vblksiz %d SIDE %c versionL %d versionR %d WANTZ %d\n",
               (int) cores_num, (int) nbchunk, (int) colpercore, (int) N, (int) N_CPU, (int) NB, (int) Vblksiz, (int) SIDE, (int) versionL, (int) versionR,  (int) WANTZ);
    }
    
    for (chunkid = 0; chunkid<nbchunk; chunkid++)
    {
        coreid  = chunkid%mycoresnb;
        corest  = chunkid*colpercore;
        corelen = min(colpercore, (N_CPU-(chunkid*colpercore)));

        if(my_core_id==coreid)
        {
            //printf("mycore id %d voici nbchunk %d  chunkid %d  coreid %d, corest %d, corelen %d\n",my_core_id,nbchunk, chunkid, coreid, corest, corelen);
            if(SIDE=='L'){
                for (bg = nbGblk; bg>0; bg--)
                {
                   firstcolj = (bg-1)*Vblksiz + 1;
                   rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
                   if(bg==nbGblk) rownbm    = plasma_ceildiv((N-(firstcolj)),NB);  // last blk has size=1 used for complex to handle A(N,N-1)
                   for (m = rownbm; m>0; m--)
                   {
                       vlen = 0;
                       vnb  = 0;
                       colj      = (bg-1)*Vblksiz; // for k=0;I compute the fst and then can remove it from the loop
                       fst       = (rownbm -m)*NB+colj +1;
                       for (k=0; k<Vblksiz; k++)
                       {
                           colj     = (bg-1)*Vblksiz + k;
                           st       = (rownbm -m)*NB+colj +1;
                           ed       = min(st+NB-1,N-1);
                           if(st>ed)break;
                           if((st==ed)&&(colj!=N-2))break;
                           vlen=ed-fst+1;
                           vnb=k+1;
                       }        
                       colst     = (bg-1)*Vblksiz;
                       findVTpos(N,NB,Vblksiz,colst,fst, &vpos, &taupos, &tpos, &blkid);
                       //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d\n",bg,m,vlen, vnb,colst+1,vpos+1,taupos+1);
                       
                       if(LOGQ) core_event_startblg(my_core_id);
                       if((vlen>0)&&(vnb>0)){
                           //lapackf77_zungqr( "L", "N", &vlen, &corelen, &vnb, V(vpos), &LDV, TAU(taupos), E(fst,corest), &LDE,  WORK, &LWORK, &INFO );
                           //ZUNMQR_BLG( "L", "N", &vlen, &corelen, &vnb, &NB, V(vpos), &LDV, TAU(taupos), E(fst,corest), &LDE,  WORK, &LWORK, &INFO );
                           if(WANTZ==1){ 
                               st  = max(colst,corest);           
                               len = corelen - (st-corest);    
                               lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &len, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,st), &LDE,  WORK, &len); 
                           }else{
                               lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &corelen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,corest), &LDE,  WORK, &corelen); // corelen for WORK was colpercore before 
                           }
                       }
                       if(INFO!=0) 
                               printf("ERROR ZUNMQR INFO %d\n", (int) INFO);
                       if(LOGQ) {
                           core_event_endblg(my_core_id);
                           core_log_eventblg(0xff0000, my_core_id);
                       }                
                   }
                }
            }else if (SIDE=='R'){
            if(versionR==91){		    
                for (bg =1; bg<=nbGblk; bg++)
                {
                   firstcolj = (bg-1)*Vblksiz + 1;
                   rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
                   if(bg==nbGblk) rownbm    = plasma_ceildiv((N-(firstcolj)),NB);  // last blk has size=1 used for complex to handle A(N,N-1)
                   for (m = 1; m<=rownbm; m++)
                   {
                       vlen = 0;
                       vnb  = 0;
                       // for k=0;I compute the fst and then can remove it from the loop
                       colj     = (bg-1)*Vblksiz; 
                       fst      = (rownbm -m)*NB+colj +1;
                       for (k=0; k<Vblksiz; k++)
                       {
                           colj     = (bg-1)*Vblksiz + k;
                           st       = (rownbm -m)*NB+colj +1;
                           ed       = min(st+NB-1,N-1);
                           if(st>ed)break;
                           if((st==ed)&&(colj!=N-2))break;
                           vlen=ed-fst+1;
                           vnb=k+1;
                       }        
                       colj     = (bg-1)*Vblksiz;
                       findVTpos(N,NB,Vblksiz,colj,fst, &vpos, &taupos, &tpos, &blkid);
                       //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d\n",bg,m,vlen, vnb,colj,vpos,taupos);
                       if((vlen>0)&&(vnb>0))
                           lapackf77_zlarfb( "R", "N", "F", "C", &corelen, &vlen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(corest,fst), &LDE,  WORK, &corelen);       
                   }
                }
                }else if(versionR==92){
                 rownbm    = plasma_ceildiv((N-1),NB);
                 for (m = 1; m<=rownbm; m++)
                 {
                    ncolinvolvd = min(N-1, m*NB);
                    avai_blksiz=min(Vblksiz,ncolinvolvd);
                    nbgr = plasma_ceildiv(ncolinvolvd,avai_blksiz);
                    for (n = 1; n<=nbgr; n++)
                    {
                        vlen = 0;
                        vnb  = 0;
                        cur_blksiz = min(ncolinvolvd-(n-1)*avai_blksiz, avai_blksiz);
                        colst = (n-1)*avai_blksiz;
                        coled = colst + cur_blksiz -1;
                        fst   = (rownbm -m)*NB+colst +1;
                        for (colj=colst; colj<=coled; colj++)
                        {
                            st       = (rownbm -m)*NB+colj +1;
                            ed       = min(st+NB-1,N-1);
                            if(st>ed)break;
                            if((st==ed)&&(colj!=N-2))break;
                            vlen=ed-fst+1;
                            vnb=vnb+1;
                        }    
                        findVTpos(N,NB,Vblksiz,colst,fst, &vpos, &taupos, &tpos, &blkid);
                        if((vlen>0)&&(vnb>0)){
                                   lapackf77_zlarfb( "R", "N", "F", "C", &corelen, &vlen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(corest,fst), &LDE,  WORK, &corelen);       
                       }  
                    }
                 }
            }
            }else{
                    printf("ERROR SIDE %d\n",SIDE);
            }
        } // END my_core_id=coreid
    } // END loop over the chunk


}
#undef E
#undef V
#undef TAU
#undef T
////////////////////////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////////////////////////////
#define E(m,n)   &(E[(m) + LDE*(n)])
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)     &(T[(m)])
void tile_bulge_applyQ_parallel2(int my_core_id)
{
    magma_int_t INFO;
    magma_int_t cores_num  = core_in_all.cores_num;
    magmaDoubleComplex *T      = core_in_all.T;
    magmaDoubleComplex *E      = core_in_all.E;
    magmaDoubleComplex *V      = core_in_all.V;
    magmaDoubleComplex *TAU    = core_in_all.TAU;
    magma_int_t  N         = core_in_all.N;
    magma_int_t  NB        = core_in_all.NB;
    magma_int_t  Vblksiz   = core_in_all.Vblksiz;
    magma_int_t  LDE       = core_in_all.LDE;
    char SIDE      = core_in_all.SIDE;
    magma_int_t  WANTZ     = core_in_all.WANTZ;

    //%===========================
    //%   local variables
    //%===========================
    magma_int_t LDT,LDV,blklen,firstcolj;
    magma_int_t bg, nbGblk,rownbm, k, m, n;
    magma_int_t st,ed,fst,vlen,vnb,colj,len;
    magma_int_t blkid, vpos,taupos,tpos;
    magmaDoubleComplex *WORK;
    magma_int_t LWORK;
    magma_int_t  cur_blksiz,avai_blksiz, ncolinvolvd;
    magma_int_t  nbgr, colst, coled, versionL;
    magma_int_t chunkid,nbchunk,colpercore,corest,corelen;
    magma_int_t coreid,mycoresnb,maxrequiredcores;

    INFO=0;    
    versionL = 113;
    LDT     = Vblksiz;
    LDV     = NB+Vblksiz-1;    
    blklen  = LDV*Vblksiz;
    nbGblk  = plasma_ceildiv((N-1),Vblksiz);
    //LWORK   = 2*N*max(Vblksiz,64);
    //WORK    = (magmaDoubleComplex *) malloc (LWORK*sizeof(magmaDoubleComplex));


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
     *
     * version v2: for each V_k and T_k thread t apply those V_k and 
     *                    T_k to E_k for all its chunck, then move to V_k+1 
     *                    T_k+1 and apply them to E_k+1 on all the chunck, 
     *                    and so on. the difference is that, thread keep V_k
     *                    and T_K while move over the E_k.
     *                    both version look like similar in perf.                   
     * */
    colpercore = min(NB,120); //colpercore = N make the code sequential running on thread=0;
    //colpercore =  N/cores_num; 

    colpercore =  plasma_ceildiv(N,cores_num);
    if(colpercore>1000)
        colpercore =  plasma_ceildiv(colpercore,10);
    else if(colpercore>800)
        colpercore =  plasma_ceildiv(colpercore,8);
    else if(colpercore>600)
        colpercore =  plasma_ceildiv(colpercore,6);
    else if(colpercore>400)
        colpercore =  plasma_ceildiv(colpercore,4);
    else if(colpercore>200)
        colpercore =  plasma_ceildiv(colpercore,2);
    if(colpercore>200)colpercore=120;





    LWORK   = 2*colpercore*max(Vblksiz,64);
    //LWORK   = 2*N*max(Vblksiz,64);
    WORK    = (magmaDoubleComplex *) malloc (LWORK*sizeof(magmaDoubleComplex));


    nbchunk    =  plasma_ceildiv(N,colpercore);

    mycoresnb = cores_num;
    maxrequiredcores = nbchunk;
    if(maxrequiredcores<1)maxrequiredcores=1;
    if(mycoresnb > maxrequiredcores)
    {
        if(my_core_id==0)printf("==================================================================================\n");
        if(my_core_id==0)printf("  WARNING only %3d threads are required to run this test optimizing cache reuse\n", (int) maxrequiredcores);
        if(my_core_id==0)printf("==================================================================================\n");
        mycoresnb = maxrequiredcores;
    }

    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            Also E is splitten by row meaning each apply consist in a block of row (horizontal block) */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n 
     *            Also E is splitten by col meaning each apply consist in a block of col (vertical block) */

     /* WANTZ = 1 meaning E is IDENTITY so form Q using optimized update. 
      *         So we use the reverse order from small q to large one, 
      *         so from q_n to q_1 so Left update to Identity.
      *         Use versionL 113 because in 114 we need to update the whole matrix and not in icreasing order.
      * WANTZ = 2 meaning E is a full matrix and need to be updated from Left or Right so use normal update
      * */
    if(WANTZ==1) 
    {
        versionL=113;
        SIDE='L';
    }

    //printf("  ENTERING FUNCTION APPLY Q_v115: same as 113(L) or 114(L) or 93(R)"\n);
    if(my_core_id==0) {
        printf("  APPLY Q_v2   parallel with threads %d   nbchunk %d  colpercore %d  N %d   NB %d   Vblksiz %d SIDE %c versionL %d\n",
               (int) cores_num, (int) nbchunk, (int) colpercore, (int) N, (int) NB, (int) Vblksiz, (int) SIDE, (int) versionL);
    }
    
    //printf("mycore id %d voici nbchunk %d  chunkid %d  coreid %d, corest %d, corelen %d\n",my_core_id,nbchunk, chunkid, coreid, corest, corelen);
    if(SIDE=='L'){
        if(versionL==113){            
            for (bg = nbGblk; bg>0; bg--)
            {
               firstcolj = (bg-1)*Vblksiz + 1;
               rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
               if(bg==nbGblk) rownbm    = plasma_ceildiv((N-(firstcolj)),NB);  // last blk has size=1 used for complex to handle A(N,N-1)
               for (m = rownbm; m>0; m--)
               {
                   vlen = 0;
                   vnb  = 0;
                   colj      = (bg-1)*Vblksiz; // for k=0;I compute the fst and then can remove it from the loop
                   fst       = (rownbm -m)*NB+colj +1;
                   for (k=0; k<Vblksiz; k++)
                   {
                       colj     = (bg-1)*Vblksiz + k;
                       st       = (rownbm -m)*NB+colj +1;
                       ed       = min(st+NB-1,N-1);
                       if(st>ed)break;
                       if((st==ed)&&(colj!=N-2))break;
                       vlen=ed-fst+1;
                       vnb=k+1;
                   }        
                   colst     = (bg-1)*Vblksiz;
                   findVTpos(N,NB,Vblksiz,colst,fst, &vpos, &taupos, &tpos, &blkid);
                   //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d\n",bg,m,vlen, vnb,colst+1,vpos+1,taupos+1);
       
                   if((vlen>0)&&(vnb>0)){
                       if(LOGQ) core_event_startblg(my_core_id);
                       for (chunkid = 0; chunkid<nbchunk; chunkid++)
                       {
                           coreid  = chunkid%mycoresnb;
                           corest  = chunkid*colpercore;
                           corelen = min(colpercore, (N-(chunkid*colpercore)));
                           if(my_core_id==coreid)
                           {   
                               if(WANTZ==1){ 
                                       st  = max(colst,corest);           
                                       len = corelen - (st-corest);    
                                       lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &len, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,st), &LDE,  WORK, &len); 
                                   }else{
                                       lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &corelen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,corest), &LDE,  WORK, &corelen); // corelen for WORK was colpercore before 
                                   } // end WANTZ
                           } // END my_core_id=coreid
                       } // END loop over the chunk
                       if(LOGQ) {
                           core_event_endblg(my_core_id);
                           core_log_eventblg(0xff0000, my_core_id);
                       }
                   } // end if vlen and vnb
               } // end for m:rowmnb
            } // end for bg 
        }else if(versionL==114){
            rownbm    = plasma_ceildiv((N-1),NB);
            for (m = rownbm; m>0; m--)
            {
                ncolinvolvd = min(N-1, m*NB);
                avai_blksiz=min(Vblksiz,ncolinvolvd);
                nbgr = plasma_ceildiv(ncolinvolvd,avai_blksiz);
                for (n = nbgr; n>0; n--)
                {
                    vlen = 0;
                    vnb  = 0;
                    cur_blksiz = min(ncolinvolvd-(n-1)*avai_blksiz, avai_blksiz);
                    colst = (n-1)*avai_blksiz;
                    coled = colst + cur_blksiz -1;
                    fst   = (rownbm -m)*NB+colst +1;
                    for (colj=colst; colj<=coled; colj++)
                    {
                        st       = (rownbm -m)*NB+colj +1;
                        ed       = min(st+NB-1,N-1);
                        if(st>ed)break;
                        if((st==ed)&&(colj!=N-2))break;
                        vlen=ed-fst+1;
                        vnb=vnb+1;
                    }        
                    findVTpos(N,NB,Vblksiz,colst,fst, &vpos, &taupos, &tpos, &blkid);
                    //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d\n",bg,m,vlen, vnb,colst+1,vpos+1,taupos+1);
                    if((vlen>0)&&(vnb>0)){
                        for (chunkid = 0; chunkid<nbchunk; chunkid++)
                        {
                            coreid  = chunkid%mycoresnb;
                            corest  = chunkid*colpercore;
                            corelen = min(colpercore, (N-(chunkid*colpercore)));
                            if(my_core_id==coreid)
                            {                 
                                lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &corelen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,corest), &LDE,  WORK, &corelen);    
                            } // END my_core_id=coreid
                        } // END loop over the chunk
                    } // end if vlen and vnb
                } // end for n = nbgr
            } // end for m=rowmnb
        }
    }else if (SIDE=='R'){
        for (bg =1; bg<=nbGblk; bg++)
        {
           firstcolj = (bg-1)*Vblksiz + 1;
           rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
           if(bg==nbGblk) rownbm    = plasma_ceildiv((N-(firstcolj)),NB);  // last blk has size=1 used for complex to handle A(N,N-1)
           for (m = 1; m<=rownbm; m++)
           {
               vlen = 0;
               vnb  = 0;
               // for k=0;I compute the fst and then can remove it from the loop
               colj     = (bg-1)*Vblksiz; 
               fst      = (rownbm -m)*NB+colj +1;
               for (k=0; k<Vblksiz; k++)
               {
                   colj     = (bg-1)*Vblksiz + k;
                   st       = (rownbm -m)*NB+colj +1;
                   ed       = min(st+NB-1,N-1);
                   if(st>ed)break;
                   if((st==ed)&&(colj!=N-2))break;
                   vlen=ed-fst+1;
                   vnb=k+1;
               }        
               colj     = (bg-1)*Vblksiz;
               findVTpos(N,NB,Vblksiz,colj,fst, &vpos, &taupos, &tpos, &blkid);
               //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d\n",bg,m,vlen, vnb,colst+1,vpos+1,taupos+1);
               if((vlen>0)&&(vnb>0)){
                    for (chunkid = 0; chunkid<nbchunk; chunkid++)
                    {
                        coreid  = chunkid%mycoresnb;
                        corest  = chunkid*colpercore;
                        corelen = min(colpercore, (N-(chunkid*colpercore)));
                        if(my_core_id==coreid)
                        {            
                            lapackf77_zlarfb( "R", "N", "F", "C", &corelen, &vlen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(corest,fst), &LDE,  WORK, &corelen);       
                        } // END my_core_id=coreid
                    } // END loop over the chunk
                } // end if vlen and vnb
           } // end for m=rowmnb
        } // end for bg
    }else{
        printf("ERROR SIDE %d\n",SIDE);
    }


}
#undef E
#undef V
#undef TAU
#undef T
////////////////////////////////////////////////////////////////////////////////////////////////////








