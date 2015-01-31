/* 
 * Preprocessing step of Communication Avoiding LU factorization
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
*/
#ifdef _WIN32    
    #include "sched.h"
    #include "pthread.h"
    #pragma comment(lib, "pthreadVC2.lib")
#else
    #include "pthread.h"    
//    #define PTHREAD_MUTEX_INITIALIZER ((pthread_mutex_t) -1)
//    #define PTHREAD_COND_INITIALIZER ((pthread_cond_t) -1)
#endif

#include "common_magma.h"

#include "magma_amc.h"

#include "core_d.h"


//#include "magma_amc_utils.h"

pthread_mutex_t *mutex_tree; /*Mutex de modification dans l'arbre*/

/* Synchronisation de processus */
int *tree_step; /* A quel etape de l'arbre dans tslu chaque thread se trouve*/
int panel_step = 0;

/*tslu buffer*/
double *shared_Tblock;
double *shared_Twork;
int shared_LDT;
int *shared_IP;
int *shared_IPwork;

void core_dtslu_alloc(int nbThreads, int m, int nb)
{
    /*Local allocation for tslu*/

if(!(tree_step = (int*) malloc(nbThreads*sizeof(int)))) magma_amc_abort("Memory allocation failed for tree_step"); /* private */

if(!(mutex_tree = (pthread_mutex_t*) malloc(nbThreads*sizeof(pthread_mutex_t)))) magma_amc_abort("Memory allocation failed for mutex_tree");

shared_LDT = m; /*possible optimization*/

shared_LDT = max(shared_LDT,2*nb); /*At least 2 blocks needed for tslu*/

    /* tslu buffer */
if(!(shared_Tblock = (double*)    malloc(shared_LDT*nb*sizeof(double)))) magma_amc_abort("Memory allocation failed for shared_Tblock");

if(!(shared_Twork = (double*) malloc(shared_LDT*nb*sizeof(double)))) magma_amc_abort("Memory allocation failed for shared_Twork"); 

if(!(shared_IPwork =(int*)    malloc(shared_LDT*sizeof(int)))) magma_amc_abort("Memory allocation failed for shared_IPwork"); //descA[tid].M

if(!(shared_IP =  (int*)    malloc(shared_LDT*sizeof(int)))) magma_amc_abort("Memory allocation failed for shared_IP");

}

void core_dtslu_init(int nbThreads)
{
int i;

panel_step = 0;

for(i=0;i<nbThreads;i++) tree_step[i]=-1;    

for(i=0;i<nbThreads;i++) pthread_mutex_init (&(mutex_tree[i]), NULL);
}

void core_dtslu_free()
{

    free(tree_step);

    free(mutex_tree);
    #ifdef USE_COND_WAIT

    free(cond_tree);
    #endif

    free(shared_Tblock);
    free(shared_Twork);
    free(shared_IP);
    free(shared_IPwork); 
}

/*Update the progress of tslu if a tree is used, return -1, if thread has no target in the tree*/
void update_tslu_tree_progress(int tslu_num, int level, int lastNode, int *target, int *done)
{
    int min_thr;

    /*Determine thread which I exchange data*/
    *target = -1;*done=0;
    if(tslu_num % (2*level) == 0)
    {
        if(tslu_num+level < lastNode) *target = tslu_num + level;
        else if(tslu_num !=0) *target = tslu_num;
        else *done =1;
    }
    else
        if(tslu_num - level >=0) *target = tslu_num - level;


    min_thr = min(tslu_num, *target);
    if(min_thr == -1) min_thr = tslu_num;

    pthread_mutex_lock (&(mutex_tree[min_thr]));

        /*update my level*/
        tree_step[tslu_num] = level;

        /*Check for the next block*/
        if(*target != -1 && tree_step[tslu_num] == tree_step[*target])
        {
            /*Should go to the next stage*/
            
        }
        else
            *target = -1;
    pthread_mutex_unlock (&(mutex_tree[min_thr]));

}

/*Do the preprocessing step of TSLU on a matrix A to factorize*/
//void core_dtslu(global_data_t *data, int tid, int jblock, int locNum)
void core_dtslu(int M, int nb, double *A, int LDA, int *IPIV, int *iinfo, int num_threads, int locNum)
{

//int bsize = data->bsize;
//int csize = data->csize; 
//int first_bsize = data->first_bsize;

//int last_csize = data->last_csize;

int nbBlock; // = data->BM - jblock;
//int M = data->M;
//int N = data->N;
//double *A = data->A;
//int LDA = data->LDA;

/*Typo*/
double *Tblock = shared_Tblock;
double *Twork = shared_Twork;
int LDT = shared_LDT;
int *IPwork = shared_IPwork;
int *IP = shared_IP;
//int *perm = data->perm ;
int Pr = num_threads;


int last_rsize; // = data->last_rsize;
//int Pc = data->Pc;
//int panel_NB; /*Panel block size*/
int panel_MB;


int nrowPblock;     /* Number of rows for each process*/

int locM;        /* Local position of the thread in the shared column block buffer */

int ONE=1;

//double dZERO=0.0, dMONE=-1.0; /* Parameter */

int PrxNB;
int twoNB;


//int pos;

int j,k,l;
int IINFO;
int aux;
int target;

int done = 0;
int level, lastNode;
int tslu_num;
//int last_row;
int j_MB;


*iinfo = 0;

if(M%nb==0) nbBlock = (int) M / nb;
else nbBlock = 1 + (int) M/nb;


/*Quick return*/
if(locNum >= nbBlock)
{
    return;    
}

/*Compute the size of the last block*/

if((M % nb)==0) last_rsize = nb;
else last_rsize = M % nb;


//pos = jblock*bsize;



//nbBlock = s_div(M-pos, nb);
//last_row = M - pos - (nbBlock-1)*nb;



/* Determine my number of rows and my local position in A */

#ifdef NOPIV
nrowPblock = nb; //nbBlock*bsize; /*Debug*/
#else
nrowPblock = numBlock2p(locNum, nbBlock, Pr)*nb;
#endif

locM = indexBlock2p(locNum, nbBlock, Pr)*nb;

#ifdef NOPIV
lastNode = 1;
#else
lastNode = min(Pr,nbBlock);
#endif

if(locNum==lastNode-1)
{
    nrowPblock = nrowPblock - nb + last_rsize;
}

//nb = bsize;




/*Maximun number of rows I will use for the reduction operation*/
panel_MB = min(nrowPblock, nb);


PrxNB = Pr*nb;
twoNB = 2*nb;

/* 
*    0. Copy my original blocks in my current position in the buffer
*/
    
#ifdef DEBUG9
printf("[%d] locNum:%d, locPosition: %d, nbrow: %d\n",tid,locNum, locM, nrowPblock);
#endif
//ca_dbg_printMat(M, data->N, A, LDA, "Global A (Before Panel)");

/*Copy my original ros in the buffer*/

dlacpy( "F", &nrowPblock, &nb, &(A[locM]), &LDA, &(Tblock[locM]),&LDT);

/* Get original indices of my rows */
for(l=0;l<nrowPblock;l++) IP[locM+l] = IPIV[locM+l];



if(nrowPblock > nb || nbBlock==1) /*If I have more than a block then do the first reduce*/
{
    
    /* keep a copy of my originals blocks */
    dlacpy( "F", &nrowPblock, &nb, &(Tblock[locM]), &LDT, &(Twork[locM]), &LDT);

    /* 
     *     1. Apply LU factorization on my local block
     */

    #if (dbglevel>=10)
        ca_dbg_printMat(nrowPblock, nb, &(Twork[locM]), LDT, "Tblock(Before dgetf2)");
    #endif

    /*Factorize my block*/

        #ifdef USE_DGETF2
        dgetf2_(&nrowPblock, &nb, &(Twork[locM]), &LDT, &(IPwork[locM]), &IINFO);
        #else 
            #ifdef USE_RGETF2
        rgetf2_(&nrowPblock, &nb, &(Twork[locM]), &LDT, &(IPwork[locM]), &IINFO);
            #else
        dgetrf(&nrowPblock, &nb, &(Twork[locM]), &LDT, &(IPwork[locM]), &IINFO);
            #endif
        #endif


    #if (dbglevel>=10)            
                ca_dbg_printMat(nrowPblock, nb, &(Twork[locM]), LDT, "Tblock(After dgetf2)");        
                //printf("locNum:%d IPwork: ",locNum);
                //for(l=0;l<nb;l++) printf("%d ",IPwork[locM+l]);printf("\n");
                ca_dbg_iprintVec(nb, &(IPwork[locM]), "IPwork");
    #endif
            
    /* Test the singularity of the block that has been factorized*/
    /*NOTE: singular blocks do not mean that the input matrix is singular. We just need to skip singular local block. */
    /*The factorization failed only if the whole block is singular.*/
        if(IINFO!=0) 
        {
#ifndef NO_WARNING
             printf("locNum:%d ***Singular block at M:%d, INFO=%d\n", locNum, M, IINFO);
            /*No need to avoid factorization, just skip the block*/
#endif
             *iinfo += IINFO;
        }

    /*Apply permutation on my original rows*/
        
        for(l=0;l<panel_MB;l++)
        {
            j= IPwork[locM+l]-1;
            /* Permute rows l and j*/
            aux = IP[locM+l]; 
            IP[locM+l]= IP[locM+j];
            IP[locM+j]=aux;
        }

    #ifdef DEBUG9
    #if (dbglevel>=10)
        ca_dbg_printMat(nrowPblock,nb,&(Twork[locM]),LDT,"Tblock (After repartition)");
    #endif
                printf("[%d] locNum:%d Swapping originals rows\n", tid, locNum);
    #endif    

            
    /* Apply Permutation on my originals rows */
                     
    if(Pr>1)/*TODO: Skip if Pr == 1, because factorisation is complete */
        dlaswp(&nb, &(Tblock[locM]), &LDT, &ONE, &panel_MB, &(IPwork[locM]), &ONE );

                    //ca_dbg_printMat(nrowPblock[locNum],csize,Pblock[locNum],LDP,"Pblock (After LU permutation)");
    #if (dbglevel>=10)
//                ca_dbg_printMat(nrowPblock, nb, &(Tblock[locM]), LDT, "Tblock(After swap)");
    #endif

    /*At this step pivot candidate is in Tblock (check?)*/
                
}
            

/*PHASE 2: reduction tree*/



/*this step use a binary tree for synchronisation*/

            tslu_num = locNum;
            level = 1;

            /*Update progress and test for the next block*/
            update_tslu_tree_progress(tslu_num, level, lastNode, &target, &done);

            while(target != -1)
            {

                if(tslu_num != target)
                {

                    if(tslu_num>target)
                    {
                        /*Make sure the master thread has the higher id*/
#ifdef DEBUG9                    
                    printf("[%d] tsluNum:%d exchange tsluNum with target:%d\n",tid, tslu_num, target);
#endif
                        k=target;
                        target=tslu_num;
                        tslu_num=k;                    
                    }

                    
                    /*    Merge first two blocks of the two threads P(i) and P(i+level) */
                     

#ifdef DEBUG9                    
                    printf("[%d] tsluNum:%d merge with my_target:%d, my step:%d target_step:%d\n",tid, tslu_num, target, tree_step[target], tree_step[target]);
#endif
                    /* compute my local position */

                    locM = indexBlock2p(tslu_num, nbBlock, Pr)*nb;

                    panel_MB = nb;

                    /*
                    //if(jblock==0 && locM>0) locM     = locM - bsize + first_bsize;
                    panel_MB = nbBloc2p(tslu_num, nbBlock, Pr)*bsize;


                    if(tslu_num==0) 
                        panel_MB = panel_MB - nb + last_row;
                    else
                        locM = locM - nb + last_row;

                    if(level==1)
                        panel_MB = MIN(panel_MB,nb);
                    else
                        panel_MB = nb;
                    */


                    dlacpy( "F", &panel_MB, &nb, &(Tblock[locM]), &LDT, &(Twork[locM]), &LDT);
                

                    /* determine local position of the thread which who I will exchange data*/

                    j = indexBlock2p(target, nbBlock, Pr)*nb;

                    j_MB = nb;

                    if(target==lastNode-1 ) //&& level==1
                    {
                        /*the last bloc may be of incomplete size*/
                        j_MB = numBlock2p(target, nbBlock, Pr)*nb;

                        j_MB = j_MB - nb + last_rsize;

                        j_MB = min(j_MB, nb);
                    }

#if (dbglevel>=10)        
//                    ca_dbg_printMat(panel_MB, nb, &(Tblock[locM]), LDT, "My Tblock");    
//                    ca_dbg_printMat(j_MB, nb, &(Tblock[j]), LDT, "Target Tblock");
#endif

                    /*Merge two blocks*/
                    dlacpy( "F", &j_MB, &nb, &(Tblock[j]), &LDT, &(Twork[locM+panel_MB]), &LDT);
                


                    /* Merge also two permutation vector information */
        
                    for(l=0;l<j_MB;l++) IP[locM+panel_MB+l]=IP[j+l];

#if (dbglevel>=10)
//                    printf("[%d]Merge::IP: ",tslu_Num);
                    //for(l=0;l<twoNB;l++) printf("%d ",IP[locM+l]);
#endif
                    panel_MB = panel_MB + j_MB;
                    /* Save originals rows */
    
                    dlacpy( "F", &panel_MB, &nb, &(Twork[locM]), &LDT, &(Tblock[locM]), &LDT); 
#if (dbglevel>=10)    
//                    ca_dbg_printMat(panel_MB, nb, &(Tblock[locM]), LDT, "Merge::block(Before dgetf2)");
                    //printf("[%d] locNum:%d Merge::IPwork(before dgetf2): ", tid, tslu_num);
                    //for(l=0;l<twoNB;l++) printf("%d ",IP[locM+l]);
#endif

                    
                    /*    Apply LU on the merged blocks */

#ifdef USE_DGETF2
                    dgetf2_(&panel_MB, &nb, &(Twork[locM]), &LDT, &(IPwork[locM]), &IINFO);
#else 
    #ifdef USE_RGETF2
                    rgetf2_(&panel_MB, &nb, &(Twork[locM]), &LDT, &(IPwork[locM]), &IINFO);
    #else
                    dgetrf(&panel_MB, &nb, &(Twork[locM]), &LDT, &(IPwork[locM]), &IINFO);
    #endif
#endif    

#if (dbglevel>=10)                
                    //ca_dbg_printMat(twoNB, nb, &(Twork[locM]), LDT, "Twork(After dgetf2)");
                    //printf("[%d] locNum:%d IPwork(After dgetf2): ", tid, tslu_num);
                    //for(l=0;l<twoNB;l++) printf("%d ",IPwork[locM+l]);
#endif

                    /* Test for singularity */

                    if(IINFO!=0) 
                    {
#ifndef NO_WARNING
                        printf("locNum:%d ***Singular block at M:%d, INFO=%d\n", tslu_num, M, IINFO);    
#endif
                        *iinfo += IINFO;
                    }
                        

                    panel_MB = min(panel_MB,nb);

                    /* apply Permutation on my original rows */
                
                    dlaswp(&nb, &(Tblock[locM]), &LDT, &ONE, &panel_MB, &(IPwork[locM]), &ONE );
#if (dbglevel>=10)                            
                    //for(l=0;l<nb;l++) printf("[%d] locNum:%d Merge:: LU_IPwork[%d]=%d\n", tid, tslu_num,locM+l,IPwork[locM+l]-1);
#endif
#if (dbglevel>=10)
                    //ca_dbg_printMat(nb, nb, &(Tblock[locM]), LDT, "Tblock(After swap)");
#endif
                /*  Update local permutation vector information */
                        
                    for(l=0;l<min(panel_MB,nb);l++)
                    {
                        j= IPwork[locM+l]-1;

                        /* Permute rows l and j*/
                        aux = IP[locM+l];
                        IP[locM+l]= IP[locM+j];
                        IP[locM+j]=aux;
                    }
    
    
                } /* else thread <locNum> wait the next step */ 
                
            //}

            /*Go to the next level*/
                level= level*2;    

                update_tslu_tree_progress(tslu_num, level, lastNode, &target, &done);        

            }


            if(done)
            {
                /* Update global permutation vector perm */
#ifdef DEBUG9                    
                printf("[%d] locNum:%d, tsluNum:%d update Global perm(step:%d)\n",tid, locNum,tslu_num, jblock);
#endif

                    for(l=0;l<panel_MB;l++) //nb
                    {
                        k=l;
                        while(k<M && IPIV[k]!=IP[l]) k++;
                    
                        if(k>=M) 
                        {
                            /*This should never happen*/
                            printf("locNum:%d IP[%d]=%d not found in perm\n", locNum, l, IP[l]);            
                            magma_amc_abort("***Error when computing permutation vector ! ");// Can never occur 
                            
                        }
                
                        // Permute rows l+pos and k 
                        aux = IPIV[k];
                        IPIV[k]=IPIV[l];
                        IPIV[l]=aux;
                    
                        // update local permutation vector
                        IP[l] = k+1;
                    
                        IPIV[l]=k+1;
                    }

                        /*Save permutation vector of this step such as to be used for left swapping */
                    /*    
                        for(l=0;l<panel_MB;l++) //NB
                        {
                            data->IP_step[jblock][l]=IP[l]; 
                        }
                        */

#ifndef NOPIV
                        /*Apply permutation on the original rows*/                    
                        if(Pr>1)/*NOTE: Skip if only one proc */
                            //do_localswap(data, jblock, jblock);
                            dlaswp_(&nb, A, &LDA, &ONE, &nb, IP, &ONE );
#endif

#if (dbglevel>=10)
                        /*
                        printf("[%d] IP: ",tid);
                        for(l=0;l<nb;l++) printf("%d ",data->IP_step[jblock][l]);
                        printf("\n");*/

                        //printf("[%d] IPIV: ",tslu_num);
                        //for(l=0;l<M;l++) printf("%d ",IPIV[l]);
                        ca_dbg_iprintVec(M, IPIV, "IPIV");
                        printf("\n");

                        ca_dbg_printMat(M, nb, A, LDA, "Ablock(After swap)");
#endif
                        /* Copy LU of the square block in A */
                         dlacpy( "F", &panel_MB, &nb, &(Twork[0]), &LDT, A, &LDA);

                        if((M==8) && IPIV[0]!=6)
                            printf("Breakpoint enabled\n");


                        
#if (dbglevel>=10)
                        
                        ca_dbg_printMat(M, nb, A, LDA, "Ablock(After copying LU)");

                        //if(jblock==1)
                        //    ca_abort("Stop here for debug");
#endif

                        /*Reset the tree for the next level*/
                        for(l=0;l<Pr;l++) tree_step[l]=-1;
                
                        /*Enabled the next Panel*/
                        //update_progress('P', jblock, 0, 0, 1,1);
            }




#ifdef DEBUG9

            printf("locNum:%d tslu_num:%d Finish tslu.\n", locNum, tslu_num);

#endif

}
