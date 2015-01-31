/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zhemm_mgpu.cpp normal z -> s, Fri Jan 30 19:00:10 2015
       @author Mark Gates
       @author Azzam Haidar
       
       This still has poor performance. Work in progress.
*/
#include "common_magma.h"
#include "magma_bulge.h"
//#include "trace.h"

extern "C"
void magmablas_ssymm_mgpu_com(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaFloat_ptr dB[],    magma_int_t lddb,
    float beta,
    magmaFloat_ptr dC[],    magma_int_t lddc,
    magmaFloat_ptr dwork[], magma_int_t dworksiz,
    float *C,          magma_int_t ldc,
    float *work[],     magma_int_t worksiz,  // TODO unused
    magma_int_t ngpu, magma_int_t nb, 
    magma_queue_t queues[][20], magma_int_t nqueue, 
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents, 
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx )
{
    #define dA(dev, i, j) (dA[dev] + (i) + (j)*ldda)
    #define dB(dev, i, j) (dB[dev] + (i) + (j)*lddb)
    #define dC(dev, i, j) (dC[dev] + (i) + (j)*lddc)
    #define dwork(dev, i, j) (dwork[dev] + (i) + (j)*lddwork)
    #define C(i, j) (C + (i) + (j)*ldc)
    //printf("####################################################\n");
    //printf("                      start ssymm                   \n");
    //printf("####################################################\n");
   
    if ( side != MagmaLeft || uplo != MagmaLower ) {
        fprintf( stderr, "%s: only Left Lower implemented\n", __func__ );
    }
    
    assert( ldda >= m );
    assert( lddb >= m );
    assert( lddc >= m );
    assert( nqueue >= ngpu );
    assert( nbevents >= ngpu*ngpu );
   
    
    float c_one  = MAGMA_S_ONE;

    magmaFloat_ptr dwork2[MagmaMaxGPUs];


    magma_int_t maxgsize    = n*m;
    magma_int_t lddwork = lddc;
    magma_int_t ldwork  = m;
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        dwork2[dev] = dwork[dev]+n*lddwork;  // size of dwork2 is maxgsize*ngpu
    }
    assert( dworksiz >= (n*lddwork+maxgsize*ngpu) );
    assert( worksiz  >= (n*ldwork) );

        
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_t cstream;
    magmablasGetKernelStream(&cstream);


    magma_int_t dev, devperm, myblk, mycolsize, myblkoffst;
    magma_int_t gmaster;
    magma_int_t masterdev, lcdev, lccolsize, myngpu;

    magma_int_t stdev       = (offset/nb)%ngpu;  
    magma_int_t blockoffset = offset % nb;  
    magma_int_t fstblksiz   = 0;
    if(blockoffset>0){
        fstblksiz   = min(m, (nb - blockoffset));
    }
    //magma_int_t nbblk       = magma_ceildiv(m, nb);
    magma_int_t nbblk       = magma_ceildiv((m+blockoffset), nb);
    magma_int_t remm        = m- fstblksiz;
    magma_int_t nbblkoffst  = offset/nb;


    magma_int_t nblstblks = -1;
    magma_int_t devlstblk = -1;
    magma_int_t lstblksiz = remm%nb;
    if(lstblksiz>0){
        nblstblks = nbblk%ngpu;
        devlstblk = (nblstblks-1+ngpu)%ngpu;
    }

    magma_int_t nbcmplxactive =  0;
    magma_int_t cmplxisactive[MagmaMaxGPUs];
    magma_int_t gpuisactive[MagmaMaxGPUs];
    memset(gpuisactive, 0, MagmaMaxGPUs*sizeof(magma_int_t));
    memset(cmplxisactive, 0, MagmaMaxGPUs*sizeof(magma_int_t));


    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magmablasSetKernelStream( queues[ dev ][ 0 ] );
        cudaMemset(dwork(dev,0,0), 0, (lddwork)*(n)*sizeof(float) );
        // put all dC on all dev to 0 except the one which
        // hold i==0 because this one has to multiply by beta.
        if(dev!=stdev){
           cudaMemset(dC(dev,0,0), 0, (lddc)*(n)*sizeof(float) );
        }
    }

    magma_int_t newoffset = offset;
    // 1. symmetrize
    if(blockoffset>0){
        newoffset  = offset+fstblksiz; // newoffset is adjusted over nb
        magma_int_t myblkoffst = (nbblkoffst/ngpu)+(nbblkoffst%ngpu > stdev?1:0);
        //printf("STDEV %d  voici offset %d remm %d   myblockoffset %d    siz %d \n", stdev, offset, remm, myblkoffst, fstblksiz);
        magma_setdevice( stdev );
        magmablasSetKernelStream( queues[ stdev ][ 0 ] );
        magmablas_ssymmetrize_tiles(  MagmaLower,  fstblksiz,  dA(stdev, offset, myblkoffst*nb+blockoffset),  ldda,  1,  ngpu*nb,  nb  );         
    }

    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_int_t newstdev      = (newoffset/nb)%ngpu;
        magma_int_t nbblk = remm/nb; // number of block of size nb. if m%nb>0 then a last block exist and is of size ib=m%nb
        magma_int_t myblk = (nbblk/ngpu) + (nbblk%ngpu > ((dev-newstdev+ngpu)%ngpu) ?  1:0 );
        magma_int_t devperm   = (dev-newstdev+ngpu)%ngpu;
        magma_int_t nbblkoffst = newoffset/nb;
        magma_int_t myblkoffst = (nbblkoffst/ngpu)+(nbblkoffst%ngpu > dev?1:0);
        //printf("dev %d  devperm %d   newoffset %d  rowoff %d    coloff %d    myblk %d  \n", dev, devperm, newoffset, newoffset+devperm*nb, myblkoffst*nb, myblk);
        magma_setdevice( dev );
        magmablasSetKernelStream( queues[ dev ][ 0 ] );
        magmablas_ssymmetrize_tiles(  MagmaLower,  nb,  dA(dev, newoffset+devperm*nb, myblkoffst*nb),  ldda,  myblk,  ngpu*nb,  nb  );
        if(remm%nb>0){
            magma_int_t nblstblks = (nbblk+1)%ngpu;
            magma_int_t devlstblk = (nblstblks-1+ngpu)%ngpu;
            //printf("==> siz %d devperm %d,    devlstblk %d,    newoffset+nbblk*nb %d,   myblkoffst*nb+ myblk*nb %d\n", remm % nb, devperm, devlstblk, newoffset+nbblk*nb, myblkoffst*nb+ myblk*nb);
            if(devperm==devlstblk)
                magmablas_ssymmetrize(  MagmaLower,  remm % nb,  dA(dev, newoffset+nbblk*nb, myblkoffst*nb+ myblk*nb),  ldda );  // last partial tile
        }
    }


    

/*
    magma_int_t siz = m+offset;
    float *R;
    magma_smalloc_cpu( &R, siz*siz );
    // collecte back A
    magmablas_sgetmatrix_1D_bcyclic( siz, siz, dA, ldda, R, siz, ngpu, nb );
    magma_setdevice( 0 );
    magmablasSetKernelStream( queues[ dev ][ 0 ] );
    //magma_sgetmatrix( siz, siz, dA[0], ldda, R, siz );
    FILE *trace_file;
    trace_file = fopen("AJETE/Aafter", "w");
    for (int j = 0; j < siz ; j++) 
          for (int i = 0; i < siz ; i++) 
                         fprintf(trace_file, "%10d%10d%40.30e\n", i+1, j+1, R[j*siz+i]);
    fclose(trace_file);
return;
*/
    

    // ROW GEMM transpose a row and make a gemm with a block
    // if only 1 GPU used the ROW GEMM is integrated with the 
    // COL GEMM (better accuracy observed) and better perf
    if(ngpu>1){
        for( magma_int_t i = fstblksiz; i < m; i += nb ) {
            magma_int_t ib     = min( nb, m-i );      // block size
            magma_int_t ioff   = i + offset;          // start global index in parent matrix
            //magma_int_t dev    = (ioff / nb) % ngpu;
            magma_int_t nbblkoffst = offset/nb;
            magma_int_t nbblk      = magma_ceildiv(i, nb);
            for( magma_int_t dev = 0; dev < ngpu; ++dev ) {


                magma_int_t myblk = (nbblk/ngpu) + (nbblk%ngpu > ((dev-stdev+ngpu)%ngpu) ?  1:0 );
                magma_int_t myblkoffst = (nbblkoffst/ngpu)+(nbblkoffst%ngpu > dev?1:0);

                magma_int_t myrowsize = myblk * nb;
                magma_int_t coloffset = myblkoffst*nb;
                if(dev==stdev) {
                    myrowsize = myrowsize -blockoffset;
                    coloffset = myblkoffst*nb+blockoffset;
                }
                //printf("ROW GEMM: voici i %d   ib %d    ioff %d   nbblkoffst %d stdev %d  dev %d myblk %d  myblkoffset %d  coloffset %d  rowsize %d\n", i, ib, ioff, nbblkoffst, stdev, dev, myblk, myblkoffst, coloffset, myrowsize);
                if(myrowsize>0){
                    magma_setdevice( dev );
                    magmablasSetKernelStream( queues[ dev ][ 1 ] );    
                    magma_sgemm( MagmaConjTrans, MagmaNoTrans, myrowsize, n, ib,
                                 alpha, dA(dev,ioff,coloffset), ldda,
                                        dB(dev,i,0),    lddb,
                                 c_one, dwork(dev,0,0), lddwork );
                }
            }
        }
        for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
            magma_setdevice( dev );
            magma_event_record(redevents[dev][1], queues[dev][1]);
        }
    }
    

    // COL GEMM
    // blockoffset is offset within first block; for subsequent blocks it is 0
    if(blockoffset>0){
        magma_int_t ib     = min( nb-blockoffset, m );  // block size
        magma_int_t iblock = (offset / nb) / ngpu;          // local block id
        magma_int_t di     = iblock*nb+blockoffset;       // local index in parent matrix
        magma_setdevice( stdev );
        magmablasSetKernelStream( queues[ stdev ][ 0 ] );        
        //printf("DEV %d COL GEMM first   ioff %d  di %d   m %d   n %d   ib %d \n", stdev, offset, di, m, n, ib);
        magma_sgemm( MagmaNoTrans, MagmaNoTrans, m, n, ib,
                        alpha, dA(stdev,offset,di), ldda,
                               dB(stdev,0,0),     lddb,
                        beta,  dC(stdev,0,0),     lddc );
    }
   


    // COL GEMM
    for( magma_int_t i = fstblksiz; i < m; i += nb ) {
        magma_int_t ib     = min( nb, m-i );      // block size
        magma_int_t ioff   = i + offset;          // start global index in parent matrix
        magma_int_t iblock = (ioff / nb) / ngpu;  // local block id
        magma_int_t dev    = (ioff / nb) % ngpu;
        magma_int_t di     = iblock*nb;           // local index in parent matrix
        
        //printf("DEV %d COL GEMM i %d      ioff %d  di %d m-i %d    n %d   ib %d \n", dev, i, ioff, di, m-i, n, ib);
        
        magma_setdevice( dev );
        magmablasSetKernelStream( queues[ dev ][ 0 ] );
        if(i==0){
           magma_sgemm( MagmaNoTrans, MagmaNoTrans, m-i, n, ib,
                        alpha, dA(dev,ioff,di), ldda,
                               dB(dev,i,0),     lddb,
                        beta,  dC(dev,i,0),     lddc );
        }else{
           magma_sgemm( MagmaNoTrans, MagmaNoTrans, m-i, n, ib,
                        alpha, dA(dev,ioff,di), ldda,
                               dB(dev,i,0),        lddb,
                        c_one, dC(dev,i,0),     lddc );
        }
        magma_event_record(redevents[dev][0], queues[dev][0]);
        // if only 1 GPU is used, do the ROW GEMM
        if(ngpu==1){
            // NOTE THAT because the COL gemm write dC below the diagonal (i) 
            // and the ROW GEMM write dC from 0 to diag-1, so they could 
            // run in parallel on different queues. 
            // 
            // NO NO NO because
            // it might happen that col finished i and strated i+1 while row still at i    
            // magmablasSetKernelStream( queues[ dev ][ 0 ] );
            magma_sgemm( MagmaConjTrans, MagmaNoTrans, i, n, ib,
                         alpha, dA(dev,ioff,offset), ldda,
                                dB(dev,i,0),    lddb,
                         c_one, dC(dev,0,0),    lddc );
        }
    }


    
    if(ngpu>1){
        for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
            magma_int_t nbblk    = magma_ceildiv((m+blockoffset), nb);
            magma_int_t nbblkrow = nbblk-1; 
            magma_int_t devperm  = (dev-stdev+ngpu)%ngpu;
            magma_int_t myblk = (nbblkrow/ngpu) + (nbblkrow%ngpu > devperm ?  1:0 );
            magma_int_t myrowsize = myblk * nb;
             if(dev==stdev) {
                myrowsize = myrowsize - blockoffset;
            }
      
            //printf("blockoffset %d nbblkrow %d devperm %d  DEV %d RECEIVING myblk %d  myrowsize %d\n", blockoffset, nbblkrow, devperm, dev, myblk, myrowsize);
            if(myrowsize>0){
                magma_setdevice( dev );
                magmablasSetKernelStream( queues[ dev ][ 0 ] );
                magma_queue_wait_event(queues[ dev ][ 0 ], redevents[dev][1]);
                //magma_queue_sync( queues[ dev ][ 1 ] );
                // for each dev add the computed ROW block each on its placment with dC
                for( magma_int_t blki = 0; blki < myblk; ++blki){
                    magma_int_t gbblki = (blki*ngpu + devperm)*nb - blockoffset;
                    magma_int_t lcblki = blki*nb;
                    magma_int_t ib     = nb;// min(nb, m-gbblki);
                    if(dev==stdev){
                        lcblki = blki*nb-blockoffset;
                        if(blki==0){
                            gbblki = 0;
                            lcblki = 0;
                            ib     = nb-blockoffset;
                        }
                    }
                    magmablas_sgeadd(ib, n, c_one, 
                                    &dwork[dev][lcblki], lddwork, 
                                    &dC[dev][gbblki]   , lddc   );
                }
                magma_event_record(redevents[dev][0], queues[dev][0]);                
            }
        }
    }




    // ===========================================================
    //             COMMUNICATION ALL_REDUCE_SUM 
    // ===========================================================
    if(ngpu==1){
        return;
    }
    // INITIALIZE COMM
    for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
        masterdev     = -1;
        gnode[cmplxid][MagmaMaxGPUs+1] = -1;
        myngpu = gnode[cmplxid][MagmaMaxGPUs];
        for( magma_int_t idev = 0; idev < myngpu; ++idev ) {
            dev         = gnode[cmplxid][idev];
            devperm     = (dev-stdev+ngpu)%ngpu;
            myblk       = (nbblk/ngpu) + (nbblk%ngpu > devperm ?  1:0 );
            mycolsize   = myblk*nb;
            myblkoffst  = nb*((nbblkoffst/ngpu)+(nbblkoffst%ngpu > dev?1:0));            
            if(dev==stdev){
                mycolsize  -=  blockoffset;
                myblkoffst +=  blockoffset;     // local index in parent matrix
            }
            if((devperm==devlstblk)&&(lstblksiz>0)){
                mycolsize -=  (nb-(remm%nb));
            }
            mycolsize = min(mycolsize, m);
            if(mycolsize>0){
                gpuisactive[dev] = mycolsize;
                if(masterdev==-1) {
                    masterdev     = dev;
                    nbcmplxactive = nbcmplxactive +1;
                    cmplxisactive[cmplxid] = 1;
                    gnode[cmplxid][MagmaMaxGPUs+1] = masterdev;
                }
            }
        }
    }
/*
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_device_sync();
    }
*/
    //*******************************
    //  each GPU send its result
    //  to its master. The master make
    //  the addition and then send to 
    //  to the masters of other real
    //  and receive from the masters of 
    //  other real make the addition 
    //  and broadcast locally the final 
    //  result.
    //*******************************
    //printf("=======================================================================\n");
    //printf("                     sending to my master                             \n");
    //printf("=======================================================================\n");
    for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        //check if real is active
        if(masterdev!=-1){ 
            for( magma_int_t idev = 0; idev < myngpu; ++idev ) {
                dev         = gnode[cmplxid][idev];
                mycolsize   = gpuisactive[dev];
                if(mycolsize>0){
                    // I am an active GPU. if I am not the master, then send my result to my master.
                    // store result on dwork[masterdev][dev*maxgsize]
                    if(dev!=masterdev){
                        magma_setdevice( dev );        
                        //printf("             GPU %d sending to my master %d\n", dev, masterdev);
                        // wait the geadd of my ROW and COL GEMM is done
                        magma_queue_wait_event(queues[ dev ][ 0 ], redevents[dev][0]);
                        // sending to the master of my real
                        magma_scopymatrix_async(
                            m, n,
                            &dC[dev][0], lddc,
                            &dwork2[masterdev][maxgsize*dev], m, queues[dev][0] );
                        magma_event_record(redevents[dev][masterdev], queues[dev][0]);
                    } // end I am not the masterdev
                }// end if mycolsize>0
            }// for idev
        }// end of if masterdev!=-1 maening real is active
    }// for cmplxid
/*
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_device_sync();
    }
*/

    //printf("=======================================================================\n");
    //printf(" each master do addition of local result and broadcast to other masters \n");
    //printf("=======================================================================\n");
    for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        //check if real is active
        if(masterdev!=-1){ 
            magma_setdevice( masterdev ); 
            // addition is done on stream 0 sequentially
            magmablasSetKernelStream( queues[ masterdev ][ 0 ] );
            // wait the geadd of my ROW and COL GEMM is done
            magma_queue_wait_event(queues[ masterdev ][ 0 ], redevents[masterdev][0]);
            // ========================================
            //     local addition
            // ========================================
            for( magma_int_t l = 0; l < myngpu; ++l ) {
                lcdev         = gnode[cmplxid][l];
                lccolsize     = gpuisactive[lcdev];
                if((lcdev!=masterdev)&&(lccolsize>0)){
                    //printf("             master %d receiving from %d and adding \n", masterdev, lcdev);
                    // this is an active GPU of my real. 
                    // wait I received what he send it to me and then do addition.
                    magma_queue_wait_event(queues[ masterdev ][ 0 ], redevents[lcdev][masterdev]);
                    magmablas_sgeadd(m, n, c_one, 
                                    &dwork2[masterdev][maxgsize*lcdev], m, 
                                    &dC[masterdev][0]   , lddc   );
                }
            }// for l=1:myngpu
            // because addition is done sequentially on stream 0, 
            // I have to record this to be able to synch using it 
            magma_event_record(redevents[masterdev][masterdev], queues[masterdev][0]);
            // ========================================
            //
            // ========================================
            //      send to other masters
            // ========================================
            for( magma_int_t k = 0; k < nbcmplx; ++k ) {
                if(k!=cmplxid){
                    gmaster = gnode[k][MagmaMaxGPUs+1];
                    if(gmaster!=-1){ //real is active
                         //Master has to  wait until finish the local addition then send using gmaster stream.
                         //use stream 0 to make it sequential or stream gmaster to make it parallel.
                         //Now both re the same.
                        //printf("             master %d from cmplx %d sending to other master %d on cmplx %d \n", masterdev, cmplxid, gmaster, k);
                        magma_queue_wait_event(queues[ masterdev ][ gmaster ], redevents[masterdev][masterdev]);
                        magma_scopymatrix_async(
                            m, n,
                            &dC[masterdev][0], lddc,
                            &dwork2[gmaster][maxgsize*masterdev], m, queues[masterdev][gmaster] );
                        magma_event_record(redevents[masterdev][gmaster], queues[masterdev][gmaster]);
                        magma_event_record(redevents[masterdev][masterdev], queues[masterdev][gmaster]);
                      } // end of gmaster!=-1
                } // end of k!=cmplxid
            }// for k = 0: nbcmplx
            // ========================================
        }// end of if masterdev!=-1 maening real is active
    }// for cmplxid
/*
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_device_sync();
    }
*/
    //printf("=======================================================================\n");
    //printf(" each master wait receiving other masters results, do the addition and broadcast locally \n");
    //printf("=======================================================================\n");
    for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        //check if real is active
        if(masterdev!=-1){ 
            magma_setdevice( masterdev ); 
            // addition is done on stream 0 sequentially
            magmablasSetKernelStream( queues[ masterdev ][ 0 ] );
            // master has to wait until finishing all the send to other masters.
            magma_queue_wait_event(queues[ masterdev ][ 0 ], redevents[masterdev][masterdev]);
            // ========================================
            //  addition of results from other masters
            // ========================================
            for( magma_int_t k = 0; k < nbcmplx; ++k ) {
                if(k!=cmplxid){
                    gmaster = gnode[k][MagmaMaxGPUs+1];
                    if(gmaster!=-1){ //real is active
                        //Master has to  wait until receiving from gmaster, then do addition using stream 0
                        //printf("             master %d from cmplx %d receiving from other master %d on cmplx %d and adding \n", masterdev, cmplxid, gmaster, k);
                        magma_queue_wait_event(queues[ masterdev ][ 0 ], redevents[gmaster][masterdev]);
                        magmablas_sgeadd(m, n, c_one, 
                                        &dwork2[masterdev][maxgsize*gmaster], m, 
                                        &dC[masterdev][0]   , lddc   );
                    } // end of gmaster!=-1
                } // end of k!=cmplxid
            }// for k = 0: nbcmplx
            // because addition is done sequentially on stream 0, 
            // I have to record this to be able to synch using it 
            magma_event_record(redevents[masterdev][masterdev], queues[masterdev][0]);
            // ========================================
            // ========================================
            //     local broadcast of final results
            // ========================================
            for( magma_int_t l = 0; l < myngpu; ++l ) {
                lcdev         = gnode[cmplxid][l];
                lccolsize     = gpuisactive[lcdev];
                if((lcdev!=masterdev)&&(lccolsize>0)){
                    // this is an active GPU of my real. 
                    // wait the previous addition is done maening stream 0 is finished and broadcast sequentially for now.
                    // to make it parallel put stream lcdev instead of stream 0
                    //printf("             master %d broadcasting local to %d  \n", masterdev, lcdev);
                    magma_queue_wait_event(queues[ masterdev ][ 0 ], redevents[masterdev][masterdev]);
                    magma_scopymatrix_async(
                        m, n,
                        &dC[masterdev][0], lddc,
                        &dC[lcdev][0],     lddc, queues[masterdev][0] );
                    magma_event_record(redevents[masterdev][lcdev], queues[masterdev][0]);
                }
            }// for l=1:myngpu
            // ========================================
        }// end of if masterdev!=-1 maening real is active
    }// for cmplxid
/*
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_device_sync();
    }
*/


    for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        //check if real is active
        if(masterdev!=-1){ 
            for( magma_int_t l = 0; l < myngpu; ++l ) {
                lcdev         = gnode[cmplxid][l];
                lccolsize     = gpuisactive[lcdev];
                if(lccolsize>0){
                    magma_setdevice( lcdev );
                    magma_queue_wait_event(queues[ lcdev ][ 0 ], redevents[lcdev][0]);
                    magma_queue_wait_event(queues[ lcdev ][ 0 ], redevents[masterdev][lcdev]);
                }
            }// for l=1:myngpu
        }// end of if masterdev!=-1 maening real is active
    }// for cmplxid


 
   //printf("****************************************************\n");
   //printf("                      finish ssymm                   \n");
   //printf("****************************************************\n");

    magma_setdevice( cdev );
    magmablasSetKernelStream( cstream );

}
