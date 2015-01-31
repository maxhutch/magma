/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zhemm_mgpu_spec.cpp normal z -> s, Fri Jan 30 19:00:10 2015
       @author Azzam Haidar
*/
#include "common_magma.h"
#include "magma_bulge.h"
//#include "trace.h"

extern "C"
void magmablas_ssymm_mgpu_spec(
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
    
    if ( side != MagmaLeft || uplo != MagmaLower ) {
        fprintf( stderr, "%s: only Left Lower implemented\n", __func__ );
    }
    
    assert( ldda >= m );
    assert( lddb >= m );
    assert( lddc >= m );
    assert( nqueue >= ngpu );
    assert( nbevents >= ngpu*ngpu );
    
    magma_int_t lddwork = lddc;
    assert( dworksiz >= (2*n*lddwork) );




        
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_t cstream;
    magmablasGetKernelStream(&cstream);


    magma_int_t dev, devperm, myblk, mycolsize, myblkoffst;
    magma_int_t gdev, gcolsize, gmaster, gngpu;
    magma_int_t masterdev, lcdev, lccolsize, myngpu;

    magma_int_t stdev       = (offset/nb)%ngpu;  
    magma_int_t blockoffset = offset % nb;  
    magma_int_t fstblksiz   = 0;
    if(blockoffset>0){
        fstblksiz   = min(m, (nb - blockoffset));
    }
    //magma_int_t nbblk       = magma_ceildiv(m, nb);
    magma_int_t nbblk       = magma_ceildiv((m+blockoffset), nb);
    magma_int_t maxgsize    = n*nb*magma_ceildiv(nbblk, ngpu);
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


    //*******************************
    //  each GPU make a GEMM with the
    //  transpose of its blocks to compute
    //  a final portion of X=A*VT
    //*******************************
    /* dB = V*T already ==> dB**H = T**H * V**H
     * compute T**H * V**H * X is equal to compute locally (VT)**H_i*X_i 
     * then  each GPU broadcast its X_i to assemble the full X which is used
     * to compute W  =  X  - 0.5 * V * T**H * V**H * X  = X - 0.5 * V *dwork3
     */
    if(ngpu ==1){
        magma_setdevice( 0 );
        magmablasSetKernelStream( queues[ 0 ][ 0 ] );
        // compute X[me] = A*VT = A[me]^tr *VT;
        magma_sgemm( MagmaConjTrans, MagmaNoTrans, m, n, m,
                     alpha, dA(0, offset, offset), ldda,
                            dB[0],         lddb,
                     beta,  dC[0], lddc );
        return;
    }
    //ngpu>1
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
                if(masterdev==-1) masterdev     = dev;
                //printf("dev %d devperm %d on cmplx %d  master %d nbblk %d myblk %d m %d n %d mycolsize %d stdev %d fstblksize %d lastdev %d lastsize %d dA(%d, %d, %d) ==> dwork(%d, %d)\n", dev, devperm, cmplxid, masterdev, nbblk, myblk, m, n, mycolsize, stdev, fstblksiz, devlstblk, remm%nb, dev, offset, myblkoffst, dev, maxgsize*dev);
                gpuisactive[dev] = mycolsize;
                magma_setdevice( dev );
                magmablasSetKernelStream( queues[ dev ][ dev ] );    

                magma_sgemm( MagmaConjTrans, MagmaNoTrans, mycolsize, n, m,
                             alpha, dA(dev, offset, myblkoffst), ldda,
                                    dB(dev, 0, 0),    lddb,
                             beta,  &dwork[dev][maxgsize*dev], mycolsize );
                magma_event_record(redevents[dev][dev*ngpu+dev], queues[dev][dev]);
            }
            if(dev == masterdev){
                nbcmplxactive = nbcmplxactive +1;
                cmplxisactive[cmplxid] = 1;
                gnode[cmplxid][MagmaMaxGPUs+1] = masterdev;
            }
        }
    }



/*
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[ dev ][ dev ] );
    }
*/


    //*******************************
    //  each Master GPU has the final
    //  result either by receiving 
    //  from CPU of by making the add
    //  by himself, so now it is time 
    //  to broadcast over the GPUs of 
    //  its board.
    //*******************************
    //printf("=======================================================================\n");
    //printf("                           sending                                     \n");
    //printf("=======================================================================\n");
    
    for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        for( magma_int_t idev = 0; idev < myngpu; ++idev ) {
            dev         = gnode[cmplxid][idev];
            mycolsize   = gpuisactive[dev];
            if(mycolsize>0){
                // I am an active GPU send my portion local 
                // to all active gpu of my cmplex and global to the 
                // active master of the other real and they should 
                // send it out to their actives slaves.
                magma_setdevice( dev );        
                //==============================================
                // sending to the master of the active real
                //==============================================
                //printf     ("\n\n**************GPU %d\n ", dev);
                //printf     ("             GPU %d sending to cmplx masters\n", dev);
                for( magma_int_t k = 0; k < nbcmplx; ++k ) {
                    if(k!=cmplxid){
                        gmaster = gnode[k][MagmaMaxGPUs+1];
                        if(gmaster!=-1){ //real is active
                            //printf     ("                    device %d from cmplx %d is sending to master %d on cmplx %d block of size %d event %d\n", dev, cmplxid, gmaster, k, mycolsize, redevents[dev][gmaster*ngpu+dev]);
                            magma_queue_wait_event(queues[ dev ][ gmaster ], redevents[dev][dev*ngpu+dev]);
                            magma_scopymatrix_async(
                                mycolsize, n,
                                &dwork[dev    ][maxgsize*dev], mycolsize,
                                &dwork[gmaster][maxgsize*dev], mycolsize, queues[dev][gmaster] );
                            magma_event_record(redevents[dev][gmaster*ngpu+dev], queues[dev][gmaster]);
                        }
                    }
                }
                //==============================================
                //
                //==============================================
                // sending to the active GPUs of my real
                //==============================================
                //printf     ("              GPU %d sending internal\n", dev);                
                for( magma_int_t l = 0; l < myngpu; ++l ) {
                    lcdev         = gnode[cmplxid][l];
                    lccolsize     = gpuisactive[lcdev];
                    if((lcdev!=dev)&&(lccolsize>0)){
                        //printf     ("                    device %d from cmplx %d is sending internal to dev %d block of size %d event %d\n", dev, cmplxid, lcdev, mycolsize, redevents[dev][lcdev*ngpu+dev]);
                        magma_queue_wait_event(queues[ dev ][ lcdev ], redevents[dev][dev*ngpu+dev]);
                        magma_scopymatrix_async(
                            mycolsize, n,
                            &dwork[dev  ][maxgsize*dev], mycolsize,
                            &dwork[lcdev][maxgsize*dev], mycolsize, queues[dev][lcdev] );
                        magma_event_record(redevents[dev][lcdev*ngpu+dev], queues[dev][lcdev]);
                    }
                }
                //==============================================
            }// end if mycolsize>0
        }// for idev
    }// for cmplxid


    //printf("=======================================================================\n");
    //printf("                master wait and resend internally                      \n");
    //printf("=======================================================================\n");
    
    for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        //==============================================
        // if I am active master so wait receiving contribution
        // of the GPUs of other real and send it locally
        //==============================================
        if(masterdev != -1){
            mycolsize   = gpuisactive[masterdev];
            magma_setdevice( masterdev );
            //printf("              GPU %d distributing internal\n", masterdev);
            for( magma_int_t k = 0; k < nbcmplx; ++k ) {
                if(k!=cmplxid){
                    gngpu   = gnode[k][MagmaMaxGPUs];
                    for( magma_int_t g = 0; g < gngpu; ++g ) {
                        gdev         = gnode[k][g];
                        gcolsize     = gpuisactive[gdev];
                        // check if I received from this GPU,
                        // if yes send it to my group
                        if(gcolsize>0){
                           magma_queue_wait_event(queues[ masterdev ][ gdev ], redevents[gdev][masterdev*ngpu+gdev]);
                           for( magma_int_t l = 0; l < myngpu; ++l ) {
                                lcdev         = gnode[cmplxid][l];
                                lccolsize     = gpuisactive[lcdev];
                                if((lcdev!=masterdev)&&(lccolsize>0)){
                                    //printf("                    Master %d on cmplx %d waiting on event %d is distributing internal results of %d to lcdev %d block of size %d event %d\n", masterdev, cmplxid, redevents[gdev][masterdev*ngpu+gdev], gdev, lcdev, gcolsize, redevents[masterdev][lcdev*ngpu+gdev]);
                                    magma_scopymatrix_async(
                                        gcolsize, n,
                                        &dwork[masterdev][maxgsize*gdev], gcolsize,
                                        &dwork[lcdev    ][maxgsize*gdev], gcolsize, queues[masterdev][gdev] );
                                    magma_event_record(redevents[masterdev][lcdev*ngpu+gdev], queues[masterdev][gdev]);
                                }
                            }
                        }
                    }
                }
            }
        }// if active master 
        //==============================================
    }// for cmplxid





/*

    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
                magma_queue_sync( queues[ dev ][ 0 ] );
        for( magma_int_t s = 0; s < ngpu; ++s ) {
                magma_queue_sync( queues[ dev ][ s ] );
        }
    }
*/
    //printf("=======================================================================\n");
    //printf("                           distributing                                \n");
    //printf("=======================================================================\n");

    magma_int_t lcblki, gbblki, gblk, ib;
    
    for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        for( magma_int_t idev = 0; idev < myngpu; ++idev ) {
            dev         = gnode[cmplxid][idev];
            mycolsize   = gpuisactive[dev];
            if(mycolsize>0){ // I am an active GPU
                //printf("\n\n==============GPU %d collecting\n", dev);
                magma_setdevice( dev );        
                // collect my results first as tyhere is no need to wait to   
                // receive nothing, just wait that my gemm are done.
                // in theory this should be inside the loop but cuda was not 
                // able to run it first for all gpu and on gpu>0 it was waiting
                // however it was on different stream so it should run. but maybe
                // this is because there are too many function call and this make 
                // cuda not handleit so nice. anyway it coul dbe removed when cuda
                // is able to lunch it first without wait.
                gdev = dev;
                gcolsize     = gpuisactive[gdev];
                if(gcolsize>0){
                    devperm     = (gdev-stdev+ngpu)%ngpu;
                    gblk        = (nbblk/ngpu) + (nbblk%ngpu > devperm ?  1:0 );
                    magmablasSetKernelStream( queues[ dev ][ gdev ] );
                    magma_queue_wait_event(queues[ dev ][ gdev ], redevents[gdev][dev*ngpu+gdev]);
                    //printf     ("              GPU %d stream %d doing slacpy\n", dev, queues[ dev ][ gdev ]);
                    for( magma_int_t blki = 0; blki < gblk; ++blki){
                        gbblki = (blki*ngpu + devperm)*nb - blockoffset;
                        lcblki = blki*nb;
                        ib     = nb;//min(nb, m-gbblki);
                        if(gdev==stdev){
                            lcblki = blki*nb-blockoffset;
                            if(blki==0){
                                gbblki = 0;
                                lcblki = 0;
                                ib     = nb-blockoffset;
                            }
                        }
                        ib     = min(ib, m-gbblki);
                        //printf("                    blockoffset %d nbblk %d stdev %d  receiving from gdev %d gblk %d  gcolsize %d copying blki %d of size ibxn %dx%d from work[%d] to C[%d]\n", blockoffset, nbblk, stdev, gdev, gblk, gcolsize, blki, ib, n, lcblki, gbblki);
                        magmablas_slacpy( MagmaFull, ib, n, &dwork[dev][maxgsize*gdev+lcblki], gcolsize, &dC[dev][gbblki], lddc);
                    }// end blki
                }


                
                for( magma_int_t k = 0; k < nbcmplx; ++k ) {
                    gngpu   = gnode[k][MagmaMaxGPUs];
                    for( magma_int_t g = 0; g < gngpu; ++g ) {
                        gdev         = gnode[k][g];
                        gcolsize     = gpuisactive[gdev];
                        // if gcolsize>0, ==> gpu gdev was active and so 
                        // I received from him/computed a portion of dwork, 
                        // so go over its gblk and distribute it on dC.
                        if(gdev!=dev){
                            if(gcolsize>0){
                                devperm     = (gdev-stdev+ngpu)%ngpu;
                                gblk        = (nbblk/ngpu) + (nbblk%ngpu > devperm ?  1:0 );
                                magmablasSetKernelStream( queues[ dev ][ gdev ] );
                                if(k==cmplxid){
                                    //we are on the same group so wait on event issued by gdev for me citing his id
                                    magma_queue_wait_event(queues[ dev ][ gdev ], redevents[gdev][dev*ngpu+gdev]);
                                    //printf     ("              GPU %d queue %d waiting on event %d to collecte from %d the size of gcolsize %d\n", dev, queues[ dev ][ gdev ], redevents[gdev][dev*ngpu+gdev], gdev, gcolsize);
                                }else{
                                    //we are on different group so:
                                    //if I am the master wait on the event issued by gdev for me citing his id
                                    //else  wait event issued by my master for me on the behalf of gdev
                                    //printf     ("              GPU %d queue %d waiting on event %d to collecte from %d the size of gcolsize %d\n", dev, queues[ dev ][ gdev ], redevents[masterdev][dev*ngpu+gdev], gdev, gcolsize);
                                    if(dev==masterdev)
                                        magma_queue_wait_event(queues[ dev ][ gdev ], redevents[gdev][dev*ngpu+gdev]);
                                    else
                                        magma_queue_wait_event(queues[ dev ][ gdev ], redevents[masterdev][dev*ngpu+gdev]);
                                }
                                //printf     ("              GPU %d stream %d doing slacpy\n", dev, queues[ dev ][ gdev ]);
                                for( magma_int_t blki = 0; blki < gblk; ++blki){
                                    gbblki = (blki*ngpu + devperm)*nb - blockoffset;
                                    lcblki = blki*nb;
                                    ib     = nb;//min(nb, m-gbblki);
                                    if(gdev==stdev){
                                        lcblki = blki*nb-blockoffset;
                                        if(blki==0){
                                            gbblki = 0;
                                            lcblki = 0;
                                            ib     = nb-blockoffset;
                                        }
                                    }
                                    ib     = min(ib, m-gbblki);
                                    //printf("                    blockoffset %d nbblk %d stdev %d  receiving from gdev %d gblk %d  gcolsize %d copying blki %d of size ibxn %dx%d from work[%d] to C[%d]\n", blockoffset, nbblk, stdev, gdev, gblk, gcolsize, blki, ib, n, lcblki, gbblki);
                                    magmablas_slacpy( MagmaFull, ib, n, &dwork[dev][maxgsize*gdev+lcblki], gcolsize, &dC[dev][gbblki], lddc);
                                }// end blki
                            }// en gcolsize>0 meaning gdev is active
                        } // end if gdev != dev
                    }// end loop over the g gpus of the cmplx k
                }//end loop over the real k
            }// end mycolsize>0 meaning that I am active
        }// end loop over idev of cmplxid
    }// end loop of the cmplx







    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_device_sync();
    }

    // put back the input gpu and its input stream 
    magma_setdevice( cdev );
    magmablasSetKernelStream( cstream );

}
