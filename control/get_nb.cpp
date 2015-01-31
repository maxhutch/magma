/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
       
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
*/

#include "magma.h"
#include "common_magma.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==== Definition of blocking sizes for Nvidia cards
#ifdef HAVE_CUBLAS

// Optimal block sizes vary with GPU and, to lesser extent, CPU.
// Kepler tuning was on K20c   705 MHz with SandyBridge 2.6 GHz host (bunsen).
// Fermi  tuning was on S2050 1147 MHz with AMD Opteron 2.4 GHz host (romulus).

// Unused arguments are commented out, like /*m*/, to silence compiler warnings,
// especially on Windows.
// Unreachable code should also be removed or commented out.

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for potrf based on m
*/
magma_int_t magma_get_spotrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  1500) return 256;
        else                return 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m <  2048) return 256;
        else                return 512;
    }
    else {                     // 1.x
        if      (m <  3328) return 128;
        else if (m <  4256) return 224;
        else                return 288;
    }
}

magma_int_t magma_get_dpotrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  3072) return 256;
        else                return 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 256;
    }
    else {                     // 1.x
        if      (m <  3328) return 128;
        else if (m <  4256) return 128;
        else                return 256;
    }
}

magma_int_t magma_get_cpotrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        return 256;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m <  1500) return 192;
        else                return 256;
    }
    else {                     // 1.x
        return 64;
    }
}

magma_int_t magma_get_zpotrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        return 256;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m <  1500) return 192;
        else                return 256;
    }
    else {                     // 1.x
        return 64;
    }
}

/* ///////////////////////////////////////////////////////////////////////// */

magma_int_t magma_get_zpotrf_right_nb( magma_int_t m )
{
    return 128;
}

magma_int_t magma_get_cpotrf_right_nb( magma_int_t m )
{
    return 128;
}

magma_int_t magma_get_dpotrf_right_nb( magma_int_t m )
{
    return 320;
}

magma_int_t magma_get_spotrf_right_nb( magma_int_t m )
{
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for geqp3 based on m
*/
magma_int_t magma_get_sgeqp3_nb( magma_int_t /*m*/ )
{
    return 32;
}

magma_int_t magma_get_dgeqp3_nb( magma_int_t /*m*/ )
{
    return 32;
}

magma_int_t magma_get_cgeqp3_nb( magma_int_t /*m*/ )
{
    return 32;
}

magma_int_t magma_get_zgeqp3_nb( magma_int_t /*m*/ )
{
    return 32;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for geqrf based on m
*/
magma_int_t magma_get_sgeqrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  4096) return 96;
        else if (m <  7168) return 128;
        else if (m < 18432) return 256;
        else                return 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m <  3072) return 64;
        else if (m <  8192) return 128;
        else                return 256;
    }
    else {                     // 1.x
        if      (m <  2048) return 32;
        else if (m <  4096) return 64;
        else                return 128;
    }
}

magma_int_t magma_get_dgeqrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  3072) return 64;
        else if (m < 10240) return 128;
        else                return 256;
    }
    else {                     // 1.x and 2.x Fermi
        if      (m <  4096) return 64;
        else                return 128;
    }
}

magma_int_t magma_get_cgeqrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  4096) return 64;
        else                return 128;
    }
    else {                     // 1.x and 2.x Fermi
        if      (m <  2048) return 32;
        else if (m <  4096) return 64;
        else                return 128;
    }
}

magma_int_t magma_get_zgeqrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  4096) return 64;
        else                return 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m <  2048) return 32;
        else if (m <  4096) return 64;
        else                return 128;
    }
    else {                     // 1.x
        if      (m <  1024) return 64;
        else                return 128;
    }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for geqlf based on m
*/
magma_int_t magma_get_sgeqlf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        return magma_get_sgeqrf_nb( m );
    }
    else {                     // 1.x
        if      (m <  1024) return 32;
        else if (m <  4032) return 64;
        else                return 128;
    }
}

magma_int_t magma_get_dgeqlf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        return magma_get_dgeqrf_nb( m );
    }
    else {                     // 1.x
        if      (m <  1024) return 32;
        else if (m <  4032) return 64;
        else                return 128;
    }
}

magma_int_t magma_get_cgeqlf_nb( magma_int_t m )
{
    if      (m <  2048) return 32;
    else if (m <  4032) return 64;
    else                return 128;
}

magma_int_t magma_get_zgeqlf_nb( magma_int_t m )
{
    if      (m <  1024) return 64;
    else                return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gelqf based on m
*/
magma_int_t magma_get_sgelqf_nb( magma_int_t m )
{
    return magma_get_sgeqrf_nb( m );
}

magma_int_t magma_get_dgelqf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        return magma_get_dgeqrf_nb( m );
    }
    else {                     // 1.x
        if      (m <  2048) return 32;
        else if (m <  4032) return 64;
        else                return 128;
    }
}

magma_int_t magma_get_cgelqf_nb( magma_int_t m )
{
    if      (m <  2048) return 32;
    else if (m <  4032) return 64;
    else                return 128;
}

magma_int_t magma_get_zgelqf_nb( magma_int_t m )
{
    if      (m <  1024) return 64;
    else                return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for getrf based on m
*/
magma_int_t magma_get_sgetrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  4096) return 256;
        else if (m < 18432) return 512;
        else                return 1024;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m <  3072) return 128;
        else if (m < 10240) return 256;
        else                return 512;
    }
    else {                     // 1.x
        if      (m <  2048) return 64;
        else                return 128;
    }
}

magma_int_t magma_get_dgetrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  3072) return 128;
        else if (m <  8192) return 256;
        else                return 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m <  3072) return 128;
        else if (m < 10240) return 256;
        else                return 512;
    }
    else {                     // 1.x
        if      (m <  2048) return 64;
        else                return 128;
    }
}

magma_int_t magma_get_cgetrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m < 4096) return 64;
        else if (m < 8192) return 256;
        else               return 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m <  2048) return 64;
        else                return 128;
    }
    else {                     // 1.x
        if      (m <  2048) return 64;
        else                return 128;
    }
}

magma_int_t magma_get_zgetrf_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m < 4096) return 64;
        else if (m < 8192) return 256;
        else               return 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m < 4096) return 64;
        else               return 128;
    }
    else {                     // 1.x
        return 128;
    }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gehrd based on m
*/
magma_int_t magma_get_sgehrd_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        if      (m <  1024) return 32;
        else                return 96;
    }
    else {                     // 1.x
        if      (m <  1024) return 32;
        else                return 64;
    }
}

magma_int_t magma_get_dgehrd_nb( magma_int_t m )
{
    if      (m <  2048) return 32;
    else                return 64;
}

magma_int_t magma_get_cgehrd_nb( magma_int_t m )
{
    if      (m <  1024) return 32;
    else                return 64;
}

magma_int_t magma_get_zgehrd_nb( magma_int_t m )
{
    if      (m <  2048) return 32;
    else                return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sytrd based on m
      Must be 64 due to zhemv_mgpu restrictions.
*/
magma_int_t magma_get_ssytrd_nb( magma_int_t /*m*/ )
{
    return 64;
}

magma_int_t magma_get_dsytrd_nb( magma_int_t /*m*/ )
{
    return 64;
}

magma_int_t magma_get_chetrd_nb( magma_int_t /*m*/ )
{
    return 64;
}

magma_int_t magma_get_zhetrd_nb( magma_int_t /*m*/ )
{
    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sytrf based on m
*/
magma_int_t magma_get_zhetrf_nb( magma_int_t m ) 
{
    return 256;
}

magma_int_t magma_get_chetrf_nb( magma_int_t m ) 
{
    return 256;
}

magma_int_t magma_get_dsytrf_nb( magma_int_t m ) 
{
    return 96; 
}

magma_int_t magma_get_ssytrf_nb( magma_int_t m ) 
{
    return 256;
}

/* //////////////////////////////////////////////////////////////////////// */
magma_int_t magma_get_zhetrf_nopiv_nb( magma_int_t m ) 
{
    return 320; 
}

magma_int_t magma_get_chetrf_nopiv_nb( magma_int_t m ) 
{
    return 96; 
}

magma_int_t magma_get_dsytrf_nopiv_nb( magma_int_t m ) 
{
    return 320; 
}

magma_int_t magma_get_ssytrf_nopiv_nb( magma_int_t m ) 
{
    return 96;  
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gebrd based on m
*/
magma_int_t magma_get_sgebrd_nb( magma_int_t /*m*/ )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        return 32;
    }
    else {                     // 1.x
        return 32;
    }
}

magma_int_t magma_get_dgebrd_nb( magma_int_t /*m*/ )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        return 32;
    }
    else {                     // 1.x
        return 32;
    }
}

magma_int_t magma_get_cgebrd_nb( magma_int_t /*m*/ )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        return 32;
    }
    else {                     // 1.x
        return 32;
    }
}

magma_int_t magma_get_zgebrd_nb( magma_int_t /*m*/ )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        return 32;
    }
    else {                     // 1.x
        return 32;
    }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sygst based on m
*/
magma_int_t magma_get_ssygst_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  4096) return 768;
        else                return 1536;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m <  2048) return 512;
        else                return 1024;
    }
    else {                     // 1.x
        return 64;
    }
}

magma_int_t magma_get_dsygst_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  2048) return 384;
        else                return 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 512;
    }
    else {                     // 1.x
        return 64;
    }
}

magma_int_t magma_get_chegst_nb( magma_int_t m )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  2048) return 384;
        else                return 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 512;
    }
    else {                     // 1.x
        return 64;
    }
}

magma_int_t magma_get_zhegst_nb( magma_int_t /*m*/ )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        return 384;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 256;
    }
    else {                     // 1.x
        return 64;
    }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for getri based on m
*/
magma_int_t magma_get_sgetri_nb( magma_int_t /*m*/ )
{
    return 64;
}

magma_int_t magma_get_dgetri_nb( magma_int_t /*m*/ )
{
    return 64;
}

magma_int_t magma_get_cgetri_nb( magma_int_t /*m*/ )
{
    return 64;
}

magma_int_t magma_get_zgetri_nb( magma_int_t /*m*/ )
{
    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gesvd based on m
*/
magma_int_t magma_get_sgesvd_nb( magma_int_t m )
{
    return magma_get_sgebrd_nb( m );
}

magma_int_t magma_get_dgesvd_nb( magma_int_t m )
{
    return magma_get_dgebrd_nb( m );
}

magma_int_t magma_get_cgesvd_nb( magma_int_t m )
{
    return magma_get_cgebrd_nb( m );
}

magma_int_t magma_get_zgesvd_nb( magma_int_t m )
{
    return magma_get_zgebrd_nb( m );
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sygst based on m
*/
magma_int_t magma_get_ssygst_nb_m( magma_int_t /*m*/ )
{
    return 256; //to be updated

    /*
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  4096) return 768;
        else                return 1536;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (m <  2048) return 512;
        else                return 1024;
    }
    else {                     // 1.x
        return 64;
    }
    */
}

magma_int_t magma_get_dsygst_nb_m( magma_int_t /*m*/ )
{
    return 256; //to be updated

    /*
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  2048) return 384;
        else                return 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 512;
    }
    else {                     // 1.x
        return 64;
    }
    */
}

magma_int_t magma_get_chegst_nb_m( magma_int_t /*m*/ )
{
    return 256; //to be updated

    /*
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (m <  2048) return 384;
        else                return 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 512;
    }
    else {                     // 1.x
        return 64;
    }
    */
}

magma_int_t magma_get_zhegst_nb_m( magma_int_t /*m*/ )
{
    return 256; //to be updated

    /*
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        return 384;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 256;
    }
    else {                     // 1.x
        return 64;
    }
    */
}


    /////////////////////////////////////////
    /////////////////////////////////////////
    // Parameters for 2-stage eigensolvers //
    /////////////////////////////////////////
    /////////////////////////////////////////

/* ////////////////////////////////////////////////////////////////////////////
   -- Return gpu_cpu_perf for  2 stage TRD
*/
magma_int_t magma_get_sbulge_gcperf( )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        return 37;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 15000;
    }
    else {                     // 1.x
        return 10000;
    }
}

magma_int_t magma_get_dbulge_gcperf( )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        return 37;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 15000;
    }
    else {                     // 1.x
        return 10000;
    }
}

magma_int_t magma_get_cbulge_gcperf( )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
            return 50;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 15000;
    }
    else {                     // 1.x
        return 10000;
    }
}

magma_int_t magma_get_zbulge_gcperf( )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
            return 50;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 15000;
    }
    else {                     // 1.x
        return 10000;
    }
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Return smlsiz for the divide and conquewr routine dlaex0 dstedx zstedx
*/
magma_int_t magma_get_smlsize_divideconquer()
{
    return 128;
}



/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for  2 stage TRD
*/
magma_int_t magma_get_sbulge_nb( magma_int_t /*m*/, magma_int_t /*nbthreads*/  )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        return 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 64;
    }
    else {                     // 1.x
        return 64;
    }
}

magma_int_t magma_get_dbulge_nb( magma_int_t /*m*/, magma_int_t /*nbthreads*/  )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        return 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 128;
    }
    else {                     // 1.x
        return 64;
    }
}

magma_int_t magma_get_cbulge_nb( magma_int_t /*m*/, magma_int_t nbthreads  )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            return 128;
        else
            return 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 64;
    }
    else {                     // 1.x
        return 64;
    }
}

magma_int_t magma_get_zbulge_nb( magma_int_t /*m*/, magma_int_t nbthreads )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            return 128;
        else
            return 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 64;
    }
    else {                     // 1.x
        return 64;
    }
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Return Vblksiz for  2 stage TRD
*/
magma_int_t magma_sbulge_get_Vblksiz( magma_int_t /*m*/, magma_int_t nb, magma_int_t /*nbthreads*/  )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        return min(nb, 128);
    }
    else {                     // 2.x Fermi or 1.x
        return min(nb, 64);
    }
}

magma_int_t magma_dbulge_get_Vblksiz( magma_int_t /*m*/, magma_int_t nb, magma_int_t /*nbthreads*/  )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        return min(nb, 64);
    }
    else {                     // 2.x Fermi or 1.x
        return min(nb, 64);
    }
}

magma_int_t magma_cbulge_get_Vblksiz( magma_int_t /*m*/, magma_int_t nb, magma_int_t nbthreads )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            return min(nb, 64);
        else
            return min(nb, 32);
    }
    else {                     // 2.x Fermi or 1.x
        return min(nb, 48);
    }
}

magma_int_t magma_zbulge_get_Vblksiz( magma_int_t /*m*/, magma_int_t nb, magma_int_t nbthreads )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            return min(nb, 64);
        else
            return min(nb, 32);
    }
    else {                     // 2.x Fermi or 1.x
        return min(nb, 48);
    }
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for  2 stage TRD_MGPU
*/
magma_int_t magma_get_sbulge_nb_mgpu( magma_int_t /*m*/ )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        return 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 64;
    }
    else {                     // 1.x
        return 64;
    }
}

magma_int_t magma_get_dbulge_nb_mgpu( magma_int_t /*m*/ )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        return 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 64;
    }
    else {                     // 1.x
        return 64;
    }
}

magma_int_t magma_get_cbulge_nb_mgpu( magma_int_t /*m*/ )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        return 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 64;
    }
    else {                     // 1.x
        return 64;
    }
}

magma_int_t magma_get_zbulge_nb_mgpu( magma_int_t /*m*/ )
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        return 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        return 64;
    }
    else {                     // 1.x
        return 64;
    }
}

#endif  // HAVE_CUBLAS

#ifdef __cplusplus
} // extern "C"
#endif
