/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/control/magma_zsort.cpp normal z -> s, Mon May  2 23:30:54 2016
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from
//  the IO functions provided by MatrixMarket

#include "magmasparse_internal.h"


#define SWAP(a, b)  { tmp = val[a]; val[a] = val[b]; val[b] = tmp; }
#define SWAPM(a, b) { tmpv = val[a]; val[a] = val[b]; val[b] = tmpv;  \
                      tmpc = col[a]; col[a] = col[b]; col[b] = tmpc;  \
                      tmpr = row[a]; row[a] = row[b]; row[b] = tmpr; }

/**
    Purpose
    -------

    Sorts an array of values in increasing order.

    Arguments
    ---------

    @param[in,out]
    x           float*
                array to sort

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_ssort(
    float *x,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    float temp;
    magma_index_t pivot,j,i;
    
    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( MAGMA_S_ABS(x[i]) <= MAGMA_S_ABS(x[pivot]) && i<last )
                i++;
            while( MAGMA_S_ABS(x[j]) > MAGMA_S_ABS(x[pivot]) )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        CHECK( magma_ssort( x, first, j-1, queue ));
        CHECK( magma_ssort( x, j+1, last, queue ));
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Sorts an array of values in increasing order.

    Arguments
    ---------

    @param[in,out]
    x           float*
                array to sort
                
    @param[in,out]
    col         magma_index_t*
                Target array, will be modified during operation.
                
    @param[in,out]
    row         magma_index_t*
                Target array, will be modified during operation.

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_smsort(
    float *x,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    float temp;
    magma_index_t pivot,j,i, tmpcol, tmprow;
    
    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( MAGMA_S_ABS(x[i]) <= MAGMA_S_ABS(x[pivot]) && i<last )
                i++;
            while( MAGMA_S_ABS(x[j]) > MAGMA_S_ABS(x[pivot]) )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
                tmpcol = col[i];
                col[i] = col[j];
                col[j] = tmpcol;
                tmprow = row[i];
                row[i] = row[j];
                row[j] = tmprow;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        CHECK( magma_smsort( x, col, row, first, j-1, queue ));
        CHECK( magma_smsort( x, col, row, j+1, last, queue ));
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Sorts an array of integers in increasing order.

    Arguments
    ---------

    @param[in,out]
    x           magma_index_t*
                array to sort

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_sindexsort(
    magma_index_t *x,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_index_t pivot,j,temp,i;

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( x[i]<=x[pivot] && i<last )
                i++;
            while( x[j]>x[pivot] )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        CHECK( magma_sindexsort( x, first, j-1, queue ));
        CHECK( magma_sindexsort( x, j+1, last, queue ));
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Sorts an array of integers, updates a respective array of values.

    Arguments
    ---------

    @param[in,out]
    x           magma_index_t*
                array to sort
                
    @param[in,out]
    y           float*
                array to sort

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_sindexsortval(
    magma_index_t *x,
    float *y,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_index_t pivot,j,temp,i;
    float tempval;

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( x[i]<=x[pivot] && i<last )
                i++;
            while( x[j]>x[pivot] )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
                tempval = y[i];
                y[i] = y[j];
                y[j] = tempval;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        
        tempval=y[pivot];
        y[pivot]=y[j];
        y[j]=tempval;
        CHECK( magma_sindexsortval( x, y, first, j-1, queue ));
        CHECK( magma_sindexsortval( x, y, j+1, last, queue ));
    }
cleanup:
    return info;
}



/**
    Purpose
    -------

    Identifies the kth smallest/largest element in an array and reorders 
    such that these elements come to the front. The related arrays col and row
    are also reordered.

    Arguments
    ---------
    
    @param[in,out]
    val         float*
                Target array, will be modified during operation.
                
    @param[in,out]
    col         magma_index_t*
                Target array, will be modified during operation.
                
    @param[in,out]
    row         magma_index_t*
                Target array, will be modified during operation.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in]
    k           magma_int_t
                Element to be identified (largest/smallest).
                
    @param[in]
    r           magma_int_t
                rule how to sort: '1' -> largest, '0' -> smallest
                
    @param[out]
    element     float*
                location of the respective element
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_smorderstatistics(
    float *val,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    float *element,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t i, st;
    float tmpv;
    magma_index_t tmpc, tmpr;
    if( r == 0 ){
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_s_isnan_inf( val[i]) ) {
                printf("error: array contains %f + %fi.\n", MAGMA_S_REAL(val[i]), MAGMA_S_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_S_ABS(val[i]) > MAGMA_S_ABS(val[length-1]) ){
                continue;
            }
            SWAPM(i, st);
            st++;
        }
     
        SWAPM(length-1, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_smorderstatistics( val, col, row, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_smorderstatistics( val+st, col+st, row+st, length-st, k-st, r, element, queue ));
        }
    } else {
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_s_isnan_inf( val[i]) ) {
                printf("error: array contains %f + %fi.\n", MAGMA_S_REAL(val[i]), MAGMA_S_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_S_ABS(val[i]) < MAGMA_S_ABS(val[length-1]) ){
                continue;
            }
            SWAPM(i, st);
            st++;
        }
     
        SWAPM(length-1, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_smorderstatistics( val, col, row, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_smorderstatistics( val+st, col+st, row+st, length-st, k-st, r, element, queue ));
        }
    }
    
cleanup:
    return info;
}



/**
    Purpose
    -------

    Identifies the kth smallest/largest element in an array.

    Arguments
    ---------
    
    @param[in,out]
    val         float*
                Target array, will be modified during operation.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in]
    k           magma_int_t
                Element to be identified (largest/smallest).
                
    @param[in]
    r           magma_int_t
                rule how to sort: '1' -> largest, '0' -> smallest
                
    @param[out]
    element     float*
                location of the respective element
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_sorderstatistics(
    float *val,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    float *element,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t i, st;
    float tmp;
    if( r == 0 ){
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_s_isnan_inf( val[i]) ) {
                printf("error: array contains %f + %fi.\n", MAGMA_S_REAL(val[i]), MAGMA_S_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_S_ABS(val[i]) > MAGMA_S_ABS(val[length-1]) ){
                continue;
            }
            SWAP(i, st);
            st++;
        }
     
        SWAP(length-1, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_sorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_sorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    } else {
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_s_isnan_inf( val[i]) ) {
                printf("error: array contains %f + %fi.\n", MAGMA_S_REAL(val[i]), MAGMA_S_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_S_ABS(val[i]) < MAGMA_S_ABS(val[length-1]) ){
                continue;
            }
            SWAP(i, st);
            st++;
        }
     
        SWAP(length-1, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_sorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_sorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    }
    
cleanup:
    return info;
}

