/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>



/**
    Purpose
    -------
    Inserts for the iterative dynamic ILU an new element in the (empty) place 
    where an element was deleted in the beginning of the loop. 
    In the new matrix, the added elements will then always be located at the 
    beginning of each row.
    
    More precisely,
    it inserts the new value in the value-pointer where the old element was
    located, changes the columnindex to the new index, modifies the row-pointer
    to point the this element, and sets the linked list element to the element
    where the row pointer pointed to previously.

    Arguments
    ---------

    @param[in]
    tri         magma_int_t
                info==0: lower trianguler, info==1: upper triangular.
                
    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    rm_loc      magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    rm_loc2     magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    LU_new      magma_z_matrix
                Elements that will be inserted stored in COO format (unsorted).

    @param[in,out]
    L           magma_z_matrix*
                matrix where new elements are inserted. 
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in,out]
    U           magma_z_matrix*
                matrix where new elements are inserted. 
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    rowlock     omp_lock_t*
                OMP lock for the rows.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicic_insert_LU(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_index_t *rm_loc2,
    magma_z_matrix *LU_new,
    magma_z_matrix *L,
    magma_z_matrix *U,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaDoubleComplex *val;
    magma_index_t *col;
    magma_index_t *rowidx;
        
    magmaDoubleComplex element;
    magma_int_t j,jn;
    
    magma_int_t i=0;
    magma_int_t num_insert = 0;
    int loc_i=0;
    int abort = 0;
    magma_int_t *success;
    magma_int_t *insert_loc;
    magma_int_t num_threads = 0;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }

    CHECK( magma_imalloc_cpu( &success, num_threads*8 ));
    CHECK( magma_imalloc_cpu( &insert_loc, num_threads*8 ));
    //omp_lock_t rowlock[LU->num_rows];
    #pragma omp parallel for
    for (magma_int_t r=0; r<omp_get_num_threads(); r++){
        success[r*8]= ( r<num_rm ) ? 1 : -1;
        insert_loc[r*8] = -1;
    }

    if(num_rm>=LU_new->nnz){
        printf("error: try to remove too many elements\n.");
        goto cleanup;
    }
    // identify num_rm-th largest elements and bring them to the front
    CHECK( magma_zmalloc_cpu( &val, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &rowidx, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &col, LU_new->nnz ));

    #pragma omp parallel for
    for( magma_int_t r=0; r<LU_new->nnz; r++ ) {
        col[r] = LU_new->col[r];
        rowidx[r] = LU_new->rowidx[r];
        val[r] = LU_new->val[r];
    }

   // this is usually sufficient to have the large elements in front
    CHECK( magma_zmorderstatistics(
    val, col, rowidx, LU_new->nnz, num_rm + (LU_new->nnz-num_rm)*.1,  1, &element, queue ) );

    CHECK( magma_zmorderstatistics(
    val, col, rowidx, num_rm + (LU_new->nnz-num_rm)*.1, num_rm, 1, &element, queue ) );
    // insert the new elements
    // has to be sequential

    //#pragma omp parallel for private(loc_i) schedule(static,3) shared(num_insert)
    for(int loc_i=0; loc_i<LU_new->nnz; loc_i++ ) {
        magma_int_t tid = omp_get_thread_num();
        if( success[ tid*8 ] > -1 ){
            if( success[ tid*8 ] == 1 ){
                #pragma omp critical(num_insert)
                {
                    insert_loc[ tid*8 ] = num_insert;
                    num_insert++;
                }
                success[ tid*8 ] = 0;
            }
            if( insert_loc[ tid*8 ] >= num_rm ){
                // enough elements added
                success[ tid*8 ] = -1;
            }
            if( success[ tid*8 ] > -1 ){
                magma_int_t loc = rm_loc[ insert_loc[ tid*8 ] ];
                magma_index_t new_row = rowidx[ loc_i ]; 
                

                
                #pragma omp critical(rowlock__)
                {
                    omp_set_lock( &(rowlock[new_row]) );
                }
                magma_index_t new_col = col[ loc_i ];
                magma_index_t old_rowstart = L->row[ new_row ];
                
                if( new_col < L->col[ old_rowstart ] ){
                 //   printf("insert in L: (%d,%d)\n", new_row, new_col);
                    L->row[ new_row ] = loc;
                    L->list[ loc ] = old_rowstart;
                    L->rowidx[ loc ] = new_row;
                    L->col[ loc ] = new_col;
                    L->val[ loc ] = MAGMA_Z_ZERO;
                    success[ tid*8 ] = 1;
                }
                else if( new_col == L->col[ old_rowstart ] ){
                    ; //printf("tried to insert duplicate!\n");
                }
                else{
        
                    j = old_rowstart;
                    jn = L->list[j];
                    // this will finish, as we consider the lower triangular
                    // and we always have the diagonal!
                    while( j!=0 ){
                        if( L->col[jn]==new_col ){
                            //printf("tried to insert duplicate!\n");
                            j=0; //break;
                        }else if( L->col[jn]>new_col ){
                            L->list[j]=loc;
                            L->list[loc]=jn;
                            L->rowidx[ loc ] = new_row;
                            L->col[ loc ] = new_col;
                            L->val[ loc ] = MAGMA_Z_ZERO;
                            success[ tid*8 ] = 1;
                            j=0; //break;
                            
                        } else{
                            j=jn;
                            jn=L->list[jn];
                        }
                    }
                }
                //#pragma omp critical(rowlock__)
                omp_unset_lock( &(rowlock[new_row]) );
            }
        }
    }// abort
    printf("L done\n");
    // Part for U
    num_insert = 0;
    #pragma omp parallel for
    for (magma_int_t r=0; r<omp_get_num_threads(); r++){
        success[r*8]= ( r<num_rm ) ? 1 : -1;
        insert_loc[r*8] = -1;
    }
    //#pragma omp parallel for private(loc_i) schedule(static,3) shared(num_insert)
    for(int loc_i=0; loc_i<LU_new->nnz; loc_i++ ) {

        magma_int_t tid = omp_get_thread_num();
        if( success[ tid*8 ] > -1 ){
            if( success[ tid*8 ] == 1 ){
                #pragma omp critical(num_insert)
                {
                    insert_loc[ tid*8 ] = num_insert;
                    num_insert++;
                }
                success[ tid*8 ] = 0;

            }

            if( insert_loc[ tid*8 ] >= num_rm ){
                // enough elements added
                success[ tid*8 ] = -1;
            }

            if( success[ tid*8 ] > -1 ){
                magma_int_t loc = rm_loc2[ insert_loc[ tid*8 ] ];
                magma_index_t new_row = col[ loc_i ]; 
                

                
                #pragma omp critical(rowlock__)
                {
                    omp_set_lock( &(rowlock[new_row]) );
                }
                magma_index_t new_col = rowidx[ loc_i ];
                magma_index_t old_rowstart = U->row[ new_row ];
                jn = old_rowstart;
                // diagonal element always exists!
                while( jn!=0 ){
                    j=jn;
                    jn=U->list[jn];
                    if( jn == 0 ){
                        U->list[j]=loc;
                        U->list[loc]=jn;
                        U->rowidx[ loc ] = new_row;
                        U->col[ loc ] = new_col;
                        U->val[ loc ] = MAGMA_Z_ONE;
                        success[ tid*8 ] = 1;
                        jn=0; //break;
                    }else if( U->col[jn]==new_col ){
                        jn=0; //break;
                    }else if( U->col[jn]>new_col ){
                        U->list[j]=loc;
                        U->list[loc]=jn;
                        U->rowidx[ loc ] = new_row;
                        U->col[ loc ] = new_col;
                        U->val[ loc ] = MAGMA_Z_ONE;
                        success[ tid*8 ] = 1;
                        jn=0; //break;
                        
                    } 
                }
                //#pragma omp critical(rowlock__)
                omp_unset_lock( &(rowlock[new_row]) );
            }
        }
    }// abort
    printf("U done\n");
    
cleanup:
    magma_free_cpu( success );
    magma_free_cpu( insert_loc );
    
    magma_free_cpu( val );
    magma_free_cpu( col );
    magma_free_cpu( rowidx );
    
    return info;
}


/**
    Purpose
    -------
    Inserts for the iterative dynamic ILU an new element in the (empty) place 
    where an element was deleted in the beginning of the loop. 
    In the new matrix, the added elements will then always be located at the 
    beginning of each row.
    
    More precisely,
    it inserts the new value in the value-pointer where the old element was
    located, changes the columnindex to the new index, modifies the row-pointer
    to point the this element, and sets the linked list element to the element
    where the row pointer pointed to previously.

    Arguments
    ---------

    @param[in]
    tri         magma_int_t
                info==0: lower trianguler, info==1: upper triangular.
                
    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    rm_loc      magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    LU_new      magma_z_matrix
                Elements that will be inserted stored in COO format (unsorted).

    @param[in,out]
    LU          magma_z_matrix*
                matrix where new elements are inserted. 
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    rowlock     omp_lock_t*
                OMP lock for the rows.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicic_insert(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_z_matrix *LU_new,
    magma_z_matrix *LU,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaDoubleComplex *val;
    magma_index_t *col;
    magma_index_t *rowidx;
        
    magmaDoubleComplex element;
    magma_int_t j,jn;
    
    magma_int_t i=0;
    magma_int_t num_insert = 0;
    int loc_i=0;
    int abort = 0;
    magma_int_t *success;
    magma_int_t *insert_loc;
    magma_int_t num_threads = 0;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    
    CHECK( magma_imalloc_cpu( &success, num_threads*8 ));
    CHECK( magma_imalloc_cpu( &insert_loc, num_threads*8 ));
    //omp_lock_t rowlock[LU->num_rows];
    #pragma omp parallel for
    for (magma_int_t r=0; r<omp_get_num_threads(); r++){
        success[r*8]= ( r<num_rm ) ? 1 : -1;
        insert_loc[r*8] = -1;
    }
    if(num_rm>=LU_new->nnz){
        printf("error: try to remove too many elements\n.");
        goto cleanup;
    }
    // identify num_rm-th largest elements and bring them to the front
    CHECK( magma_zmalloc_cpu( &val, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &rowidx, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &col, LU_new->nnz ));
    #pragma omp parallel for
    for( magma_int_t r=0; r<LU_new->nnz; r++ ) {
        col[r] = LU_new->col[r];
        rowidx[r] = LU_new->rowidx[r];
        val[r] = LU_new->val[r];
    }
    // this is usually sufficient to have the large elements in front
    CHECK( magma_zmorderstatistics(
    val, col, rowidx, LU_new->nnz, num_rm + (LU_new->nnz-num_rm)*.1,  1, &element, queue ) );

    CHECK( magma_zmorderstatistics(
    val, col, rowidx, num_rm + (LU_new->nnz-num_rm)*.1, num_rm, 1, &element, queue ) );
    // insert the new elements
    // has to be sequential

    //#pragma omp parallel for private(loc_i) schedule(static,3) shared(num_insert)
    for(int loc_i=0; loc_i<LU_new->nnz; loc_i++ ) {
        magma_int_t tid = omp_get_thread_num();
        if( success[ tid*8 ] > -1 ){
            if( success[ tid*8 ] == 1 ){
                #pragma omp critical(num_insert)
                {
                    insert_loc[ tid*8 ] = num_insert;
                    num_insert++;
                }
                success[ tid*8 ] = 0;
            }
            if( insert_loc[ tid*8 ] >= num_rm ){
                // enough elements added
                success[ tid*8 ] = -1;
            }
            if( success[ tid*8 ] > -1 ){
                magma_int_t loc = rm_loc[ insert_loc[ tid*8 ] ];
                magma_index_t new_row = rowidx[ loc_i ]; 
                

                
                #pragma omp critical(rowlock__)
                {
                    omp_set_lock( &(rowlock[new_row]) );
                }
                magma_index_t new_col = col[ loc_i ];
                magma_index_t old_rowstart = LU->row[ new_row ];
                
                           //                         printf("tid*8:%d loc_i:%d loc_num_insert:%d num_rm:%d target loc:%d  element (%d,%d)\n",
                           //     tid*8, loc_i, insert_loc[ tid*8 ], num_rm, loc, new_row, new_col); fflush(stdout);
                           //     printf("-->(%d,%d)\n", new_row, new_col); fflush(stdout);

                

                if( new_col < LU->col[ old_rowstart ] ){
                    LU->row[ new_row ] = loc;
                    LU->list[ loc ] = old_rowstart;
                    LU->rowidx[ loc ] = new_row;
                    LU->col[ loc ] = new_col;
                    LU->val[ loc ] = MAGMA_Z_ZERO;
                    success[ tid*8 ] = 1;
                }
                else if( new_col == LU->col[ old_rowstart ] ){
                    ; //printf("tried to insert duplicate!\n");
                }
                else{
        
                    j = old_rowstart;
                    jn = LU->list[j];
                    // this will finish, as we consider the lower triangular
                    // and we always have the diagonal!
                    while( j!=0 ){
                        if( LU->col[jn]==new_col ){
                            //printf("tried to insert duplicate!\n");
                            j=0; //break;
                        }else if( LU->col[jn]>new_col ){
                            
                            
                            printf("insert: (%d,%d)\n", new_row, new_col);
                            LU->list[j]=loc;
                            LU->list[loc]=jn;
                            LU->rowidx[ loc ] = new_row;
                            LU->col[ loc ] = new_col;
                            LU->val[ loc ] = MAGMA_Z_ZERO;
                            success[ tid*8 ] = 1;
                            j=0; //break;
                            
                        } else{
                            j=jn;
                            jn=LU->list[jn];
                        }
                    }
                }
                //#pragma omp critical(rowlock__)
                //{
                omp_unset_lock( &(rowlock[new_row]) );
            }
        }
    }// abort
    
cleanup:
    magma_free_cpu( success );
    magma_free_cpu( insert_loc );
    
    magma_free_cpu( val );
    magma_free_cpu( col );
    magma_free_cpu( rowidx );
    
    return info;
}


/**
    Purpose
    -------
    Inserts for the iterative dynamic ILU an new element in the (empty) place 
    where an element was deleted in the beginning of the loop. 
    In the new matrix, the added elements will then always be located at the 
    beginning of each row.
    
    More precisely,
    it inserts the new value in the value-pointer where the old element was
    located, changes the columnindex to the new index, modifies the row-pointer
    to point the this element, and sets the linked list element to the element
    where the row pointer pointed to previously.

    Arguments
    ---------

    @param[in]
    tri         magma_int_t
                info==0: lower trianguler, info==1: upper triangular.
                
    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    rm_loc      magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    LU_new      magma_z_matrix
                Elements that will be inserted stored in COO format (unsorted).

    @param[in,out]
    LU          magma_z_matrix*
                matrix where new elements are inserted. 
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    rowlock     omp_lock_t*
                OMP lock for the rows.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicic_insert_U(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_z_matrix *LU_new,
    magma_z_matrix *LU,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaDoubleComplex *val;
    magma_index_t *col;
    magma_index_t *rowidx;
        
    magmaDoubleComplex element;
    magma_int_t j,jn;
    
    magma_int_t i=0;
    magma_int_t num_insert = 0;
    int loc_i=0;
    int abort = 0;
    magma_int_t *success;
    magma_int_t *insert_loc;
    magma_int_t num_threads = 0;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    
    CHECK( magma_imalloc_cpu( &success, num_threads*8 ));
    CHECK( magma_imalloc_cpu( &insert_loc, num_threads*8 ));
    //omp_lock_t rowlock[LU->num_rows];
    #pragma omp parallel for
    for (magma_int_t r=0; r<omp_get_num_threads(); r++){
        success[r*8]= ( r<num_rm ) ? 1 : -1;
        insert_loc[r*8] = -1;
    }
    if(num_rm>=LU_new->nnz){
        printf("error: try to remove too many elements\n.");
        goto cleanup;
    }
    // identify num_rm-th largest elements and bring them to the front
    CHECK( magma_zmalloc_cpu( &val, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &rowidx, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &col, LU_new->nnz ));
    #pragma omp parallel for
    for( magma_int_t r=0; r<LU_new->nnz; r++ ) {
        col[r] = LU_new->col[r];
        rowidx[r] = LU_new->rowidx[r];
        val[r] = LU_new->val[r];
    }
    // this is usually sufficient to have the large elements in front
    CHECK( magma_zmorderstatistics(
    val, col, rowidx, LU_new->nnz, num_rm + (LU_new->nnz-num_rm)*.1,  1, &element, queue ) );

    CHECK( magma_zmorderstatistics(
    val, col, rowidx, num_rm + (LU_new->nnz-num_rm)*.1, num_rm, 1, &element, queue ) );
    // insert the new elements
    // has to be sequential

    //#pragma omp parallel for private(loc_i) schedule(static,3) shared(num_insert)
    for(int loc_i=0; loc_i<LU_new->nnz; loc_i++ ) {
        magma_int_t tid = omp_get_thread_num();
        if( success[ tid*8 ] > -1 ){
            if( success[ tid*8 ] == 1 ){
                #pragma omp critical(num_insert)
                {
                    printf("->next element->");
                    insert_loc[ tid*8 ] = num_insert;
                    num_insert++;
                }
                success[ tid*8 ] = 0;
            }
            if( insert_loc[ tid*8 ] >= num_rm ){
                // enough elements added
                success[ tid*8 ] = -1;
            }
            if( success[ tid*8 ] > -1 ){
                magma_int_t loc = rm_loc[ insert_loc[ tid*8 ] ];
                magma_index_t new_row = col[ loc_i ]; 
                

                
                #pragma omp critical(rowlock__)
                {
                    omp_set_lock( &(rowlock[new_row]) );
                }
                magma_index_t new_col = rowidx[ loc_i ];
                magma_index_t old_rowstart = LU->row[ new_row ];
                
                           //                         printf("tid*8:%d loc_i:%d loc_num_insert:%d num_rm:%d target loc:%d  element (%d,%d)\n",
                           //     tid*8, loc_i, insert_loc[ tid*8 ], num_rm, loc, new_row, new_col); fflush(stdout);
                                printf("-->(%d,%d)\t", new_row, new_col); fflush(stdout);

                

                if( new_col < LU->col[ old_rowstart ] ){
                    LU->row[ new_row ] = loc;
                    LU->list[ loc ] = old_rowstart;
                    LU->rowidx[ loc ] = new_row;
                    LU->col[ loc ] = new_col;
                    LU->val[ loc ] = MAGMA_Z_ONE;
                    success[ tid*8 ] = 1;
                }
                else if( new_col == LU->col[ old_rowstart ] ){
                    printf("tried to insert duplicate!\n"); fflush(stdout);
                }
                else{
        
                    j = old_rowstart;
                    jn = LU->list[j];
                    // this will finish, as we consider the lower triangular
                    // and we always have the diagonal!
                    magma_int_t breakpoint = 0;
                    //if ( new_row == 0 ) {
                    //    LU->row[new_row+1];
                    //}
                    while( j!=breakpoint ){
                        if( LU->col[jn]==new_col ){
                            printf("tried to insert duplicate!\n");
                            j=breakpoint; //break;
                        }else if( LU->col[jn]>new_col ){
                            
                            
                            printf("insert: (%d,%d)\t", new_row, new_col); fflush(stdout);
                            LU->list[j]=loc;
                            LU->list[loc]=jn;
                            LU->rowidx[ loc ] = new_row;
                            LU->col[ loc ] = new_col;
                            LU->val[ loc ] = MAGMA_Z_ONE;
                            success[ tid*8 ] = 1;
                            j=breakpoint; //break;
                            
                        } else{
                            j=jn;
                            jn=LU->list[jn];
                        }
                    }printf("done\n"); fflush(stdout);
                }
                //#pragma omp critical(rowlock__)
                //{
                omp_unset_lock( &(rowlock[new_row]) );
            }
        }
    }// abort
    
cleanup:
    magma_free_cpu( success );
    magma_free_cpu( insert_loc );
    
    magma_free_cpu( val );
    magma_free_cpu( col );
    magma_free_cpu( rowidx );
    
    return info;
}


/**
    Purpose
    -------
    This routine removes matrix entries from the structure that are smaller
    than the threshold.

    Arguments
    ---------

    @param[out]
    thrs        magmaDoubleComplex*
                Thrshold for removing elements.
                
    @param[out]
    num_rm      magma_int_t*
                Number of Elements that have been removed.
                
    @param[in,out]
    LU          magma_z_matrix*
                Current ILU approximation where the identified smallest components
                are deleted.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[out]
    rm_loc      magma_index_t*
                List containing the locations of the elements deleted.
                
    @param[in]
    rowlock     omp_lock_t*
                OMP lock for the rows.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicilu_rm_thrs(
    magmaDoubleComplex *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *LU,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    
    omp_lock_t counter;
    omp_init_lock(&(counter));
    // never forget elements
    // magma_int_t offset = LU_new->diameter;
    // never forget last rm
    magma_int_t offset = 0;
    
    #pragma omp parallel for
    for( magma_int_t r=0; r < LU->num_rows; r++ ) {
        magma_int_t i = LU->row[r];
        magma_int_t lasti=i;
        magma_int_t nexti=LU->list[i];
        while( nexti!=0 ){
            if( MAGMA_Z_ABS( LU->val[ i ] ) <  MAGMA_Z_ABS(*thrs) ){
                // the condition nexti!=0 esures we never remove the diagonal
                    LU->val[ i ] = MAGMA_Z_ZERO;
                    LU->list[ i ] = -1;
                    if( LU->col[ i ] == r ){
                        printf("error: try to rm diagonal in L.\n");   
                    }
                    
                    omp_set_lock(&(counter));
                    rm_loc[ count_rm ] = i; 
                    // keep it as potential fill-in candidate
                    LU_new->col[ count_rm+offset ] = LU->col[ i ];
                    LU_new->rowidx[ count_rm+offset ] = r;
                    LU_new->val[ count_rm+offset ] = MAGMA_Z_ZERO; // MAGMA_Z_MAKE(1e-14,0.0);
                   // printf("rm: (%d,%d)\n", r, LU->col[ i ]); fflush(stdout);
                    count_rm++;
                    omp_unset_lock(&(counter));
                    // either the headpointer or the linked list has to be changed
                    // headpointer if the deleted element was the first element in the row
                    // linked list to skip this element otherwise
                    if( LU->row[r] == i ){
                            LU->row[r] = nexti;
                            lasti=i;
                            i = nexti;
                            nexti = LU->list[nexti];
                    }
                    else{
                        LU->list[lasti] = nexti;
                        i = nexti;
                        nexti = LU->list[nexti];
                    }
            }
            else{
                lasti = i;
                i = nexti;
                nexti = LU->list[nexti];
            }
            
        }
    }
    // never forget elements
    // LU_new->diameter = count_rm+LU_new->diameter;
    // not forget the last rm
    LU_new->diameter = count_rm;
    LU_new->nnz = LU_new->diameter;
    *num_rm = count_rm;

    omp_destroy_lock(&(counter));
    return info;
}


/**
    Purpose
    -------
    This routine removes matrix entries from the structure that are smaller
    than the threshold.

    Arguments
    ---------

    @param[out]
    thrs        magmaDoubleComplex*
                Thrshold for removing elements.
                
    @param[out]
    num_rm      magma_int_t*
                Number of Elements that have been removed.
                
    @param[in,out]
    U          magma_z_matrix*
                Current ILU approximation where the identified smallest components
                are deleted.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[out]
    rm_loc      magma_index_t*
                List containing the locations of the elements deleted.
                
    @param[in]
    rowlock     omp_lock_t*
                OMP lock for the rows.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicilu_rm_thrs_U(
    magmaDoubleComplex *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *U,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    
    omp_lock_t counter;
    omp_init_lock(&(counter));
    
    // set elements to zero
    #pragma omp parallel for
    for( magma_int_t rm=0; rm < *num_rm; rm++ ) {
        magma_int_t rm_row = LU_new->col[rm];
        magma_int_t rm_col = LU_new->rowidx[rm];
        magma_int_t el = U->row[rm_row];
        magma_int_t success = 0;
     //   printf("rm: (%d,%d)\t", rm_row, rm_col); fflush(stdout);
      //  printf(" el:%d\t", el);                    fflush(stdout);
        while( success == 0 ){
            if( U->col[el] == rm_col ){
                U->val[el] = MAGMA_Z_ZERO;
                success = 1;
            //    printf("done.\n"); fflush(stdout);
            }
            el = U->list[el];
            //printf("Ucol:%d->%d  ", el, U->col[el]);
        }
    }
  //  printf("first part done.\n"); fflush(stdout);
        
    #pragma omp parallel for
    for( magma_int_t r=0; r < U->num_rows; r++ ) {
        magma_int_t lasti = U->row[r];
        magma_int_t i=U->list[lasti];
        magma_int_t nexti=U->list[i];
        while( i!=0 ){
            if( MAGMA_Z_ABS( U->val[ i ] ) == MAGMA_Z_ZERO ){
                // the condition nexti!=0 ensures we never remove the diagonal
                    U->list[ i ] = -1;
                    if( U->col[ i ] == r ){
                        printf("error: try to rm diagonal.\n");   
                    }
                    
                    omp_set_lock(&(counter));
                    rm_loc[ count_rm ] = i; 
                    // keep it as potential fill-in candidate
                   // printf("rm: (%d,%d)\n", r, U->col[ i ]); fflush(stdout);
                    count_rm++;
                    omp_unset_lock(&(counter));
                    
                    // either the headpointer or the linked list has to be changed
                    // headpointer if the deleted element was the first element in the row
                    // linked list to skip this element otherwise

                    U->list[lasti] = nexti;
                    i = nexti;
                    nexti = U->list[nexti];
            }
            else{
                lasti = i;
                i = nexti;
                nexti = U->list[nexti];
            }
            
        }
    }
     //   printf("second part done.\n"); fflush(stdout);
    omp_destroy_lock(&(counter));
    return info;
}



/**
    Purpose
    -------
    This routine removes matrix entries from the structure that are smaller
    than the threshold.

    Arguments
    ---------

    @param[out]
    thrs        magmaDoubleComplex*
                Thrshold for removing elements.
                
    @param[out]
    num_rm      magma_int_t*
                Number of Elements that have been removed.
                
    @param[in,out]
    L           magma_z_matrix*
                Current L approximation where the identified smallest components
                are deleted.
                
    @param[in,out]
    U           magma_z_matrix*
                Current U approximation where the identified smallest components
                are deleted.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[out]
    rm_loc      magma_index_t*
                List containing the locations of the elements deleted.
                
    @param[in]
    rowlock     omp_lock_t*
                OMP lock for the rows.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicilu_rm_thrs_LU(
    magmaDoubleComplex *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    
    omp_lock_t counter;
    omp_init_lock(&(counter));
    // never forget elements
    // magma_int_t offset = LU_new->diameter;
    // never forget last rm
    magma_int_t offset = 0;
    
    //#pragma omp parallel for
    for( magma_int_t r=0; r < L->num_rows; r++ ) {
        magma_int_t i = L->row[r];
        magma_int_t lasti=i;
        magma_int_t nexti=L->list[i];
        while( nexti!=0 ){
            if( MAGMA_Z_ABS( L->val[ i ] ) <  MAGMA_Z_ABS(*thrs) ){
                // the condition nexti!=0 esures we never remove the diagonal
                    L->val[ i ] = MAGMA_Z_ZERO;
                    L->list[ i ] = -1;
                    if( L->col[ i ] == r ){
                        printf("error: try to rm diagonal.\n");   
                    }
                    
                    omp_set_lock(&(counter));
                    rm_loc[ count_rm ] = i; 
                    // keep it as potential fill-in candidate]
                    magma_int_t rm_col = L->col[ i ];
                    magma_int_t rm_row = r;
                    LU_new->col[ count_rm+offset ] = rm_col;
                    LU_new->rowidx[ count_rm+offset ] = r;
                    LU_new->val[ count_rm+offset ] = MAGMA_Z_ZERO; // MAGMA_Z_MAKE(1e-14,0.0);
                    count_rm++;
                    omp_unset_lock(&(counter));
                    
                    magma_int_t i_U = i;
                    
                    // either the headpointer or the linked list has to be changed
                    // headpointer if the deleted element was the first element in the row
                    // linked list to skip this element otherwise
                    if( L->row[r] == i ){
                            L->row[r] = nexti;
                            lasti=i;
                            i = nexti;
                            nexti = L->list[nexti];
                    }
                    else{
                        L->list[lasti] = nexti;
                        i = nexti;
                        nexti = L->list[nexti];
                    }
                    
                    // now also update U
                    magma_int_t nexti_U = U->list[ i_U ];
                    if( U->row[ rm_col ] == i_U ){
                            U->row[ rm_col ] = nexti_U;
                    }
                    else{
                        while( nexti_U != 0 ){
                            if( U->col[ nexti_U ] == rm_row ){
                                U->list[ i_U ] = U->list[ nexti_U ];
                            }
                            i_U = nexti_U;
                            nexti_U = U->list[ i_U ];
                        }
                    }
                    
            }
            else{
                lasti = i;
                i = nexti;
                nexti = L->list[nexti];
            }
            
        }
    }
    // never forget elements
    // LU_new->diameter = count_rm+LU_new->diameter;
    // not forget the last rm
    LU_new->diameter = count_rm;
    LU_new->nnz = LU_new->diameter;
    *num_rm = count_rm;

    omp_destroy_lock(&(counter));
    return info;
}




/**
    Purpose
    -------
    This routine computes the threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.
                
    @param[in,out]
    LU          magma_z_matrix*
                Current ILU approximation.
                
    @param[out]
    thrs        magmaDoubleComplex*
                Size of the num_rm-th smallest element.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicilu_set_thrs(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magmaDoubleComplex *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magmaDoubleComplex element;
    magmaDoubleComplex *val;
    const magma_int_t ione = 1;
    
    CHECK( magma_zmalloc_cpu( &val, LU->nnz ));
    blasf77_zcopy(&LU->nnz, LU->val, &ione, val, &ione );
    
    // identify num_rm-th smallest element
    CHECK( magma_zorderstatistics(
    val, LU->nnz, num_rm, 0, &element, queue ) );
    //CHECK( magma_zsort( val, 0, LU->nnz, queue ) );
    //element = val[num_rm];
    *thrs = element;

cleanup:
    magma_free_cpu( val );
    return info;
}


/**
    Purpose
    -------
    This function does an iterative ILU sweep.

    Arguments
    ---------
    
    @param[in]
    A           magma_int_t
                System matrix A in CSR.

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in,out]
    LU          magma_z_matrix*
                Current ILU approximation 
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicic_sweep(
    magma_z_matrix A,
    magma_z_matrix *LU,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<LU->nnz; e++){
        if( LU->list[e]!=-1){
            magma_int_t i,j,icol,jcol,jold;
            
            magma_index_t row = LU->rowidx[ e ];
            magma_index_t col = LU->col[ e ];
            // as we look at the lower triangular, col<=row
            
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A.row[row]; i<A.row[row+1]; i++){
                if( A.col[i] == col ){
                    A_e = A.val[i];
                }
            }
            
            //now do the actual iteration
            i = LU->row[ row ]; 
            j = LU->row[ col ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                jold = j;
                icol = LU->col[i];
                jcol = LU->col[j];
                if( icol == jcol ){
                    lsum = LU->val[i] * LU->val[j];
                    sum = sum + lsum;
                    i = LU->list[i];
                    j = LU->list[j];
                }
                else if( icol<jcol ){
                    i = LU->list[i];
                }
                else {
                    j = LU->list[j];
                }
            }while( i!=0 && j!=0 );
            sum = sum - lsum;
            
            // write back to location e
            if ( row == col ){
                LU->val[ e ] = magma_zsqrt( A_e - sum );
            } else {
                LU->val[ e ] =  ( A_e - sum ) / LU->val[jold];
            }
        }// end check whether part of LU
        
    }// end omp parallel section
        
    return info;
}

 



/**
    Purpose
    -------
    This function computes the residuals for the candidates.

    Arguments
    ---------
    
    @param[in]
    A           magma_z_matrix
                System matrix A.
    
    @param[in]
    LU          magma_z_matrix
                Current LU approximation.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicic_residuals(
    magma_z_matrix A,
    magma_z_matrix LU,
    magma_z_matrix *LU_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<LU_new->nnz; e++){
        magma_int_t i,j,icol,jcol;
        
        magma_index_t row = LU_new->rowidx[ e ];
        magma_index_t col = LU_new->col[ e ];
        // as we look at the lower triangular, col<=row
        
        magmaDoubleComplex A_e = MAGMA_Z_ZERO;
        // check whether A contains element in this location
        for( i = A.row[row]; i<A.row[row+1]; i++){
            if( A.col[i] == col ){
                A_e = A.val[i];
            }
        }
        
        //now do the actual iteration
        i = LU.row[ row ]; 
        j = LU.row[ col ];
        magmaDoubleComplex sum = MAGMA_Z_ZERO;
        magmaDoubleComplex lsum = MAGMA_Z_ZERO;
        do{
            lsum = MAGMA_Z_ZERO;
            icol = LU.col[i];
            jcol = LU.col[j];
            if( icol == jcol ){
                lsum = LU.val[i] * LU.val[j];
                sum = sum + lsum;
                i = LU.list[i];
                j = LU.list[j];
            }
            else if( icol<jcol ){
                i = LU.list[i];
            }
            else {
                j = LU.list[j];
            }
        }while( i!=0 && j!=0 );
        
        // write back to location e
        LU_new->val[ e ] =  ( A_e - sum );
        
    }// end omp parallel section
        
    return info;
}




/**
    Purpose
    -------
    This function identifies the candidates.

    Arguments
    ---------
    
    @param[in]
    LU          magma_z_matrix
                Current LU approximation.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicic_candidates(
    magma_z_matrix LU,
    magma_z_matrix *LU_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    //LU_new->nnz = 0;
    
    omp_lock_t counter;
    omp_init_lock(&(counter));
    
    magma_index_t *numadd;
    CHECK( magma_index_malloc_cpu( &numadd, LU.num_rows+1 ));
    
    #pragma omp parallel for
    for( magma_int_t i = 0; i<LU.num_rows+1; i++ ){
        numadd[i] = 0;  
    }
     
 
    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<LU.num_rows; row++){
        magma_index_t start = LU.row[ row ];
        
        magma_index_t lcol1 = start;
        // loop first element over row - only for elements smaller the diagonal
        while( LU.list[lcol1] != 0 ) {
            magma_index_t lcol2 = start;
            // loop second element over row
            while( lcol2 != lcol1 ) {
                // check whether the candidate already is included in LU
                magma_int_t exist = 0;
                magma_index_t col1 = LU.col[lcol1];
                magma_index_t col2 = LU.col[lcol2]; 
                // col1 is always larger as col2

                // we only look at the lower triangular part
                magma_index_t checkrow = col1;
                magma_index_t checkelement = col2;
                magma_index_t check = LU.row[ checkrow ];
                magma_index_t checkcol = LU.col[check];
                while( checkcol <= checkelement && check!=0 ) {
                    if( checkcol == checkelement ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    check = LU.list[ check ];
                    checkcol = LU.col[check];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numadd[ row+1 ]++;
                }
                // go to next element
                lcol2 = LU.list[ lcol2 ];
            }
            // go to next element
            lcol1 = LU.list[ lcol1 ];
        }
    }
    
    // get the total candidate count
    //LU_new->nnz = 0;
    // should become fan-in
    numadd[ 0 ] = LU_new->nnz;
    for( magma_int_t i = 0; i<LU.num_rows; i++ ){
        LU_new->nnz=LU_new->nnz + numadd[ i+1 ];
        numadd[ i+1 ] = LU_new->nnz;
    }

    if( LU_new->nnz > LU.nnz*5 ){
        printf("error: more candidates than space allocated. Increase candidate allocation.\n");
        goto cleanup;
    }
    
    // now insert - in parallel!
    #pragma omp parallel for
    for( magma_index_t row=0; row<LU.num_rows; row++){
        magma_index_t start = LU.row[ row ];
        magma_int_t ladd = 0;
        
        magma_index_t lcol1 = start;
        // loop first element over row
        while( LU.list[lcol1] != 0 ) {
            magma_index_t lcol2 = start;
            // loop second element over row
            while( lcol2 != lcol1 ) {
                // check whether the candidate already is included in LU
                magma_int_t exist = 0;
                
                magma_index_t col1 = LU.col[lcol1];
                magma_index_t col2 = LU.col[lcol2]; 
                // col1 is always larger as col2
 
                // we only look at the lower triangular part
                magma_index_t checkrow = col1;
                magma_index_t checkelement = col2;
                magma_index_t check = LU.row[ checkrow ];
                magma_index_t checkcol = LU.col[check];
                while( checkcol <= checkelement && check!=0 ) {
                    if( checkcol == checkelement ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    check = LU.list[ check ];
                    checkcol = LU.col[check];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allof for parallel insertion
                if( exist == 0 ){
                     //  printf("---------------->>>  candidate at (%d, %d)\n", col2, col1);
                    //add in the next location for this row
                    LU_new->val[ numadd[row] + ladd ] = MAGMA_Z_ZERO; // MAGMA_Z_MAKE(1e-14,0.0);
                    LU_new->rowidx[ numadd[row] + ladd ] = col1;
                    LU_new->col[ numadd[row] + ladd ] = col2;
                    ladd++;
                }
                // go to next element
                lcol2 = LU.list[ lcol2 ];
            }
            // go to next element
            lcol1 = LU.list[ lcol1 ];
        }
    }

cleanup:
    magma_free_cpu( numadd );
    omp_destroy_lock(&(counter));
    return info;
}



/**
    Purpose
    -------
    This function identifies the candidates.

    Arguments
    ---------
    
    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.
                
    @param[in]
    U           magma_z_matrix
                Current upper triangular factor.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmilu0_candidates(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *LU_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    //LU_new->nnz = 0;
    
    
    magma_index_t *numadd;
    CHECK( magma_index_malloc_cpu( &numadd, L.num_rows+1 ));
    
    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
        numadd[i] = 0;  
    }
     
    // how to determine candidates:
    // for each node i, look at any "intermediate" neighbor nodes numbered
    // less, and then see if this neighbor has another neighbor j numbered
    // more than the intermediate; if so, fill in is (i,j) if it is not
    // already nonzero

    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            magma_index_t start2 = U.row[ col1 ];
            magma_index_t el2 = start2;
            while( U.list[ el2 ] != 0 ) {
                magma_index_t col2 = U.col[ el2 ];
                magma_index_t cand_row = max( row, col2);
                magma_index_t cand_col = min(row, col2);
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = el1;
                magma_index_t checkcol = L.col[ checkel ];
                while( checkcol <= cand_col && checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    checkel = L.list[ checkel ];
                    checkcol = L.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numadd[ row+1 ]++;
                }
                el2 = U.list[ el2 ];
            }
            el1 = L.list[ el1 ];
        }
    } //loop over all rows
        
    // get the total candidate count
    LU_new->nnz = 0;
    numadd[ 0 ] = LU_new->nnz;
    for( magma_int_t i = 0; i<L.num_rows; i++ ){
        LU_new->nnz=LU_new->nnz + numadd[ i+1 ];
        numadd[ i+1 ] = LU_new->nnz;
    }
    // printf("cand count:%d\n", LU_new->nnz);
    if( LU_new->nnz > L.nnz*5 ){
        printf("error: more candidates than space allocated. Increase candidate allocation.\n");
        goto cleanup;
    }
    
        // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_int_t ladd = 0;
        
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            magma_index_t start2 = U.row[ col1 ];
            magma_index_t el2 = start2;
            while( U.list[ el2 ] != 0 ) {
                magma_index_t col2 = U.col[ el2 ];
                magma_index_t cand_row = max( row, col2);
                magma_index_t cand_col = min(row, col2);
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = el1;
                magma_index_t checkcol = L.col[ checkel ];
                while( checkcol <= cand_col && checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        // printf("exist ---------------->>>  candidate at (%d, %d)\n", cand_row, cand_col);
                        exist = 1;
                        break;
                    }
                    checkel = L.list[ checkel ];
                    checkcol = L.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    //   printf("---------------->>>  candidate at (%d, %d)\n", cand_row, cand_col);
                    //add in the next location for this row
                    LU_new->val[ numadd[row] + ladd ] = MAGMA_Z_ZERO; // MAGMA_Z_MAKE(1e-14,0.0);
                    LU_new->rowidx[ numadd[row] + ladd ] = cand_row;
                    LU_new->col[ numadd[row] + ladd ] = cand_col;
                    ladd++;
                }
                el2 = U.list[ el2 ];
            }
            el1 = L.list[ el1 ];
        }
    } //loop over all rows
    
    

cleanup:
    magma_free_cpu( numadd );
    return info;
}


#endif  // _OPENMP
