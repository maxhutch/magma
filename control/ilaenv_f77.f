!   -- MAGMA (version 1.6.1) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      @date January 2015
!      
!      @author Mark Gates

!     Wrapper around LAPACK ilaenv,
!     this takes name and opts as character arrays with length,
!     then calls LAPACK ilaenv with them as character strings.

      integer function magmaf77_ilaenv(
     &      ispec, name, namelen, opts, optslen, n1, n2, n3, n4 )
         implicit none
         character(*) name, opts
         integer ispec, namelen, optslen, n1, n2, n3, n4
         
!        external functions
         integer ilaenv
         
!        print *, 'name', name, "=", name(1:namelen), "=", namelen
!        print *, 'opts', opts, "=", opts(1:optslen), "=", optslen
!        print *, 'args', ispec, n1, n2, n3, n4
         
         magmaf77_ilaenv = ilaenv( ispec, name(1:namelen),
     &                             opts(1:optslen), n1, n2, n3, n4 )
      end function
