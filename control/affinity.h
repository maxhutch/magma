/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Azzam Haidar
       @author Raffaele Solca

*/
#ifndef MAGMA_AFFINITY_H
#define MAGMA_AFFINITY_H

#ifndef MAGMA_NOAFFINITY

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>

#if __GLIBC_PREREQ(2,3)

class affinity_set
{
public:

    affinity_set();

    affinity_set(int cpu_nr);

    void add(int cpu_nr);

    int get_affinity();

    int set_affinity();

    void print_affinity(int id, const char* s);

    void print_set(int id, const char* s);

private:

    cpu_set_t set;
};

#else
#error "Affinity requires Linux glibc version >= 2.3.3, which isn't available. Please add -DMAGMA_NOAFFINITY to the CFLAGS in make.inc."
#endif

#endif  // MAGMA_NOAFFINITY

#endif  // MAGMA_AFFINITY_H
