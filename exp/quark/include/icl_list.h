/**
 * @file
 *
 * Header file for the icl_list routines.
 *
 */
/* $Id: icl_list.h 1532 2010-09-07 15:38:18Z yarkhan $ */
/* $UTK_Copyright: $ */

#ifndef ICL_LIST_H
#define ICL_LIST_H

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

struct icl_list_s {
  void *data;
  struct icl_list_s *flink;
  struct icl_list_s *blink;
};

typedef struct icl_list_s icl_list_t;

icl_list_t
  * icl_list_new(),
  * icl_list_insert(icl_list_t *, icl_list_t *, void *),
  * icl_list_search(icl_list_t *, void *, int (*)(void*, void*)),
  * icl_list_first(icl_list_t *),
  * icl_list_last(icl_list_t *),
  * icl_list_next(icl_list_t *, icl_list_t *),
  * icl_list_prev(icl_list_t *, icl_list_t *),
  * icl_list_concat(icl_list_t *, icl_list_t *),
  * icl_list_prepend(icl_list_t *, void *),
  * icl_list_append(icl_list_t *, void *);

int
  icl_list_delete(icl_list_t *, icl_list_t *, void (*)(void *)) ,
  icl_list_destroy(icl_list_t *, void (*)(void*)),
  icl_list_size(icl_list_t *);

#define icl_list_foreach( list, ptr) \
    for (ptr = icl_list_first(list); ptr != NULL; ptr = icl_list_next(list, ptr)) 

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif


#endif /* ICL_LIST_H */
