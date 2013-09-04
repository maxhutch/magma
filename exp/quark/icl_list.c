/**
 * @file
 *
 * Dependency free doubly linked list implementation.
 *
 */
/* $Id: icl_list.c 1693 2010-10-26 18:29:56Z yarkhan $ */
/* $UTK_Copyright: $ */

#include <stdio.h>
#include <stdlib.h>

#include "icl_list.h"

/**
 * Create new linked list.
 *
 * @returns pointer to new linked list.  returns NULL on error.
 */

icl_list_t *
icl_list_new()
{
  icl_list_t *node;

  node = (icl_list_t*)malloc(sizeof(icl_list_t));

  if(!node) return NULL;

  node->flink = NULL;
  node->blink = node;
  node->data = NULL;

  return(node);
}

/**
 * Insert a new node after the specified node.
 *
 * @param head -- the linked list
 * @param pos -- points to the position of the new node (it will
 *   be inserted after this node)
 * @param data -- pointer to the data that is to be inserted
 *
 * @returns pointer to the new node.  returns NULL on error.
 */

icl_list_t *
icl_list_insert(icl_list_t *head, icl_list_t *pos, void *data)
{
  icl_list_t *node;

  if(!head || !pos) return NULL;

  node = (icl_list_t*)malloc(sizeof(icl_list_t));

  if(!node) return NULL;

  node->blink = pos;
  node->flink = pos->flink;
  node->data = data;

  if(pos->flink)
    pos->flink->blink = node;
  else
    head->blink = node; /* node at end of list */

  pos->flink = node;

  return node;
}

/**
 * Delete the specified node.
 *
 * @param head -- the linked list containing the node to be deleted
 * @param pos -- the node to be deleted
 * @param free_function -- pointer to function that frees the node data
 *
 * @returns 0 on success, -1 on failure.
 */

int
icl_list_delete(icl_list_t *head, icl_list_t *pos, void (*free_function)(void *))
{
  if (!pos || !head) return -1;
  if (pos == head) return -1;

  if(free_function && pos->data)
    (*free_function)(pos->data);

  pos->blink->flink = pos->flink;

  if(pos->flink)
    pos->flink->blink = pos->blink;
  else
    head->blink = pos->blink; /* pos at end of list */

  free(pos);

  return 0;
}

/**
 * Finds a data item in the specified linked list.
 *
 * @param head -- the linked list
 * @param data -- the data to be found
 * @param compare -- function that compares the data items
 *
 * @returns pointer to the node, if found.  otherwise returns NULL.
 */

icl_list_t *
icl_list_search(icl_list_t *head, void *data, int (*compare)(void*, void*))
{
  icl_list_t *node;

  if (!head) return NULL;

  for (node=head->flink; node!=NULL; node=node->flink) {
    if(!node->data)
      continue;
    if((compare && (*compare)(node->data, data)==0))
      break;
    else if (node->data==data)
      break; /* compare == NULL, then direct comparison of pointers */
  }

  return(node);
}

/**
 * Frees the resources associated with this linked list.
 *
 * @param head -- the linked list to be destroyed
 * @param free_function -- pointer to function that frees the node's data
 *
 * @returns 0 on success, -1 on failure.
 */

int
icl_list_destroy(icl_list_t *head, void (*free_function)(void*))
{
  icl_list_t *node, *tmpnode;

  if (!head) return -1;

  for(node=head; node!=NULL; ) {
    tmpnode = node->flink;

    if(free_function!=NULL && node->data!=NULL)
      (*free_function)(node->data);

    free(node);
    node = tmpnode;
  }

  return 0;
}

/**
 * Get the number of items in this linked list.
 *
 * @param head -- the linked list
 *
 * @returns the number of items in the list.  returns -1 on error.
 */

int
icl_list_size(icl_list_t *head)
{
  int size=0;

  if(!head) return -1;

  while((head=head->flink))
    size++;

  return size;
}

/**
 * Get the first item in this linked list.
 *
 * @param head -- the linked list
 *
 * @returns pointer to the first item.  returns NULL on error.
 */

icl_list_t *
icl_list_first(icl_list_t *head)
{
  if(head)
    return head->flink;

  return NULL;
}

/**
 * Get the last item in this linked list.
 *
 * @param head -- the linked list
 *
 * @returns pointer to the last item.  returns NULL on error.
 */

icl_list_t *
icl_list_last(icl_list_t *head)
{
    icl_list_t *pos;
    for ( pos=head; pos->flink!=NULL; pos=pos->flink ) {}
    if ( pos->blink==pos ) return NULL;
    else return pos;
}


/**
 * Get the node following the specified node.
 *
 * @param head -- the list containing the specified node
 * @param pos -- the node whose successor should be returned
 *
 * @returns pointer to the next node.  returns NULL on error.
 */

icl_list_t *
icl_list_next(icl_list_t *head, icl_list_t *pos)
{
  if(pos)
    return pos->flink;

  return NULL;
}

/**
 * Get the node preceding the specified node.
 *
 * @param head -- the list containing the specified node
 * @param pos -- the node whose predecessor should be returned
 *
 * @returns pointer to the previous node.  returns NULL on error.
 */

icl_list_t *
icl_list_prev(icl_list_t *head, icl_list_t *pos)
{
  if(pos && pos->blink!=NULL && pos!=head && pos->blink!=head && pos->blink!=pos )
    return pos->blink;

  return NULL;
}

/**
 * Concatenate two linked lists.
 *
 * @param head1 -- the first linked list
 * @param head2 -- the second linked list
 *
 * @returns pointer to the new linked list, which consists of
 *   <head1,head2>.  returns NULL on error.
 */

icl_list_t *
icl_list_concat(icl_list_t *head1, icl_list_t *head2)
{
  if(!head1 || !head2 || !head1->blink || !head2->flink)
    return NULL;

  head1->blink->flink = head2->flink;
  head2->flink->blink = head1->blink;
  head1->blink = head2->blink;

  free(head2);

  return(head1);
}

/**
 * Insert a node at the beginning of this list.
 *
 * @param head -- the linked list
 * @param data -- the data to be inserted
 *
 * @returns pointer to the new node.  returns NULL on error.
 */

icl_list_t *
icl_list_prepend(icl_list_t *head, void *data)
{
  return(icl_list_insert(head, head, data));
}

/**
 * Insert a node at the end of this list.
 *
 * @param head -- the linked list
 * @param data -- the data to be inserted
 *
 * @returns pointer to the new node.  returns NULL on error.
 */

icl_list_t *
icl_list_append(icl_list_t *head, void *data)
{
  return(icl_list_insert(head, head->blink, data));
}
