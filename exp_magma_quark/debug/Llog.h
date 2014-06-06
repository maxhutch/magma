/*
 *    Llog.h
 *    Enable the visual tracing of threads and GPUs.
 *	  @auteur: Simplice D.
 *    @Last modif:30 oct 2013
 */
#ifndef LLOG_H
#define LLOG_H




typedef struct Llog_cell Llog;

struct Llog_cell
{

    char type; /*Type de la tache, une couleur par type*/
    int step; /*Etape de la tache*/
    int col;
    int row;
    
    int stolen; /*tell if the task was stolen*/

    long tStart; /* Temps de debut de la tache*/
    int tid; /*Numero du processus 0..NBTHREADS*/
    long tEnd; /*Temps de fin de la tache*/
    Llog *next;
    Llog *last; /*Dernier element de la liste*/
};

/* typedef struct Llog_cell LlogCell;*/

/*
* Imprimer la liste;
*/
void Llog_print(Llog *list);

/*Dessiner la liste dans un fichier style html*/
void Llog_fplot(Llog **llogs, int nbthreads, long beginTime, long endTime, long scale, char *desc);

/*
* Creer un element de liste
*/

Llog* Llog_new();


/*
 * Inserer un element de type liste a la queue;
 */
void Llog_add_ptr_tail(Llog **list, Llog *p);

/*
* Ajouter un element a la queue de la liste
*/
void Llog_add_tail(Llog **list, int tid, char type, long tStart, long tEnd, int step, int col, int row, int stolen);

/*
 * Compter le nombre d'element d'une liste
 */

int Llog_count(Llog *list);

/* 
* Supprimer tous les elements d'une liste et liberer l'espace utilise 
*/
void Llog_clear(Llog **list);

/*
 * Supprimer l'espace alloue par une Liste
 */
void Llog_free(Llog **l);

#endif


