/*
 *    Llog.h
 *    Enable the visual tracing of threads and GPUs.
 *	  @auteur: Simplice D.
 *    @Last modif:30 oct 2013
 */
#include <stdio.h>
#include <stdlib.h>
#include "Llog.h"
#include "time.h"
#ifdef _WIN32    
    #include "sched.h"
    #include "pthread.h"
    #pragma comment(lib, "pthreadVC2.lib")
#else
    #include "pthread.h"    
//    #define PTHREAD_MUTEX_INITIALIZER ((pthread_mutex_t) -1)
//    #define PTHREAD_COND_INITIALIZER ((pthread_cond_t) -1)
#endif



//pthread_mutex_t mutex_llogs = PTHREAD_MUTEX_INITIALIZER;

/**/
Llog* Llog_new()
{
    Llog *p;
    p =(Llog *) malloc(sizeof(Llog));
    p->next  = NULL;
    return p;
}


void Llog_add_ptr_tail(Llog **list,Llog *p)
{
    Llog *z;
//pthread_mutex_lock (&mutex_llogs);
/*    if (*list==NULL) *list = p;
    else
    {
        z = *list;
        while (z->next!=NULL) z = z->next;
        z->next = p;

    }
*/
    if(*list == NULL)*list=p;
    else
    {
        z = (*list)->last;
        z->next = p;
    }

    (*list)->last = p;

//pthread_mutex_unlock (&mutex_llogs);
}

/*
 * Inserer un element a la queue;
 */
void Llog_add_tail(Llog **list, int tid, char type, long tStart, long tEnd, int step, int col, int row, int stolen)
{
    Llog *p = Llog_new();
    p->tid =tid;
    p->type = type;
    p->step = step;
    p->col = col;
    p->row = row;
    p->stolen = stolen;
    p->tStart = tStart;
    p->tEnd = tEnd;
    
   Llog_add_ptr_tail(list,p);
}

void Llog_print(Llog *list)
{
//pthread_mutex_lock (&mutex_llogs);
    printf("La liste(%d): ",Llog_count(list));
        while(list!=NULL) 
        {

            printf("[%d] Type:%c, Start:%ld, end:%ld ",list->tid, list->type, list->tStart, list->tEnd);
            
            list=list->next ;
        }
        printf("\n");
//pthread_mutex_unlock (&mutex_llogs);
}

/*Dessiner la liste dans un fichier style html*/
void Llog_fplot(Llog **llogs, int nbthreads, long beginTime, long endTime, long scale, char *desc)
{
    char fname[30];
    char buffer[512];
    long left, width;
    int height = 20;
    int top;
    FILE *fp;
    Llog *list;
    char color[30];

    double X1,X2;

    long totalTime = endTime - beginTime;
    int i;

    sprintf(fname,"Llog.htm");

//pthread_mutex_lock (&mutex_llogs);
fp = fopen(fname,"w+");
/*write html header*/
fprintf(fp,"<html>\n");
fprintf(fp,"<body>\n");
if(fp!=NULL)
{
    for(i=0;i<nbthreads;i++)
    {
        list = llogs[i];

        printf("Liste[%d]: %d\n",i,Llog_count(list));
        
        while(list!=NULL) 
        {
            switch(list->type )
            {
            case 'P':
                sprintf(color,"%s","red");
                break;
            case 'W':
                sprintf(color,"%s","gray");
                break;
            case 'L':
                sprintf(color,"%s","yellow");
                break;
            case 'U':
                if(list->stolen) sprintf(color,"%s","#FFCCFF");
                else sprintf(color,"%s","violet");
            break;
            case 'S':
                if(list->stolen) sprintf(color,"%s","#00FF00");
                else sprintf(color,"%s","green");
            break;
            case 'O':
                sprintf(color,"%s","orange");
                break;
            case 'C':
                sprintf(color,"%s","black");
                break;
            case 'T':
                sprintf(color,"%s","blue");
                break;            
            case 'G':
                sprintf(color,"%s","indigo");
                break;
            default:
                sprintf(color,"%s","white");
            }

            X1= (double) ((list->tStart - beginTime) / (double) totalTime);

            X2    = (double) ((list->tEnd - beginTime) / (double) totalTime);

            left = (long) (X1 * scale);

            width = (long) ((X2 * scale) - left);

            top = (list->tid*height);

            //width = (long) x - left; //1000*(list->tEnd - list->tStart)/totalTime; 
            
            sprintf(buffer,"<div tag=\"%c\" style=\"position:absolute;border:solid 1px black; left:%ldpx; top:%dpx;height:%dpx; width:%ldpx; background-color:%s\" >&nbsp;</div>", list->type, left,top,height-1,width-1,color);
            //sprintf(buffer,"<div tag=\"%c\" style=\"position:absolute;border:solid 1px black; left:%ldpx; top:%dpx;height:%dpx; width:%ldpx; background-color:%s\" >%d:%d:%d</div>", list->type, left,top,height-1,width-1,color, list->step, list->col, list->row);

            fprintf(fp,"%s\n",buffer);
            //printf("[%d] Type:%c, Start:%ld, end:%ld ",list->tid, list->type, list->tStart, list->tEnd);
            
            list=list->next ;
        }
    }
        printf("\n");

        /*write Legend*/
        fprintf(fp,"<div style=\"position:absolute;top:%dpx;\">",height*(nbthreads+4));
        fprintf(fp,"<table cellpadding=10 cellspacing=0>\n");
        fprintf(fp,"<caption>%s (step:col:row)</caption>\n",desc);
        fprintf(fp,"<tr><td bgcolor=red> </td><td>dgetrf</td></tr>\n");
        fprintf(fp,"<tr><td bgcolor=yellow> </td><td>dtrsm L</td></tr>\n");
        fprintf(fp,"<tr><td bgcolor=violet> </td><td>dtrsm U (+ dlaswap)</td></tr>\n");
        fprintf(fp,"<tr><td bgcolor=green> </td><td>dgemm</td></tr>\n");
        fprintf(fp,"<tr><td bgcolor=gray> </td><td>swap</td></tr>\n");
        fprintf(fp,"<tr><td bgcolor=#00FF00> </td><td>dgemm (stolen)</td></tr>\n");
        fprintf(fp,"<tr><td bgcolor=#FFCCFF> </td><td>dtrsm U (stolen)</td></tr>\n");
        fprintf(fp,"<tr><td bgcolor=blue> </td><td>setMatrix</td></tr>\n");
        fprintf(fp,"<tr><td bgcolor=indigo> </td><td>getMatrix</td></tr>\n");
        fprintf(fp,"<tr><td bgcolor=black> </td><td>device_sync</td></tr>\n");
        fprintf(fp,"</table>\n");
        fprintf(fp,"</div>");

        /*write html footer*/
        fprintf(fp,"</body>\n");
        fprintf(fp,"</html>\n");

        fclose(fp);
}
else
{
    printf("Can not open file: %s\n",fname);
}
//pthread_mutex_unlock (&mutex_llogs);

}

int Llog_count(Llog *list)
{
    int count = 0;

        while(list!=NULL)
        {
            list = list->next;
            count++;
        }
        return count ;    
}

void Llog_clear(Llog **list){

    Llog *p;

    while(*list!=NULL)
    {
        
        p = *list;
        
        //str_free(p->cdata); 

        *list = (*list)->next;

        Llog_free(&p);
    }
}

/*
 * Supprimer l'espace alloué par une cellule
 */
void Llog_free(Llog **l)
{
    Llog *p;

    if((*l)!=NULL)
    {
        p= *l;
        free(p);
        (*l) = (*l)->next; 
    }
}

