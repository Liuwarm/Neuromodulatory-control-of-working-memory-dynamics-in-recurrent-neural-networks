#define _CRT_SECURE_NO_WARNINGS  // Add this line to disable the security warning

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <signal.h>

#include "randdev_cyclic_alpha_tau_D.h"

#define ROUND(x) ((int)(x+0.5))
#define STIM_DELAY_E 0.0
#define STIM_DELAY_I 0.0

/* ------------------------------------------------------------------------ */
/* ------------------------------- Parameters ----------------------------- */
/* ------------------------------------------------------------------------ */

/* ------------------------------------------------------------------------ */
/* network parameters */

unsigned int Ne=8000, Ni=2000; // number of E and I neurons
float muEext=23.10, sEext=1.00; // external inputs to E neuron - mean [mV]; variance [mV^2]
//float sEext=1.00; // external inputs to E neuron - mean [mV]; variance [mV^2]
float muIext=21.00, sIext=1.00; // external inputs to E neuron - mean [mV]; variance [mV^2]
float pconn=0.20; // probability of connections
float Cp=.10; // fraction of potentiated synapses among non-selective neurons
float Jdep=0.10, Jpot=0.45; // depressed and potentiated E->E efficacies
float Jie=-.25f, Jei=.135, Jii=-.20f; //remaining efficacies Jab = b->a

float f=.10f; // coding level - prob an E neuron belongs to a memory
int pops=5; // number of memories
float qp=1.0; // probability that a synapse between to selective neurons is potentiated
float qm=0.1; // probability that a synapses from a selective to a non-selective neuron is depressed
/* ------------------------------------------------------------------------ */

/* ------------------------------------------------------------------------ */
/* neuronal parameters */

float Taue=15, Taui=10; // membrane time constants [ms]	(E & I)
float Tarpge=2, Tarpgi=2; // absolute refractory periods [ms] (E & I)
float ThrE=20, ThrI=20; // spike-emission threshold [mV] (E & I)
float He=.8, Hi=.65; // post-spike reset voltage Vr=H*Thr (E & I)

/* ------------------------------------------------------------------------ */
/* synaptic parameters */

float U=0.2; // baseline probability of release
//float tau_D=200.0; // time constant for depression [ms]
float tau_F=1500.0; // time constant for facilitation [ms]
float TauNMDA=100.0; // time constant for slow E currents [ms]
float xNMDA_EE=0.0; // fraction of slow E->E currents - 0.0 = no slow currents
float xNMDA_IE=0.0; // fraction of slow E->I currents - 0.0 = no slow currents

/* ------------------------------------------------------------------------ */
/* simulation parameters */

float T_SA=1000; // duration of the spontaneous activity trial [ms]
int num_pres=1; // number of trials - stimuli are presented cyclically, i.e., 1->2->..->pops->1->2..
float Tprestim_PT=100000; // pre-stimulus interval [ms]
//float Tcue_PT=350; // cue presentation interval [ms]
//float Tdelay_PT=8600; // delay interval [ms]
float Tcue_PT=0; // cue presentation interval [ms]
float Tdelay_PT=0; // delay interval [ms]
//float contrastE_sel=1.15f; // constrast factor during cue presentation
float contrastE_sel=1.0f; // Remove selective stimulation

/* ------------------------------------------------------------------------ */
/* useful flags */
int flagstruct=1; // 1 - make synaptic structuring; 0 - NO synaptic structuring
int flag_fac=1;  // 1 - facilitation ON; 0 - facilitation OFF, i.e., only depression
int flag_fs=1; // 1 - no finite-size effects; 0 - finite size effetcs
/* no finite-size effects means that all neurons receive the same number of connections */
/* and all memories contain the same number of neurons (i.e., exactly fNe). Note that this */
/* is possible only when f*pops<=1. */

int flag_SA, flag_popout; // internally used - DO NOT SET

/* ------------------------------------------------------------------------ */
/* miscellaneous parameters */

float contrastE_nonsel=1.0f;
float contrastI_sel=1.0, contrastI_nonsel=1.00;
float contrastE_sel_noise=1.0f, contrastE_nonsel_noise=1.0f;
float contrastI_sel_noise=1.0, contrastI_nonsel_noise=1.00;

float T_popout=2000, DT_popout=700, g_popout=1.00;

float f_noise=.15f, x=0.00f;
float width_stim=0.25;
float width_popout=0.00;

//Initialization Parameters
float nue=.001, nui=.005;
float mu_e_mf, s2_e_mf, mu_i_mf, s2_i_mf;
float avg_x, avg_x2, avg_u;
float muV_e, muV_i, s2V_e, s2V_i;

float Tprestim, Tcue, Tdelay, Tonset, Toffset;

float dt = 0.1f;     /* step minimo in ms                      */
float TCamp = 10.0;      /* Tempo[ms] di campionamento attivita' neurale*/
float MinDelay  = 0.1f; /* Ritardo sinaptico minimo in ms (per eccitatori) */
float StepDelay = 0.1f; /* Larghezza step delay  */
int MaxStep = 10.0;

//float erpha = 2.105;

float erpha,tau_D;

/* ------------------------------------------------------------------------*/
/* ---------------------- INIZIO DEFINIZIONE DATI ------------------------ */
/* ------------------------------------------------------------------------*/

unsigned int Ntot;            /* Neuroni totali */

float *Inmda, *Iext;

char *gg, **P, **D, **sel_old;

int **nofs_conn, **nofs_conn_inv, *nconn;
int *delay_stim, *duration_stim, *duration_popout;
int *spikes, spikesI;
float *curr, currI;

float theta;

typedef struct _neuron Neuron;
typedef struct _population Population;
typedef struct _efficacy Efficacy;
typedef struct _arrayroot EffRoot;

struct _neuron
{
    float V;
    int status;
    int tspike;
    float Hr;
    float xtm;
    float utm;
    float uxtm;
};

struct _population
{
    unsigned int num;
    int *index;
};

struct _efficacy
{
    int dist; /* distanza del prossimo neurone */
    float efficacy; /* efficacia sinaptica */
    char delay;
};

struct _arrayroot
{
    unsigned int num;
    Efficacy *eff;
};

Population *Pop;
Neuron *neur;
EffRoot *J;
float **Events;
float **EventsNMDA;

FILE *curr_out, *ratepops, *spikeout, *synvar_u, *synvar_x;

unsigned int Time;

/*************************************************************************/

void shifttspike()
{
    int life;
    int i;
    life= (int)ROUND((Tprestim+Tcue+Tdelay)/dt);
    fprintf(stdout,">>>>>>\n");
    for (i=0; i<Ntot; i++)
    {
        neur[i].tspike-=life;
        fprintf(stdout,"%d\t%d\n",i,neur[i].tspike);
    }
}

void STP(int i, int t1)
{
    int t0;
    float x0, x1, delt, u0, u1;
    x0=neur[i].xtm;
    u0=neur[i].utm;
    t0=neur[i].tspike;
    delt=(t1-t0)*dt;

    if (flag_fac==0)
        u0=U;
    x1=1.0-(1.0-(1.0-u0)*x0)*exp(-delt/tau_D);
    u0=erpha*U+u0*(1-erpha*U);
    u1=erpha*U-(erpha*U-u0)*exp(-delt/tau_F);
    neur[i].uxtm=u1*x1;
    neur[i].xtm=x1;
    if (flag_fac==0)
        u1=U;
    neur[i].utm=u1;
}

float readout(int i,int t1,char flag_ux)
{
    int t0;
    float x0,x1,delt, u0, u1;
    float ris;
    x0=neur[i].xtm;
    u0=neur[i].utm;
    t0=neur[i].tspike;
    delt=(t1-t0)*dt;
    if (flag_fac==0)
        u0=U;
    if (flag_ux==0)
        ris=1.0-(1.0-x0)*exp(-delt/tau_D);
    else
        ris=erpha*U-(erpha*U-u0)*exp(-delt/tau_F);
    return(ris);
}

/** Populations' routines - BEGIN **/

void makePop(char **sel)
{
    int i, p, count;

    Pop=(Population*)malloc(pops*sizeof(Population));
    for (p=0; p<pops; p++)
    {
        count=0;
        for (i=0; i<Ne; i++)
            count+=sel[i][p];
        Pop[p].num=count;
        Pop[p].index=(int*)malloc(count*sizeof(int));

        count=0;
        for (i=0; i<Ne; i++)
        {
            if (sel[i][p]==1)
            {
                Pop[p].index[count]=i;
                count++;
            }
        }
    }
}

char** createPop(char flag_fix)
{
    char **sel;
    int i,p, mul, count, Nfix_E=ROUND(f*Ne), Nfix_I=ROUND(f*Ni);

    // Allocation and Initialization of matrix selectivity

    sel=(char **)malloc((Ne+Ni)*sizeof(char *));
    for (i=0; i<(Ne+Ni); i++)
    {
        sel[i]=(char *)malloc((pops+1)*sizeof(char));
        for (p=0; p<=pops; p++)
            sel[i][p]=0;
    }

    // Generation of Population

    for (p=0; p<pops; p++)
    {
        if (flag_fix==0)
        {
            for (i=0; i<(Ne+Ni); i++)
                if (Random()<f)
                    sel[i][p]=1;
        }
        else
        {
            count=0;
            while (count<Nfix_E)
            {
                i=((int)(Random()*Ne));
                if (sel[i][p]==0)
                {
                    sel[i][p]=1;
                    count++;
                }
            }
            count=0;
            while (count<Nfix_I)
            {
                i=Ne+((int)(Random()*Ni));
                if (sel[i][p]==0)
                {
                    sel[i][p]=1;
                    count++;
                }
            }
        }
    }

    // Generation of Multiplicity

    for (i=0; i<(Ne+Ni); i++)
    {
        mul=0;
        for (p=0; p<pops; p++)
            mul+=sel[i][p];
        sel[i][pops]=mul;
    }

    return(sel);
}

char** createPop_noov()
{
    char **sel;
    int i,p, mul, count, Nfix_E=ROUND(f*Ne), Nfix_I=ROUND(f*Ni);
    int N1, N2;
    // Allocation and Initialization of matrix selectivity

    sel=(char **)malloc((Ne+Ni)*sizeof(char *));
    for (i=0; i<(Ne+Ni); i++)
    {
        sel[i]=(char *)malloc((pops+1)*sizeof(char));
        for (p=0; p<=pops; p++)
            sel[i][p]=0;
    }

    // Generation of Population
    for (p=0; p<pops; p++)
    {
        N1=p*Nfix_E; N2=(p+1)*Nfix_E;
        for (i=N1; i<N2; i++)
            sel[i][p]=1;

        N1=p*Nfix_I; N2=(p+1)*Nfix_I;
        for (i=Ne+N1; i<Ne+N2; i++)
            sel[i][p]=1;
    }

    // Generation of Multiplicity

    for (i=0; i<(Ne+Ni); i++)
    {
        mul=0;
        for (p=0; p<pops; p++)
            mul+=sel[i][p];
        sel[i][pops]=mul;
    }

    return(sel);
}

void computePD(char **sel)
{
    int p, i, j;
    int running_P, running_D;

    printf("> computePD()\n");

    P=(char **)malloc(Ne*sizeof(char *));
    D=(char **)malloc(Ne*sizeof(char *));

    for (i=0; i<Ne; i++)
    {
        P[i]=(char *)malloc(Ne*sizeof(char));
        D[i]=(char *)malloc(Ne*sizeof(char));
    }

    for (i=0; i<Ne; i++)
    {
        for (j=0; j<Ne; j++)
        {
            running_P=running_D=0;
            for (p=0; p<pops; p++)
            {
                running_P+=((sel[i][p]==1)&&(sel[j][p]==1));
                running_D+=((sel[i][p]==1)&&(sel[j][p]==0));
            }
            P[i][j]=running_P;
            D[i][j]=running_D;
        }
    }

    printf("> end computePD()\n");
}

/** Populations' routines - END **/

/** Synapses' routines - BEGIN **/

void initSynapses_nofs()
{
    char theend, theend_mem;
    char *excit, *inhib;
    int mem, ne, ni, Nmem, Nrest;
    int C_mem, C_rest, C_totE, C_totI;
    int post, pre, count_mem, count_others, count_I, count;
    int k, run_pre;

    //nofs_conn[N][cN] - to be declared outside

    nofs_conn=(int **)calloc((Ne+Ni),sizeof(int *));
    for (k=0; k<(Ne+Ni); k++)
        nofs_conn[k]=(int *)calloc(ROUND(pconn*(Ne+Ni)),sizeof(int));

    //excitatory cells
    Nmem=ROUND(f*Ne); Nrest=Ne-(pops*Nmem);
    C_totE=ROUND(pconn*Ne);  C_totI=ROUND(pconn*Ni);
    C_mem=ROUND(pconn*f*Ne);  C_rest=C_totE-C_mem;

    excit=(char *)calloc(Ne,sizeof(char));
    inhib=(char *)calloc(Ni,sizeof(char));

    for (post=0; post<(Ne+Ni); post++)
    {
        for (k=0; k<Ne; k++)
            excit[k]=0;
        for (k=0; k<Ni; k++)
            inhib[k]=0;

        theend=0; count=0; mem=0;

        while(theend==0)
        {
            theend_mem=0; count_mem=0;
            while (theend_mem==0)
            {
                pre=((int)(Random()*Nmem));
                run_pre=mem*Nmem+pre;

                if (excit[run_pre]==0)
                {
                    excit[run_pre]=1;
                    nofs_conn[post][count]=run_pre;
                    count++; count_mem++;

                    if (count_mem==C_mem)
                    {
                        theend_mem=1;
                        mem++;
                    }
                }

                if (mem==pops)
                    theend=1;
            }
        }

        theend=0;
        while(theend==0)
        {
            pre=((int)(Random()*Nrest));
            run_pre=mem*Nmem+pre;
            if (excit[run_pre]==0)
            {
                excit[run_pre]=1;
                nofs_conn[post][count]=run_pre;
                count++;
                if (count==(C_mem+C_rest))
                    theend=1;
            }
        }

        //select inhibitory
        theend=0;
        while(theend==0)
        {
            pre=((int)(Random()*Ni));
            run_pre=Ne+pre;
            if (inhib[pre]==0)
            {
                inhib[pre]=1;
                nofs_conn[post][count]=run_pre;
                count++;
                if (count==(C_mem+C_rest+C_totI))
                    theend=1;
            }
        }
    }
    free(excit); free(inhib);
}

void invert_nofs_udo()
{
    int post, k, max_conn, run_i;
    int C_tot=ROUND(pconn*(Ne+Ni));
    int *count;

    nconn=(int *)calloc((Ne+Ni),sizeof(int));

    for (post=0; post<(Ne+Ni); post++)
        for (k=0; k<C_tot; k++)
            nconn[nofs_conn[post][k]]++;

    max_conn=0;
    for (k=0; k<(Ne+Ni); k++)
        if (max_conn<nconn[k])
            max_conn=nconn[k];

    count=(int *)calloc((Ne+Ni),sizeof(int));

    nofs_conn_inv=(int **)calloc((Ne+Ni),sizeof(int*));
    for (k=0; k<(Ne+Ni); k++)
        nofs_conn_inv[k]=(int *)calloc(nconn[k],sizeof(int));

    for (post=0; post<(Ne+Ni); post++)
    {
        for (k=0; k<C_tot; k++)
        {
            run_i=nofs_conn[post][k];
            nofs_conn_inv[run_i][count[run_i]]=post;
            count[run_i]++;
        }
    }

    free(count);

    for (post=0; post<(Ne+Ni); post++)
        free(nofs_conn[post]);
    free(nofs_conn);
}

void InitSynapses()
{
    int i1, i2, i3, kk, count, oldi2, dummy;
    int* numarray=NULL;
    float ppot;
    float Jr;
    FILE *fpo_chk;

    Ntot=Ne+Ni;
    printf("Drawing synaptic matrix: please, wait...\n");
    J=(EffRoot*)malloc(Ntot*sizeof(EffRoot));

    if (flagstruct==0)
        printf(">> no structuring!!\n");
    else
        computePD(sel_old);

    fpo_chk=fopen("chk_syndist.log","w");

    for(i1=0; i1<Ntot; i1++)
    {
        if(!numarray)
            numarray=(int*)malloc(sizeof(int)*((int)(5*pconn*Ntot)));

        if (flag_fs==0)
        {
            count=oldi2=0;
            for(i2=0; i2<Ntot; i2++)
            {
                if(Random()<pconn)
                {
                    numarray[count]=i2-oldi2 ;
                    if( ((count+1)>(5*pconn*Ntot-1)))
                        break;
                    count++;
                    oldi2=i2;
                }
            }
        }
        else
        {
            numarray[0]=nofs_conn_inv[i1][0];
            oldi2=0;
            for (i2=0; i2<nconn[i1]; i2++)
            {
                numarray[i2]=nofs_conn_inv[i1][i2]-oldi2;
                oldi2=nofs_conn_inv[i1][i2];
            }
            count=nconn[i1];
        }

        fprintf(fpo_chk,"%d\n",count);

        J[i1].num=count;
        J[i1].eff=(Efficacy*)malloc(sizeof(Efficacy)*count);
        i3=0;
        for(i2=0; i2<count; i2++)
        {
            J[i1].eff[i2].dist=numarray[i2];
            i3+=numarray[i2];
            if (i1<Ne)
                (i3<Ne) ? (Jr=Jdep) : (Jr=Jei);
            else
                (i3<Ne) ? (Jr=Jie) : (Jr=Jii);
            J[i1].eff[i2].efficacy=Jr;
            if ((i1<Ne)&&(i3<Ne))
            {
                if (flagstruct==1)
                {
                    if ((P[i1][i3]==0)&&(D[i1][i3]==0))
                        ppot=Cp;
                    else
                        ppot=qp*P[i1][i3]/(qp*P[i1][i3]+qm*D[i1][i3]);
                }
                else
                    ppot=Cp;
                if (Random()<ppot){
                    J[i1].eff[i2].efficacy=Jpot;
                }
            }

            if (i1>Ne)
                J[i1].eff[i2].delay=(char)(0);
            else
            {
                kk=(int)(Random()*MaxStep);
                J[i1].eff[i2].delay=(char)(kk);
            }
        }
        free(numarray);
        numarray=NULL;
    }

    fclose(fpo_chk);

    for (i1=0; i1<Ne; i1++)
    {
        free(P[i1]);
        free(D[i1]);
    }
    free(P); free(D);

    if (flag_fs==1)
    {
        for (i1=0; i1<(Ne+Ni); i1++)
            free(nofs_conn_inv[i1]);
        free(nofs_conn_inv);
    }
}

/** Synapses' routines - END **/

/** Initialization routines - BEGIN **/

void InitEventMatrix()
{
    int i, t, tbound, deltamin;
    float run_mu;

    tbound=(int)((MinDelay + MaxStep*StepDelay)/dt+0.5)+1;
    deltamin=(int)((MinDelay)/dt+0.5)+1;
    Events=(float **)malloc(tbound*sizeof(*Events));
    EventsNMDA=(float **)malloc(tbound*sizeof(*EventsNMDA));
    for(t=0; t<tbound; t++)
    {
        Events[t]=(float *)malloc((Ne+Ni)*sizeof(**Events));
        EventsNMDA[t]=(float *)malloc((Ne+Ni)*sizeof(**EventsNMDA));
        for(i=0; i<(Ne+Ni); i++)
        {
            Events[t][i]=0.0;
            EventsNMDA[t][i]=0.0;
        }
    }
}

void InitV()
{
    int count;
    int tauarp;
    float run_V, w;
    float gu=1.0;

    Ntot=Ne+Ni;

    neur=(Neuron *)malloc(Ntot*sizeof(Neuron));

    gg=(char *)malloc(Ntot*sizeof(char));
    Inmda=(float*)malloc(Ntot*sizeof(float));
    Iext=(float*)malloc(Ntot*sizeof(float));
    delay_stim = (int*) malloc(Ntot*sizeof(int));
    duration_stim = (int*) malloc(Ntot*sizeof(int));
    duration_popout = (int*) malloc(Ntot*sizeof(int));
    spikes=(int*) malloc( Ne * sizeof(int));
    curr=(float*) malloc( Ne * sizeof(float));

    for(count=0; count<Ntot; count++)
    {
        if (count<Ne)
        {
            //neur[count].Hr=.5+.4*Random();
            neur[count].Hr=He;
            if (Random()<nue*Tarpge)
            {
                tauarp=ROUND(Tarpge/dt);
                neur[count].status=-(int)(tauarp*Random());
                neur[count].V=0;
                neur[count].tspike=0.0;
                neur[count].xtm=0.1;
                neur[count].utm=U*gu;
            }
            else
            {
                run_V=muV_e+sqrt(s2V_e)*NormDev();
                if (run_V>=ThrE)
                    neur[count].V=muV_e;
                else
                    neur[count].V=run_V;
                neur[count].status=0;
                neur[count].xtm=0.1;
                neur[count].utm=U*gu;
                w=1200.0;
                neur[count].tspike=-((int)(w/dt));
            }
        }
        else
        {
            neur[count].Hr=Hi;
            if (Random()<nui*Tarpgi)
            {
                tauarp=ROUND(Tarpgi/dt);
                neur[count].status=-(int)(tauarp*Random());
                neur[count].V=0;
                neur[count].tspike=0.0;
            }
            else
            {
                run_V=muV_i+sqrt(s2V_i)*NormDev();
                if (run_V>=ThrI)
                    neur[count].V=muV_i;
                else
                    neur[count].V=run_V;
                neur[count].status=0;
            }

            neur[count].xtm=0;
            neur[count].utm=0;
            neur[count].tspike=0.0;
        }

        Inmda[count]=0.0;
        Iext[count]=0.0;
        delay_stim[count]=0;
        gg[count]=0;

        if (count<Ne)
            spikes[count]=0;
    }
}

/** Initialization routines - END **/
void prepare_popout()
{
    int i;
    for (i=0; i<Ntot; i++)
        duration_popout[i]=ROUND((DT_popout*(1+width_popout*(.5-Random())))/dt);
}

void select_VR(int cue)
{
    int i;
    for (i=0; i<Ntot; i++)
    {
        gg[i]=0;
        delay_stim[i]=0;
        duration_stim[i]=0;
    }

    if (cue<pops)
    {
        //select one of the stored items

        /* select excitatory visual responsive */

        for (i=0; i<Ne; i++)
        {
            if ((sel_old[i][cue]==1)&&(Random()<(1-x*(1-f))))
            {
                gg[i]=1;
                delay_stim[i]=-ROUND(Random()*STIM_DELAY_E/dt);
                duration_stim[i]=ROUND((Tcue_PT*(1+width_stim*(.5-Random())))/dt);
            }
            if ((sel_old[i][cue]==0)&&(Random()<f*x))
            {
                gg[i]=1;
                delay_stim[i]=-ROUND(Random()*STIM_DELAY_E/dt);
                duration_stim[i]=ROUND((Tcue_PT*(1+width_stim*(.5-Random())))/dt);
            }
        }

        /* select inhibitory visual responsive */

        for (i=Ne; i<Ne+Ni; i++)
        {
            if ((sel_old[i][cue]==1)&&(Random()<(1-x*(1-f))))
            {
                gg[i]=1;
                delay_stim[i]=-ROUND(Random()*STIM_DELAY_I/dt);
                duration_stim[i]=ROUND((Tcue_PT*(1+width_stim*(.5-Random())))/dt);
            }
            if ((sel_old[i][cue]==0)&&(Random()<f*x))
            {
                gg[i]=1;
                delay_stim[i]=-ROUND(Random()*STIM_DELAY_I/dt);
                duration_stim[i]=ROUND((Tcue_PT*(1+width_stim*(.5-Random())))/dt);
            }
        }
    }
    else
    {
        //generate a noise pattern
        for (i=0; i<Ne; i++)
        {
            if (Random()<f_noise)
            {
                duration_stim[i]=ROUND((Tcue_PT*(1+width_stim*(.5-Random())))/dt);
                delay_stim[i]=-ROUND(Random()*STIM_DELAY_E/dt);
                gg[i]=1;
            }
        }
    }
}

void single_trial(int cue)
{
    int i, i2, n, j,memory_id;

    char flag_swap;
    float mext, vext, tau, h;
    float cE_sel, cE_nsel, cI_sel, cI_nsel;
    int tauarp;

    int p;

    unsigned int life;
    unsigned int totsp_cue, totsp_test, totsp_others, totsp_bk, totsp_total;

    int popstim, flagstim;
    int tbound, deltatmin, deltatstep, deltat;
    int Ncue, Ntest, Nn, Ntotal;

    unsigned int tcamp, tsyn, tcampV;
    int *stimrate, i_sel, i_nonsel;  // Use pointers instead of variable-length arrays

    char buf[30], flag_i_sel=0, flag_i_nonsel=0;
    float nuI_s, nuI_n;
    int count_swap;

    float tot_currE;
    float u_s, x_s, ux_s, u_n, x_n, ux_n, run_u, run_x;

    if (flag_SA==1)
    {
        Tprestim=T_SA;
        Tcue=0.0;
        Tdelay=0.0;
    }
    else
    {
        Tprestim=Tprestim_PT;
        Tcue=Tcue_PT;
        Tdelay=Tdelay_PT;
    }

    Time=0;
    life=(unsigned int)ROUND((Tprestim+Tcue+Tdelay)/dt);
    tbound=ROUND((MinDelay + MaxStep*StepDelay)/dt)+1;
    deltatmin=ROUND((MinDelay)/dt);
    deltatstep=ROUND((StepDelay)/dt);

    tcamp=(unsigned int)ROUND(TCamp/dt);

    (void)select_VR(cue);
    Tonset=Tprestim; Toffset=Tprestim+Tcue;
    (void)prepare_popout();
    if (cue<pops)
    {
        cE_sel=contrastE_sel; cE_nsel=contrastE_nonsel;
        cI_sel=contrastI_sel; cI_nsel=contrastI_nonsel;
    }
    else
    {
        cE_sel=contrastE_sel_noise; cE_nsel=contrastE_nonsel_noise;
        cI_sel=contrastI_sel_noise; cI_nsel=contrastI_nonsel_noise;
    }

    currI=0; spikesI=0;

    // Allocate memory to stimrate
    stimrate = (int*)malloc(pops * sizeof(int));
    if (stimrate == NULL)
    {
        printf("Error: Failed to allocate stimrate in single_trial\n");
        return;
    }
    while(Time<life)
    {
        tauarp=ROUND(Tarpge/dt); theta=ThrE;
        tau=Taue;

        /* sampling rate */

        if (Time%tcamp==tcamp-1)
        {
            fprintf(ratepops,"%.2f ",(Time+1)*dt);
            fprintf(curr_out,"%.2f ",(Time+1)*dt);
            totsp_cue=0; totsp_others=0; totsp_total=0;
            for (p=0; p<pops; p++)
            {
                totsp_cue=0; tot_currE=0; Nn=0;
                for (i=0; i<Ne; i++)
                    if (sel_old[i][p]==1)
                    {
                        totsp_cue+=spikes[i];
                        tot_currE+=curr[i];
                        Nn++;
                    }
                fprintf(ratepops,"%.2f ",(float)totsp_cue/TCamp/Nn*1000);
                fprintf(curr_out,"%.4f ",tot_currE/TCamp/Nn);
            }

            fprintf(ratepops,"%.2f\n",(float)spikesI/TCamp/Ni*1000);
            fprintf(curr_out,"%.4f\n",currI/TCamp/Ni);

            fflush(ratepops); fflush(curr_out);

            spikesI=0; currI=0;
            for (i=0; i<Ne; i++)
            { spikes[i]=0; curr[i]=0; }

            /* end sampling rate */
        }

        /* Gaussian Currents  - Prepare external currents */

        flag_popout=0;
        if ((flag_SA==0)&&((Time*dt)>=T_popout))
            flag_popout=1;

        for (i=0; i<Ntot; i++)
        {
            mext=muEext*dt/Taue; vext=sEext*dt/Taue;
            if (i>Ne)
            { mext=muIext*dt/Taui; vext=sIext*dt/Taui; }

            if ((flag_popout==1)&&(i<Ne))
            {
                duration_popout[i]--;
                if (duration_popout[i]>=0)
                    mext*=g_popout;
                else
                    duration_popout[i]=-1;
            }

            flagstim=0;
            if (((Time*dt)>=Tonset)&&((Time*dt)<Toffset)&&(flag_SA==0))
                flagstim=1;

            if ((flagstim==1))
            {
                delay_stim[i]++;
                if (delay_stim[i]>=0)
                {
                    delay_stim[i]=1;
                    duration_stim[i]--;
                    if (duration_stim[i]>0)
                    {
                        if (i<Ne)
                        {
                            if (gg[i]==1)
                            { mext*=cE_sel; vext*=cE_sel; }
                            else
                            { mext*=cE_nsel; vext*=cE_nsel; }
                        }
                        else
                        {
                            if (gg[i]==1)
                            { mext*=cI_sel; vext*=cI_sel; }
                            else
                            { mext*=cI_nsel; vext*=cI_nsel; }
                        }
                    }
                    else
                        duration_stim[i]=-1;
                }
            }

            Iext[i]=mext+NormDev()*sqrt(vext);
        }

        /* Cycle over neurons - i = pre-synaptic */

        for (i=0; i<Ntot; i++)
        {
            tau=Taue;
            if (i>Ne)
            {
                tauarp=ROUND(Tarpgi/dt);
                theta=ThrI;
                tau=Taui;
            }

            /* manage NMDA */

            Inmda[i]*=exp(-dt/TauNMDA);
            Inmda[i]+=EventsNMDA[Time%tbound][i];
            EventsNMDA[Time%tbound][i]=0;

            if (i<Ne)
                curr[i]+=Events[Time%tbound][i];
            else
                currI+=Events[Time%tbound][i];

            if(neur[i].status>=0) /* ci troviamo in fase refrattaria? */
            {
                neur[i].status++;
                neur[i].V*=exp(-dt/tau);
                neur[i].V+=Inmda[i]*dt/tau;
                neur[i].V+=Iext[i];
                neur[i].V+=Events[Time%tbound][i]; /* incoming spikes */
                Events[Time%tbound][i]=0;

                if(neur[i].V>=theta)
                {
                    neur[i].V=-1.0;
                    neur[i].status=-tauarp;

                    if (i<Ne)
                        (void)STP(i,Time);

                    //--------------------------------------------------raster output-----------------=
					//write out spike rasters
					if ((i%10==0))
					{
					    fprintf(spikeout,"%d\t%.2f\t",i,(Time*dt));
					    if (i<Ne)
					    {
					 
					        memory_id = 0;
					        for (p = 0; p < pops; p++) 
							{
					            if (sel_old[i][p] == 1) 
								{
					                memory_id = p + 1;  // The memory number starts from 1
					                break;
					            }
					        }
					        fprintf(spikeout,"%d\t", memory_id);
					        // ========================
					        
					        if (gg[i]==1)
					            fprintf(spikeout,"1\n");
					        else
					            fprintf(spikeout,"0\n");
					    }
					    else
					        fprintf(spikeout,"-1\t-1\n");
					
					    fflush(spikeout);
					}

                    /* propagate spike */
                    i2=0;
                    for(n=0;n<J[i].num;n++)
                    {
                        i2+=J[i].eff[n].dist;
                        deltat=(int)(deltatmin+deltatstep*(J[i].eff[n].delay));
                        if (i<Ne)
                        {
                            if(i2<Ne)
                            {
                                Events[(Time+deltat)%tbound][i2]+=((1-xNMDA_EE)*neur[i].uxtm*J[i].eff[n].efficacy);
                                EventsNMDA[(Time+deltat)%tbound][i2]+=(Taue*xNMDA_EE*neur[i].uxtm*J[i].eff[n].efficacy/TauNMDA);
                            }
                            else
                            {
                                Events[(Time+deltat)%tbound][i2]+=(1-xNMDA_IE)*J[i].eff[n].efficacy;
                                EventsNMDA[(Time+deltat)%tbound][i2]+=Taui*xNMDA_IE*(J[i].eff[n].efficacy)/TauNMDA;
                            }
                        }
                        else
                            Events[(Time+deltat)%tbound][i2]+=J[i].eff[n].efficacy;
                    }
                    //update tspike
                    neur[i].tspike=Time;

                    if (i<Ne)
                        spikes[i]++;
                    else
                        spikesI++;
                }
            }
            else
            {
                /* fase refrattaria */
                neur[i].status++;
                Events[Time%tbound][i]=0;

                if(neur[i].status==0) 
                    neur[i].V=(float)(theta*neur[i].Hr);
            }
        }

        if (Time%tcamp==tcamp-1)
        {
            /* start sampling synaptic variables */
            fprintf(synvar_u,"%.2f\t",(Time+1)*dt);
            fprintf(synvar_x,"%.2f\t",(Time+1)*dt);
            for (p=0; p<pops; p++)
            {
                u_s=0; x_s=0; Nn=0;
                for (i=0; i<Ne; i++)
                    if (sel_old[i][p]==1)
                    {
                        u_s+=readout(i,Time,1);
                        x_s+=readout(i,Time,0);
                        Nn++;
                    }
                fprintf(synvar_u,"%.5f\t",u_s/Nn);
                fprintf(synvar_x,"%.5f\t",x_s/Nn);
            }
            fprintf(synvar_u,"\n"); fprintf(synvar_x,"\n");
            fflush(synvar_u); fflush(synvar_x);
        }
        Time++; 
    }

    free(stimrate);
}

void naive_mf()
{
    float Ce, Ci, Jee;

    avg_u=U*(1+tau_F*nue)/(1+U*tau_F*nue);
    avg_x=1/(1+(1-avg_u)*tau_D*nue);
    avg_x2=avg_x/(1+avg_u*tau_D*nue*(1-.5*avg_u));

    Ce=pconn*Ne; Ci=pconn*Ni;
    Jee=Cp*Jpot+(1-Cp)*Jdep;

    mu_e_mf=Ce*Jee*avg_u*avg_x*nue+Ci*Jie*nui;
    mu_i_mf=Ce*Jei*nue+Ci*Jii*nui;
    s2_e_mf=Ce*Jee*Jee*avg_u*avg_u*avg_x2*nue+Ci*Jie*Jie*nui;
    s2_i_mf=Ce*Jei*Jei*nue+Ci*Jii*Jii*nui;

    muV_e=Taue*mu_e_mf+muEext; s2V_e=Taue*s2_e_mf+sEext;
    muV_i=Taui*mu_i_mf+muIext; s2V_i=Taui*s2_i_mf+sIext;
}

int main()
{
    int ai, mi, nt, cue;
    float erpha_val, tau_D_val;
    int *stim_to_be;
    char filename[256];
    
    // Set the random seed
    SetRandomSeed(12345);

    // Outer loop erpha (1.0 to 3.0, step size 0.1)
    for (ai = 0; ai <= 20; ai++)
    {
        erpha_val = 1.0f + ai * 0.1f;

        //Inner loop tau_D (ranging from 100.0 to 250.0, with a step size of 10.0)
        for (mi = 0; mi <= 15; mi++)
        {
            tau_D_val = 100.0f + mi * 10.0f;

            // Set global parameters
            erpha = erpha_val;
            tau_D = tau_D_val;

            
            naive_mf();
            InitV();
            InitEventMatrix();
            stim_to_be = (int*)malloc(pops * sizeof(int));
            if (stim_to_be == NULL)
            {
                printf("Error: Failed to allocate stim_to_be\n");
                return 1;
            }

            // Build the synaptic structure
            if (flag_fs == 1)
            {
                sel_old = createPop_noov();
                makePop(sel_old);
                initSynapses_nofs();
                invert_nofs_udo();
                InitSynapses();
            }
            else
            {
                sel_old = createPop(1);
                makePop(sel_old);
                InitSynapses();
            }

            // ============ Spontaneous Activity (SA) Test ============
            flag_SA = 1;

            sprintf(filename, "SA_rates_pops_erpha_%.2f_tau_D_%.2f.log", erpha, tau_D);
            ratepops = fopen(filename, "w");
            sprintf(filename, "SA_rasters_erpha_%.2f_tau_D_%.2f.log", erpha, tau_D);
            spikeout = fopen(filename, "w");
            sprintf(filename, "SA_stp_u_erpha_%.2f_tau_D_%.2f.log", erpha, tau_D);
            synvar_u = fopen(filename, "w");
            sprintf(filename, "SA_stp_x_erpha_%.2f_tau_D_%.2f.log", erpha, tau_D);
            synvar_x = fopen(filename, "w");
            sprintf(filename, "SA_currents_erpha_%.2f_tau_D_%.2f.log", erpha, tau_D);
            curr_out = fopen(filename, "w");

            single_trial(0);
            shifttspike();

            fclose(ratepops);
            fclose(spikeout);
            fclose(synvar_u);
            fclose(synvar_x);
            fclose(curr_out);

            // ============ Task (PT) Test ============
            flag_SA = 0;
            for (nt = 0; nt < num_pres; nt++)
            {
                cue = (nt < pops) ? nt : nt - pops;

                sprintf(filename, "rates_pops_%04d_erpha_%.2f_tau_D_%.2f.log", nt, erpha, tau_D);
                ratepops = fopen(filename, "w");
                sprintf(filename, "rasters_%04d_erpha_%.2f_tau_D_%.2f.log", nt, erpha, tau_D);
                spikeout = fopen(filename, "w");
                sprintf(filename, "stp_u_%04d_erpha_%.2f_tau_D_%.2f.log", nt, erpha, tau_D);
                synvar_u = fopen(filename, "w");
                sprintf(filename, "stp_x_%04d_erpha_%.2f_tau_D_%.2f.log", nt, erpha, tau_D);
                synvar_x = fopen(filename, "w");
                sprintf(filename, "currents_%04d_erpha_%.2f_tau_D_%.2f.log", nt, erpha, tau_D);
                curr_out = fopen(filename, "w");

                single_trial(cue);
                shifttspike();

                fclose(ratepops);
                fclose(spikeout);
                fclose(synvar_u);
                fclose(synvar_x);
                fclose(curr_out);
            }

            free(stim_to_be);

            printf("Finished erpha=%.2f tau_D=%.2f\n", erpha, tau_D);
        }
    }

    return 0;
}