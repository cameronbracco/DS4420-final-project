// you will find a civilised, full, version on the website. This is as compact as bearable
//  no, it is not perfectly clean
// setup the CPU_THREAD depending on your CPU. 4 is run of the mill and default
// to compile without GPU: gcc sonn.c -o sonn -O2 -pthread
// if have a GPU: uncomment #define GPU and see your CUDA documentation.
//
//  throughout, variable ‘a’ is a group, ’n’ a neuron in a group, ‘s’ a connection in a neuron
//#define GPU 1
#define CPU_THREAD 4
// warning without GPU: 180 hours with 4 threads to 98.9% , 18h to 98.65%

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>

// dataset
#define NTYPE 10
#define INSIZE 784
// size. NSYN: For 98.9%: 7920. For 98.65% : 792
#define NSYN 10 // number of connections per neuron. 
#define NSIZE 792 // multiple of CPU_THREAD. Number of neurons per column
// params
#define THRESHOLD 1000000
#define THRESHOLD_MIN 300000
#define THRESHOLD_MAX 1200000
#define MIN_WEIGHT 97000

#define NCYC 4
unsigned char wd[4] = {192, 128, 64, 1};

int list[3][60000], listSize[3] = {60000, 10000, 1000}, listSet[3] = {0, 1, 0}, setsize[2];
unsigned char type[2][60000], in[2][60000][INSIZE];
int notnuls[60000], nbc[NTYPE], nbcn[NTYPE][NSIZE], zero[60000 * NTYPE], err[60000];
int nsp[3][60000][NTYPE]; // number of spikes per group and sample
int **weight[NTYPE], *tmpw;
short int **from[NTYPE], *tmpf;

// params
float quantP, quantUpP, quantN, quantUpN, quantDivP, quantDivN;
int adjp, adjn, lossw, losswNb, nbNew, divNew, briefStep;
// globals
int nbu, nblearn, nbtested, listNb;
int toreload[NTYPE];

// threading
struct thrd_arg
{
    int from;
    int to;
    int trhdNo;
};

struct timespec mtwait;
int mtact_learn[CPU_THREAD], mtact_test[CPU_THREAD];
struct thrd_arg thrd_args[CPU_THREAD], thrd_test_args[CPU_THREAD];
int thrd_test_first[CPU_THREAD], thrd_test_last[CPU_THREAD];
int mtadj, mtsample, mta;

int rnd(int max) { return rand() % max; }

void init()
{
    int a, n, set, l;

    for (a = 0; a < NTYPE; a++)
    {
        weight[a] = (int **)calloc(NSIZE, sizeof(int *));
        from[a] = (short int **)calloc(NSIZE, sizeof(short int *));
        for (n = 0; n < NSIZE; n++)
        {
            weight[a][n] = (int *)calloc(NSYN, sizeof(int));
            from[a][n] = (short int *)calloc(NSYN, sizeof(short int));
        }
    }
    tmpw = (int *)calloc(NTYPE * NSIZE * NSYN, sizeof(int));
    tmpf = (short int *)calloc(NTYPE * NSIZE * NSYN, sizeof(short int));
    for (set = 0; set < 2; set++)
        for (l = 0; l < setsize[set]; l++)
            list[set][l] = l;
}

int compinc(const void *a, const void *b)
{
    if (*((int *)a) > *((int *)b))
        return 1;
    else
        return -1;
}

void quant(int sample)
{
    int is, isnot, tot, a; 
    // "is" - correct class (0-9)
    // "isnot" - incorrect class (0-9)
    // "tot" - ??
    // "a" - ??
    // "d" - ??
    float d;

    is = type[0][sample];

    if ((float)(err[sample]) != quantP) // quantP = 0
    {
        if ((float)(err[sample]) >= quantP) // quantP = 0
            quantP += quantUpP / quantDivP; // quantP += 19 / 100
        else
            quantP -= 1.0 / quantDivP; // quantP -= 1 / 100
    }

    tot = 0;
    for (a = 0; a < NTYPE; a++) // NTYPE = classes (10)
        tot += nsp[2][sample][a]; // what is nsp?? I think its the number of spikes? (0 - 10)??

    for (isnot = 0; isnot < NTYPE; isnot++) 
    {
        if (isnot != is)
        {
            // total = 9
            // 
            // d = 1 - (total - 1 / 9)
            d = (float)(nsp[2][sample][isnot] - (tot - nsp[2][sample][isnot]) / 9);
            if (d != quantN) // quantN = 0
            {
                if (d >= quantN)
                    quantN += quantUpN / quantDivN; // quantN += 150 / 900
                else
                    quantN -= 1.0 / quantDivN; // quantN -= 1 / 900
            }
        }
    }
}

#ifdef GPU
cudaStream_t streams[60000];
short int *gpu_from;
int *gpu_weight, *gpu_nsp, *gpu_list[3], *gpu_nspg, *gpu_togrp;
unsigned char *gpu_wd, *gpu_in[2];

__global__ void Kernel_Test(int nsize, int nsyn, int ncyc, short int *from,
                            int *weight, unsigned char *in, unsigned char *wd, int *nsp,
                            int listNb, int *list, int sh, int nsh)
{
    int s, n, index, ana, tot, ns, cycle, off7sp, sp;

    n = blockIdx.x * blockDim.x + threadIdx.x;
    ns = n * nsyn;
    ana = n / nsize;
    n = n - nsize * ana;
    if (n < nsize && ana < NTYPE)
    {
        for (index = sh; index < sh + nsh; index++)
        {
            off7sp = list[index] * INSIZE;
            tot = 0;
            sp = 0;
            for (cycle = 0; cycle < ncyc; cycle++)
            {
                for (s = 0; s < nsyn; s++)
                    tot += (in[off7sp + from[ns + s]] >= wd[cycle]) * weight[ns + s];

                if (tot > THRESHOLD)
                {
                    sp++;
                    tot = 0;
                }
                else
                    tot >>= 1;
            }
            if (sp)
                atomicAdd(nsp + index * NTYPE + ana, sp);
        }
    }
}

void init_gpu()
{
    int set, i;

    cudaSetDevice(0);

    cudaMalloc((void **)&gpu_wd, NCYC * sizeof(unsigned char));
    for (set = 0; set < 2; set++)
        cudaMalloc((void **)&gpu_in[set], setsize[set] * INSIZE * sizeof(unsigned char));
    for (set = 0; set < 3; set++)
        cudaMalloc((void **)&gpu_list[set], listSize[set] * sizeof(int));
    cudaMalloc((void **)&gpu_from, NTYPE * NSIZE * NSYN * sizeof(short int));
    cudaMalloc((void **)&gpu_weight, NTYPE * NSIZE * NSYN * sizeof(int));
    cudaMalloc((void **)&gpu_nsp, setsize[0] * NTYPE * sizeof(int));
    cudaDeviceSynchronize();

    cudaMemcpy(gpu_wd, wd, NCYC * sizeof(unsigned char), cudaMemcpyHostToDevice);
    for (set = 0; set < 2; set++)
    {
        for (i = 0; i < setsize[set]; i++)
            cudaMemcpy(gpu_in[set] + i * INSIZE, in[set][i], INSIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_list[set], list[set], setsize[set] * sizeof(int), cudaMemcpyHostToDevice);
    }
    for (i = 0; i < 6000; i++)
        cudaStreamCreate(&(streams[i]));
    cudaDeviceSynchronize();
}

void test_gpu()
{
    int i, a, n, b, is, isnot, ok, sample, g, bssize, startat, set, l;
    static int tmpn[NTYPE * 60000];

    set = listSet[listNb];
    for (a = 0; a < NTYPE; a++)
    { // upload from & weight
        if (toreload[a] == 1)
        {
            for (n = 0; n < NSIZE; n++)
            {
                bcopy(from[a][n], (void *)(tmpf + (a * NSIZE + n) * NSYN), NSYN * sizeof(short int));
                bcopy(weight[a][n], (void *)(tmpw + (a * NSIZE + n) * NSYN), NSYN * sizeof(int));
            }
            cudaMemcpy(gpu_from + a * NSIZE * NSYN, tmpf + a * NSIZE * NSYN, NSIZE * NSYN * sizeof(short int), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_weight + a * NSIZE * NSYN, tmpw + a * NSIZE * NSYN, NSIZE * NSYN * sizeof(int), cudaMemcpyHostToDevice);
            toreload[a] = 0;
        }
    }

    if (listNb == 2)
        cudaMemcpy(gpu_list[listNb], list[listNb], listSize[listNb] * sizeof(int), cudaMemcpyHostToDevice); // upload list
    cudaMemcpy(gpu_nsp, zero, NTYPE * setsize[0] * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // run
    if (listNb == 2)
        bssize = listSize[listNb];
    else
        bssize = 100;

    for (startat = 0; startat < listSize[listNb]; startat += bssize)
        Kernel_Test<<<((NTYPE * NSIZE + 31)) / 32, 32, 0, streams[startat]>>>(NSIZE, NSYN, NCYC, gpu_from, gpu_weight, gpu_in[set], gpu_wd, gpu_nsp,
                                                                              listNb, gpu_list[listNb], startat, bssize);
    cudaDeviceSynchronize();

    // download nsp
    cudaMemcpy(tmpn, gpu_nsp, listSize[listNb] * NTYPE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // dispatch
    for (l = 0; l < listSize[listNb]; l++)
    {
        if (listNb == 2)
            bcopy(tmpn + l * NTYPE, nsp[2][list[listNb][l]], NTYPE * sizeof(int));
        else
            bcopy(tmpn + l * NTYPE, nsp[set][l], NTYPE * sizeof(int));
    }
}
#endif

void readset(int set, char *name_lbl, char *name_data)
{
    int nn, n, fdl, fdi;

    setsize[set] = listSize[set];
    fdl = open(name_lbl, O_RDONLY);
    lseek(fdl, 8, SEEK_SET);
    fdi = open(name_data, O_RDONLY);
    lseek(fdi, 16, SEEK_SET);

    for (n = 0; n < listSize[set]; n++)
    {
        read(fdl, type[set] + n, 1);
        read(fdi, in[set][n], INSIZE);
        if (set == 0)
        {
            notnuls[n] = 0;
            for (nn = 0; nn < INSIZE; nn++)
                if (in[set][n][nn])
                    notnuls[n]++;
        }
    }
    close(fdl);
    close(fdi);
}

void *test_mt_one(void *args)
{
    struct thrd_arg *arg;
    int trhdNo, a, n, t, c, s, r, l, set, sample;

    arg = (struct thrd_arg *)args;
    trhdNo = arg->trhdNo;

    while (1 == 1)
    {
        while (!mtact_test[arg->trhdNo])
            nanosleep(&mtwait, NULL);

        set = listSet[listNb];
        for (l = thrd_test_first[trhdNo]; l < thrd_test_last[trhdNo]; l++)
        {
            sample = list[listNb][l];
            for (a = 0; a < NTYPE; a++)
            {
                nsp[listNb][sample][a] = 0;
                for (n = 0; n < NSIZE; n++)
                {
                    t = 0;
                    for (c = 0; c < NCYC; c++)
                    {
                        for (s = 0; s < NSYN; s++)
                            if (in[set][sample][from[a][n][s]] >= wd[c])
                                t += weight[a][n][s];

                        if (t > THRESHOLD)
                        {
                            nsp[listNb][sample][a]++;
                            t = 0;
                        }
                        else
                            t >>= 1;
                    }
                }
            }
        }

        mtact_test[trhdNo] = 0;
    }
}

int test_mt()
{
    int trhdNo, done;
    int set, ok, l, is, b, isnot, sample, a;

    l = 0;
    a = listSize[listNb] / CPU_THREAD;
    b = listSize[listNb] - a * CPU_THREAD;
    for (trhdNo = 0; trhdNo < CPU_THREAD; trhdNo++)
    {
        thrd_test_first[trhdNo] = l;
        thrd_test_last[trhdNo] = l + a;
        if (trhdNo < b)
            thrd_test_last[trhdNo]++;
        l = thrd_test_last[trhdNo];
        mtact_test[trhdNo] = 1;
    }

    done = 0;
    while (done != CPU_THREAD)
    {
        nanosleep(&mtwait, NULL);
        done = 0;
        for (trhdNo = 0; trhdNo < CPU_THREAD; trhdNo++)
            done += 1 - mtact_test[trhdNo];
    }
}

int test()
{
    int set, ok, l, is, b, isnot, sample, a;

#ifdef GPU
    test_gpu();
#else
    test_mt();
#endif
    set = listSet[listNb];
    ok = 0;
    for (l = 0; l < listSize[listNb]; l++)
    {
        sample = list[listNb][l];
        is = type[set][sample];
        b = -1;
        for (a = 0; a < NTYPE; a++)
        {
            if (a != is && nsp[listNb][sample][a] > b)
            {
                b = nsp[listNb][sample][a];
                isnot = a;
            }
        }
        if (nsp[listNb][sample][is] > b)
            ok++;
        if (listNb == 2)
        {
            err[sample] = b - nsp[2][sample][is];
            quant(sample);
        }
    }
    return ok;
}

void test_mt_init()
{
    int trhdNo;
    pthread_t thrds[CPU_THREAD];

    for (trhdNo = 0; trhdNo < CPU_THREAD; trhdNo++)
    {
        thrd_test_args[trhdNo].trhdNo = trhdNo;
        pthread_create(&thrds[trhdNo], NULL, test_mt_one,
                       (void *)&thrd_test_args[trhdNo]);
    }
}

void loss(int b)
{
    int a, n, s, w;

    for (a = 0; a < NTYPE; a++)
    {
        for (n = 0; n < NSIZE; n++)
        {
            for (s = 0; s < NSYN; s++)
            {
                w = weight[a][n][s];
                if (w)
                {
                    if (lossw && b % losswNb == 0)
                    {
                        if (w > 0)
                            w -= lossw;
                        else
                            w += lossw;
                    }
                    if (abs(w) < MIN_WEIGHT)
                    {
                        w = 0;
                        nbc[a]--;
                        nbcn[a][n]--;
                    }
                    weight[a][n][s] = w;
                }
            }
        }
        toreload[a] = 1;
    }
}

void connect(int nb, int sample, int a, int init, int g)
{
    int p, p0, f, f0, n, s, i, ii, pn;
    // short int f0 ;
    static int ps[1000];

    if (nbc[a] == NSIZE * NSYN)
        return;
    if (NSIZE * NSYN - nbc[a] < nb)
        nb = NSIZE * NSYN - nbc[a];

    for (i = 0; i < nb; i++)
    {
        ii = -1;
        while (i != ii)
        {
            ps[i] = rnd(NSIZE * NSYN - nbc[a]) + 1;
            for (ii = 0; ii < i; ii++)
                if (ps[ii] == ps[i])
                    break;
        }
    }
    qsort(ps, nb, sizeof(int), compinc);

    n = 0;
    s = 0;
    p0 = 0;

    for (i = 0; i < nb; i++)
    {
        p = ps[i];
        for (; n < NSIZE && p0 != p; n += (p0 != p))
        {
            if (p0 + NSYN - nbcn[a][n] < p)
                p0 += NSYN - nbcn[a][n];
            else
                for (s *= (pn == n); s < NSYN && p0 != p; s += (p0 != p))
                    if (!weight[a][n][s])
                        p0++;
        }

        f = rnd(notnuls[sample]) + 1;
        for (f0 = 0; f0 < INSIZE && f; f0 += (f != 0))
            if (in[0][sample][f0])
                f--;

        from[a][n][s] = f0;
        weight[a][n][s] = init;
        nbc[a]++;
        nbcn[a][n]++;
        pn = n;
    }
}

void *learn_mt_one(void *args)
{
    struct thrd_arg *arg;
    int to, start, trhdNo;
    int a, n, c, s, sample, adj;
    int tot, cnta, mask;

    arg = (struct thrd_arg *)args;
    while (1 == 1)
    {
        while (!mtact_learn[arg->trhdNo])
            nanosleep(&mtwait, NULL);

        for (n = arg->from; n < arg->to; n++)
        {
            tot = 0;
            cnta = 0;
            for (c = 0; c < NCYC; c++)
            {
                mask = 1;
                for (s = 0; s < NSYN; s++)
                {
                    if (weight[mta][n][s] && in[0][mtsample][from[mta][n][s]] >= wd[c])
                    {
                        tot += weight[mta][n][s];
                        cnta |= mask;
                    }
                    mask <<= 1;
                }

                if (tot > THRESHOLD_MIN)
                {
                    if (tot < THRESHOLD_MAX)
                    {
                        mask = 1;
                        for (s = 0; s < NSYN; s++)
                        {
                            if ((cnta & mask) != 0 && weight[mta][n][s])
                            {
                                if (abs(weight[mta][n][s] + mtadj) < THRESHOLD)
                                {
                                    weight[mta][n][s] += mtadj;
                                    if (abs(weight[mta][n][s]) < MIN_WEIGHT)
                                    {
                                        weight[mta][n][s] = 0;
                                        nbcn[mta][n]--;
                                    }
                                }
                            }
                            mask <<= 1;
                        }
                    }
                    tot = 0;
                    cnta = 0;
                }
                tot >>= 1;
            }
        }
        mtact_learn[arg->trhdNo] = 0;
    }
}

void learn(int sample, int a, int adj)
{
    int trhdNo, done, n;

    mta = a;
    mtadj = adj;
    mtsample = sample;
    for (trhdNo = 0; trhdNo < CPU_THREAD; trhdNo++)
        mtact_learn[trhdNo] = 1;

    done = 0;
    while (done != CPU_THREAD)
    {
        nanosleep(&mtwait, NULL);
        done = 0;
        for (trhdNo = 0; trhdNo < CPU_THREAD; trhdNo++)
            done += 1 - mtact_learn[trhdNo];
    }
    nbc[mta] = 0;
    for (n = 0; n < NSIZE; n++)
        nbc[mta] += nbcn[mta][n];
}

void learn_mt_init()
{
    int start, trhdNo, sssize;
    pthread_t thrds[CPU_THREAD];

    sssize = (NSIZE + CPU_THREAD - 1) / CPU_THREAD;
    start = 0;
    trhdNo = 0;

    while (start < NSIZE)
    {
        thrd_args[trhdNo].from = start;
        thrd_args[trhdNo].to = start + sssize;
        if (start + sssize >= NSIZE)
            thrd_args[trhdNo].to = NSIZE;
        thrd_args[trhdNo].trhdNo = trhdNo;
        pthread_create(&thrds[trhdNo], NULL, learn_mt_one,
                       (void *)&thrd_args[trhdNo]);
        start += sssize;
        trhdNo++;
    }
}

void batch()
{
    int b, bu, a, l, sample, is, isnot, g, tot, led, prevStep;
    static int sel[60000];
    float d;

    prevStep = 0;
    nbtested = 0;
    bu = 0;
    b = 0;
    while (1 == 1)
    {                                          // batch selection
        bcopy(zero, sel, 60000 * sizeof(int)); // no duplicate
        for (l = 0; l < listSize[2]; l++)
        {
            list[2][l] = rnd(60000);
            while (sel[list[2][l]])
                list[2][l] = rnd(60000);
            sel[list[2][l]] = 1;
        }

        listNb = 2;
        test();
        nbtested++;

        for (l = 0; l < listSize[2]; l++)
        {
            sample = list[2][l];
            is = type[0][sample];

            if ((float)(err[sample]) >= quantP || nbtested > b /*lim.@start*/)
            {
                connect(nbNew, sample, is, THRESHOLD / divNew, g);
                learn(sample, is, adjp);
                toreload[is] = 1;
                nblearn++;
                b++;
                if (lossw && nblearn % losswNb == 0)
                    loss(b);
            }

            if (bu > 3 * b / 2)
                continue; // limiter for faster startup.

            tot = 0;
            for (a = 0; a < NTYPE; a++)
                tot += nsp[2][sample][a];
            for (isnot = 0; isnot < NTYPE; isnot++)
            {
                if (isnot != is)
                {
                    d = (float)(nsp[2][sample][isnot] - (tot - nsp[2][sample][isnot]) / 9);
                    if (quantUpN != 0.0 && d >= quantN)
                    {
                        connect(nbNew, sample, isnot, -THRESHOLD / divNew, g);
                        learn(sample, isnot, adjn);
                        toreload[isnot] = 1;
                        bu++;
                    }
                }
            }
            if (/*b/briefStep >prevStep*/ b % briefStep == 0 && b != prevStep)
            {
                printf("testing %d: ", nblearn);
                listNb = 0;
                printf("0: %2.3f  ", (float)test() / 600.0);
                listNb = 1;
                printf("1: %2.2f\n", (float)test() / 100.0);
                prevStep = b /*/briefStep*/;
            }
        }
    }
}

int main()
{
    srand(1000);          // time(NULL) for random
    setbuf(stdout, NULL); // unbuffering stdout
    mtwait.tv_sec = 0;
    mtwait.tv_nsec = 10;
    // Download the files at www.yann.lecun.com/
    readset(0, "./train-labels-idx1-ubyte", "./train-images-idx3-ubyte");
    readset(1, "./t10k-labels-idx1-ubyte", "./t10k-images-idx3-ubyte");
    init();
#ifdef GPU
    init_gpu();
#else
    test_mt_init();
#endif
    learn_mt_init();

    // params
    briefStep = 20000; // increase to speed up
    listSize[2] = 200;
    quantP = 0.0;
    quantDivP = 100;
    quantUpP = 19.0;
    quantN = 0.0;
    quantDivN = 900;
    quantUpN = 150.0; // 98.9 : quantDivP=100
    adjp = 500;
    adjn = -500; // 98.9 : 1000 / -1000
    divNew = 10;
    nbNew = 10; // 98.9 : 10 / 20
    lossw = 20;
    losswNb = 100; // 98.9 : 10 / 100

    batch();
}