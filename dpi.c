// you will find a civilised, full, version on the website. This is as compact as bearable
//  no, it is not perfectly clean
// setup the CPU_THREAD depending on your CPU. 4 is run of the mill and default
// to compile without GPU: gcc sonn.c -o sonn -O2 -pthread
// if have a GPU: uncomment #define GPU and see your CUDA documentation.
//
//  throughout, variable ‘a’ is a group, ’n’ a neuron in a group, ‘s’ a connection in a neuron
#define GPU 1
#define CPU_THREAD 4
// warning without GPU: 180 hours with 4 threads to 98.9% , 18h to 98.65%

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <strings.h>

// dataset
#define OUTPUT_SIZE 10
#define INPUT_SIZE 784
#define NUM_EXAMPLES_TRAIN 60000
#define NUM_EXAMPLES_TEST 10000
#define NUM_EXAMPLES_SOMETHING 1000
// size. NUM_CONNECTIONS_PER_NEURON: For 98.9%: 7920. For 98.65% : 792
#define NUM_CONNECTIONS_PER_NEURON 10 // number of connections per neuron.
#define NUM_NEURONS_PER_COLUMN 792 // multiple of CPU_THREAD. Number of neurons per column
// params
#define THRESHOLD 1000000
#define THRESHOLD_MIN 300000
#define THRESHOLD_MAX 1200000
#define MIN_WEIGHT 97000

#define NUM_THRESHOLD_CYCLES 4 // Number of pixel intensity thresholds
unsigned char intensityThresholds[NUM_THRESHOLD_CYCLES] = {192, 128, 64, 1};

int indexOfSampleInSplit[3][NUM_EXAMPLES_TRAIN];
int splitSizes[3] = {NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST, NUM_EXAMPLES_SOMETHING};
int splitToDatasetMap[3] = {0, 1, 0};
int splitSize[2];
unsigned char rawLabels[2][NUM_EXAMPLES_TRAIN]; // Maybe for train/test?
int rawData[2][NUM_EXAMPLES_TRAIN][INPUT_SIZE]; // Don't know why 2 "channels"
int notnuls[NUM_EXAMPLES_TRAIN];
int nbc[OUTPUT_SIZE];
int nbcn[OUTPUT_SIZE][NUM_NEURONS_PER_COLUMN];
int zero[NUM_EXAMPLES_TRAIN * OUTPUT_SIZE]; // Literally just a zero-d out array used for memory stuff
int err[NUM_EXAMPLES_TRAIN]; // Maximum false column spike count - true column spike count
int spikeCounts[3][NUM_EXAMPLES_TRAIN][OUTPUT_SIZE]; // number of spikes per group and sample
int **weights[OUTPUT_SIZE];
int *tempWeights;
short int **receptorIndices[OUTPUT_SIZE];
short int *tempReceptorIndices;

// params
// DIFFERENCE: ONLY ONE POSITIVE AND NEGATIVE QUANTILIZER
// DIFFERENCE: QUANTILIZERS USE FLOATS
float positiveQuantVal;
float positiveQuantValDeltaUp;
float positiveQuantValDenom;

float negativeQuantVal;
float negativeQuantValDeltaUp;
float negativeQuantValDenom;
int adjp, adjn, lossw, losswNb, nbNew, divNew, briefStep;
// globals
int nbu;
int nblearn;
int numBatchesTested;
int listNb;
int toreload[OUTPUT_SIZE];

// threading
struct ThreadArgs {
    int startIndex; // What index to start at
    int endIndex; // What index to end at
    int threadNumber; // This thread's index
};

struct timespec multiThreadedWait;
int mtact_learn[CPU_THREAD];
int mtact_test[CPU_THREAD];
struct ThreadArgs threadArgs[CPU_THREAD];
struct ThreadArgs threadTestArgs[CPU_THREAD];
int thrd_test_first[CPU_THREAD];
int thrd_test_last[CPU_THREAD];
int mtadj;
int mtsample;
int mta;

int rnd(int max) { return rand() % max; }

void init() {

    for (int column = 0; column < OUTPUT_SIZE; column++) {
        // Allocate space for all the neuron weights
        weights[column] = (int **) calloc(NUM_NEURONS_PER_COLUMN, sizeof(int *));
        // Allocate space for neuron connections
        receptorIndices[column] = (short int **) calloc(NUM_NEURONS_PER_COLUMN, sizeof(short int *));

        for (int neuron = 0; neuron < NUM_NEURONS_PER_COLUMN; neuron++) {
            weights[column][neuron] = (int *) calloc(NUM_CONNECTIONS_PER_NEURON, sizeof(int));
            receptorIndices[column][neuron] = (short int *) calloc(NUM_CONNECTIONS_PER_NEURON, sizeof(short int));
        }
    }

    // Allocates space flattened for use in GPU
    tempWeights = (int *) calloc(OUTPUT_SIZE * NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON, sizeof(int));
    tempReceptorIndices = (short int *) calloc(OUTPUT_SIZE * NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON, sizeof(short int));

    for (int split = 0; split < 2; split++) {
        for (int sampleIndex = 0; sampleIndex < splitSize[split]; sampleIndex++) {
            // Literally just a list from 1 to N for the first two splits
            // What does it do for list three?
            indexOfSampleInSplit[split][sampleIndex] = sampleIndex;
        }
    }
}

int compinc(const void *a, const void *b) {
    if (*((int *) a) > *((int *) b)) {
        return 1;
    } else {
        return -1;
    }
}

void quant(int sampleIdx) {
    int yTrue = rawLabels[0][sampleIdx];

    if ((float) (err[sampleIdx]) != positiveQuantVal)
    {
        // If we are "above the mark"
        if ((float) (err[sampleIdx]) >= positiveQuantVal)
        {
            positiveQuantVal += positiveQuantValDeltaUp / positiveQuantValDenom; // Always add 19/100
        }
        else {
            positiveQuantVal -= 1.0 / positiveQuantValDenom; // quantP -= 1 / 100 // Always subtract 1/100
        }
    }

    int totalSpikeCount = 0; // Total spike count across all columns
    for (int column = 0; column < OUTPUT_SIZE; column++) { // OUTPUT_SIZE = classes (10)
        totalSpikeCount += spikeCounts[2][sampleIdx][column];
    }

    // Quantilizer stuff for false columns
    for (int colIndex = 0; colIndex < OUTPUT_SIZE; colIndex++) {
        if (colIndex != yTrue) {
            int falseColumnSpikeCount = spikeCounts[2][sampleIdx][colIndex];
            float allOtherColAvgSpikeCount = (totalSpikeCount - falseColumnSpikeCount) / (OUTPUT_SIZE - 1);

            float deltaFalse = ((float) falseColumnSpikeCount - allOtherColAvgSpikeCount);
            if (deltaFalse != negativeQuantVal)
            {
                if (deltaFalse >= negativeQuantVal) {
                    negativeQuantVal += negativeQuantValDeltaUp / negativeQuantValDenom; // Always add 150/900
                }
                else {
                    negativeQuantVal -= 1.0 / negativeQuantValDenom; // Always subtract 1/900
                }
            }
        }
    }
}

#ifdef GPU
cudaStream_t streams[NUM_EXAMPLES_TRAIN];
short int *gpu_from;
int *gpu_weight, *gpu_nsp, *gpu_list[3], *gpu_nspg, *gpu_togrp;
unsigned char *gpu_wd, *gpu_in[2];

__global__ void Kernel_Test(int nsize, int nsyn, int ncyc, short int *from,
                            int *weights, unsigned char *in, unsigned char *intensityThresholds, int *nsp,
                            int listNb, int *list, int sh, int nsh)
{
    int s, n, index, ana, tot, ns, cycle, off7sp, sp;

    n = blockIdx.x * blockDim.x + threadIdx.x;
    ns = n * nsyn;
    ana = n / nsize;
    n = n - nsize * ana;
    if (n < nsize && ana < OUTPUT_SIZE)
    {
        for (index = sh; index < sh + nsh; index++)
        {
            off7sp = list[index] * INPUT_SIZE;
            tot = 0;
            sp = 0;
            for (cycle = 0; cycle < ncyc; cycle++)
            {
                for (s = 0; s < nsyn; s++)
                    tot += (in[off7sp + from[ns + s]] >= intensityThresholds[cycle]) * weights[ns + s];

                if (tot > THRESHOLD)
                {
                    sp++;
                    tot = 0;
                }
                else
                    tot >>= 1;
            }
            if (sp)
                atomicAdd(nsp + index * OUTPUT_SIZE + ana, sp);
        }
    }
}

void init_gpu()
{
    int set, i;

    cudaSetDevice(0);

    cudaMalloc((void **)&gpu_wd, NUM_THRESHOLD_CYCLES * sizeof(unsigned char));
    for (set = 0; set < 2; set++)
        cudaMalloc((void **)&gpu_in[set], splitSize[set] * INPUT_SIZE * sizeof(unsigned char));
    for (set = 0; set < 3; set++)
        cudaMalloc((void **)&gpu_list[set], splitSizes[set] * sizeof(int));
    cudaMalloc((void **)&gpu_from, OUTPUT_SIZE * NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON * sizeof(short int));
    cudaMalloc((void **)&gpu_weight, OUTPUT_SIZE * NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON * sizeof(int));
    cudaMalloc((void **)&gpu_nsp, splitSize[0] * OUTPUT_SIZE * sizeof(int));
    cudaDeviceSynchronize();

    cudaMemcpy(gpu_wd, intensityThresholds, NUM_THRESHOLD_CYCLES * sizeof(unsigned char), cudaMemcpyHostToDevice);
    for (set = 0; set < 2; set++)
    {
        for (i = 0; i < splitSize[set]; i++)
            cudaMemcpy(gpu_in[set] + i * INPUT_SIZE, in[set][i], INPUT_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_list[set], indexOfSampleInSplit[set], splitSize[set] * sizeof(int), cudaMemcpyHostToDevice);
    }
    for (i = 0; i < 6000; i++)
        cudaStreamCreate(&(streams[i]));
    cudaDeviceSynchronize();
}

void test_gpu()
{
    int i, a, n, b, is, isnot, ok, sample, g, bssize, startat, set, l;
    static int tmpn[OUTPUT_SIZE * NUM_EXAMPLES_TRAIN];

    set = splitToDatasetMap[listNb];
    for (a = 0; a < OUTPUT_SIZE; a++)
    { // upload from & weights
        if (toreload[a] == 1)
        {
            for (n = 0; n < NUM_NEURONS_PER_COLUMN; n++)
            {
                bcopy(from[a][n], (void *)(tempReceptorIndices + (a * NUM_NEURONS_PER_COLUMN + n) * NUM_CONNECTIONS_PER_NEURON), NUM_CONNECTIONS_PER_NEURON * sizeof(short int));
                bcopy(weights[a][n], (void *)(tempWeights + (a * NUM_NEURONS_PER_COLUMN + n) * NUM_CONNECTIONS_PER_NEURON), NUM_CONNECTIONS_PER_NEURON * sizeof(int));
            }
            cudaMemcpy(gpu_from + a * NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON, tempReceptorIndices + a * NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON, NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON * sizeof(short int), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_weight + a * NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON, tempWeights + a * NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON, NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON * sizeof(int), cudaMemcpyHostToDevice);
            toreload[a] = 0;
        }
    }

    if (listNb == 2)
        cudaMemcpy(gpu_list[listNb], indexOfSampleInSplit[listNb], splitSizes[listNb] * sizeof(int), cudaMemcpyHostToDevice); // upload list
    cudaMemcpy(gpu_nsp, zero, OUTPUT_SIZE * splitSize[0] * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // run
    if (listNb == 2)
        bssize = splitSizes[listNb];
    else
        bssize = 100;

    for (startat = 0; startat < splitSizes[listNb]; startat += bssize)
        Kernel_Test<<<((OUTPUT_SIZE * NUM_NEURONS_PER_COLUMN + 31)) / 32, 32, 0, streams[startat]>>>(NUM_NEURONS_PER_COLUMN, NUM_CONNECTIONS_PER_NEURON, NUM_THRESHOLD_CYCLES, gpu_from, gpu_weight, gpu_in[set], gpu_wd, gpu_nsp,
                                                                              listNb, gpu_list[listNb], startat, bssize);
    cudaDeviceSynchronize();

    // download nsp
    cudaMemcpy(tmpn, gpu_nsp, splitSizes[listNb] * OUTPUT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // dispatch
    for (l = 0; l < splitSizes[listNb]; l++)
    {
        if (listNb == 2)
            bcopy(tmpn + l * OUTPUT_SIZE, spikeCounts[2][indexOfSampleInSplit[listNb][l]], OUTPUT_SIZE * sizeof(int));
        else
            bcopy(tmpn + l * OUTPUT_SIZE, spikeCounts[set][l], OUTPUT_SIZE * sizeof(int));
    }
}
#endif

void readset(int set, char *name_lbl, char *name_data) {
    int nn, n, fdl, fdi;

    splitSize[set] = splitSizes[set];
    fdl = open(name_lbl, O_RDONLY);
    lseek(fdl, 8, SEEK_SET);
    fdi = open(name_data, O_RDONLY);
    lseek(fdi, 16, SEEK_SET);

    for (n = 0; n < splitSizes[set]; n++) {
        read(fdl, rawLabels[set] + n, 1);
        read(fdi, rawData[set][n], INPUT_SIZE);
        if (set == 0) {
            notnuls[n] = 0;
            for (nn = 0; nn < INPUT_SIZE; nn++)
                if (rawData[set][n][nn])
                    notnuls[n]++;
        }
    }
    close(fdl);
    close(fdi);
}

void *test_mt_one(void *args) {
    struct ThreadArgs *arg;
    int threadNum;
    int column;
    int neuron;
    int cumulativeActivation;
    int cycle;
    int connection;
    int r;
    int l;
    int set;
    int sample;

    arg = (struct ThreadArgs *) args;
    threadNum = arg->threadNumber;

    while (1 == 1) {
        while (!mtact_test[arg->threadNumber])
            nanosleep(&multiThreadedWait, NULL);

        set = splitToDatasetMap[listNb];
        for (l = thrd_test_first[threadNum]; l < thrd_test_last[threadNum]; l++) {
            sample = indexOfSampleInSplit[listNb][l];
            for (column = 0; column < OUTPUT_SIZE; column++) {
                spikeCounts[listNb][sample][column] = 0;
                for (neuron = 0; neuron < NUM_NEURONS_PER_COLUMN; neuron++) {
                    cumulativeActivation = 0;
                    for (cycle = 0; cycle < NUM_THRESHOLD_CYCLES; cycle++) {
                        for (connection = 0; connection < NUM_CONNECTIONS_PER_NEURON; connection++) {
                            // Is this connection on for a specific intensity cycle threshold?
                            // DIFFERENCE: ADDING ALL WEIGHTS ACROSS CYCLES, COMPARE AGAINST THRESHOLD
                            //             This allows for adding the weight multiple times
                            if (rawData[set][sample][receptorIndices[column][neuron][connection]] >= intensityThresholds[cycle]) {
                                cumulativeActivation += weights[column][neuron][connection];
                            }
                        }

                        if (cumulativeActivation > THRESHOLD) {
                            spikeCounts[listNb][sample][column]++;
                            cumulativeActivation = 0;
                        } else
                            cumulativeActivation >>= 1;
                    }
                }
            }
        }

        mtact_test[threadNum] = 0;
    }
}

int testMultiThreaded() {
    int threadNum;
    int done;
    int set;
    int ok;
    int l;
    int is;
    int b;
    int isnot;
    int sample;
    int a;

    l = 0;
    a = splitSizes[listNb] / CPU_THREAD;
    b = splitSizes[listNb] - a * CPU_THREAD;
    for (threadNum = 0; threadNum < CPU_THREAD; threadNum++) {
        thrd_test_first[threadNum] = l;
        thrd_test_last[threadNum] = l + a;
        if (threadNum < b)
            thrd_test_last[threadNum]++;
        l = thrd_test_last[threadNum];
        mtact_test[threadNum] = 1;
    }

    done = 0;
    while (done != CPU_THREAD) {
        nanosleep(&multiThreadedWait, NULL);
        done = 0;
        for (threadNum = 0; threadNum < CPU_THREAD; threadNum++)
            done += 1 - mtact_test[threadNum];
    }
}

int test() {
    int set = splitToDatasetMap[listNb];;
    int numCorrectPredictions = 0;

#ifdef GPU
    test_gpu();
#else
    testMultiThreaded();
#endif
    for (int indexInSplit = 0; indexInSplit < splitSizes[listNb]; indexInSplit++) {
        int sampleIdx = indexOfSampleInSplit[listNb][indexInSplit];
        int yTrue = rawLabels[set][sampleIdx];
        int trueSpikeCount = spikeCounts[listNb][sampleIdx][yTrue];
        // Find the maximum spike count among all the false columns
        int maxFalseSpikeCount = -1;
        for (int column = 0; column < OUTPUT_SIZE; column++) {
            // If column is not the true column and larger than the previous max false column spike count
            if (column != yTrue && spikeCounts[listNb][sampleIdx][column] > maxFalseSpikeCount) {
                maxFalseSpikeCount = spikeCounts[listNb][sampleIdx][column];
            }
        }

        // See if we predicted the true column more than any of the false columns
        if (trueSpikeCount > maxFalseSpikeCount) {
            numCorrectPredictions++;
        }

        // If we're learning, error is maxFalseSpikeCount - trueSpikeCount
        if (listNb == 2) {
            err[sampleIdx] = maxFalseSpikeCount - trueSpikeCount;
            quant(sampleIdx);
        }
    }
    return numCorrectPredictions;
}

void test_mt_init() {
    int trhdNo;
    pthread_t thrds[CPU_THREAD];

    for (trhdNo = 0; trhdNo < CPU_THREAD; trhdNo++) {
        threadTestArgs[trhdNo].threadNumber = trhdNo;
        pthread_create(&thrds[trhdNo], NULL, test_mt_one,
                       (void *) &threadTestArgs[trhdNo]);
    }
}

void loss(int b) {
    int a, n, s, w;

    for (a = 0; a < OUTPUT_SIZE; a++) {
        for (n = 0; n < NUM_NEURONS_PER_COLUMN; n++) {
            for (s = 0; s < NUM_CONNECTIONS_PER_NEURON; s++) {
                w = weights[a][n][s];
                if (w) {
                    if (lossw && b % losswNb == 0) {
                        if (w > 0)
                            w -= lossw;
                        else
                            w += lossw;
                    }
                    if (abs(w) < MIN_WEIGHT) {
                        w = 0;
                        nbc[a]--;
                        nbcn[a][n]--;
                    }
                    weights[a][n][s] = w;
                }
            }
        }
        toreload[a] = 1;
    }
}

void connect(int nb, int sample, int a, int init, int g) {
    int p, p0, f, f0, n, s, i, ii, pn;
    // short int f0 ;
    static int ps[1000];

    if (nbc[a] == NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON)
        return;
    if (NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON - nbc[a] < nb)
        nb = NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON - nbc[a];

    for (i = 0; i < nb; i++) {
        ii = -1;
        while (i != ii) {
            ps[i] = rnd(NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON - nbc[a]) + 1;
            for (ii = 0; ii < i; ii++)
                if (ps[ii] == ps[i])
                    break;
        }
    }
    qsort(ps, nb, sizeof(int), compinc);

    n = 0;
    s = 0;
    p0 = 0;

    for (i = 0; i < nb; i++) {
        p = ps[i];
        for (; n < NUM_NEURONS_PER_COLUMN && p0 != p; n += (p0 != p)) {
            if (p0 + NUM_CONNECTIONS_PER_NEURON - nbcn[a][n] < p)
                p0 += NUM_CONNECTIONS_PER_NEURON - nbcn[a][n];
            else
                for (s *= (pn == n); s < NUM_CONNECTIONS_PER_NEURON && p0 != p; s += (p0 != p))
                    if (!weights[a][n][s])
                        p0++;
        }

        f = rnd(notnuls[sample]) + 1;
        for (f0 = 0; f0 < INPUT_SIZE && f; f0 += (f != 0))
            if (rawData[0][sample][f0])
                f--;

        receptorIndices[a][n][s] = f0;
        weights[a][n][s] = init;
        nbc[a]++;
        nbcn[a][n]++;
        pn = n;
    }
}

void *learn_mt_one(void *args) {
    struct ThreadArgs *arg;
    int to, start, trhdNo;
    int a, n, c, s, sample, adj;
    int tot, cnta, mask;

    arg = (struct ThreadArgs *) args;
    while (1 == 1) {
        while (!mtact_learn[arg->threadNumber])
            nanosleep(&multiThreadedWait, NULL);

        for (n = arg->startIndex; n < arg->endIndex; n++) {
            tot = 0;
            cnta = 0;
            for (c = 0; c < NUM_THRESHOLD_CYCLES; c++) {
                mask = 1;
                for (s = 0; s < NUM_CONNECTIONS_PER_NEURON; s++) {
                    if (weights[mta][n][s] && rawData[0][mtsample][receptorIndices[mta][n][s]] >= intensityThresholds[c]) {
                        tot += weights[mta][n][s];
                        cnta |= mask;
                    }
                    mask <<= 1;
                }

                if (tot > THRESHOLD_MIN) {
                    if (tot < THRESHOLD_MAX) {
                        mask = 1;
                        for (s = 0; s < NUM_CONNECTIONS_PER_NEURON; s++) {
                            if ((cnta & mask) != 0 && weights[mta][n][s]) {
                                if (abs(weights[mta][n][s] + mtadj) < THRESHOLD) {
                                    weights[mta][n][s] += mtadj;
                                    if (abs(weights[mta][n][s]) < MIN_WEIGHT) {
                                        weights[mta][n][s] = 0;
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
        mtact_learn[arg->threadNumber] = 0;
    }
}

void learn(int sample, int a, int adj) {
    int trhdNo, done, n;

    mta = a;
    mtadj = adj;
    mtsample = sample;
    for (trhdNo = 0; trhdNo < CPU_THREAD; trhdNo++)
        mtact_learn[trhdNo] = 1;

    done = 0;
    while (done != CPU_THREAD) {
        nanosleep(&multiThreadedWait, NULL);
        done = 0;
        for (trhdNo = 0; trhdNo < CPU_THREAD; trhdNo++)
            done += 1 - mtact_learn[trhdNo];
    }
    nbc[mta] = 0;
    for (n = 0; n < NUM_NEURONS_PER_COLUMN; n++)
        nbc[mta] += nbcn[mta][n];
}

void learn_mt_init() {
    int start, trhdNo, sssize;
    pthread_t thrds[CPU_THREAD];

    sssize = (NUM_NEURONS_PER_COLUMN + CPU_THREAD - 1) / CPU_THREAD;
    start = 0;
    trhdNo = 0;

    while (start < NUM_NEURONS_PER_COLUMN) {
        threadArgs[trhdNo].startIndex = start;
        threadArgs[trhdNo].endIndex = start + sssize;
        if (start + sssize >= NUM_NEURONS_PER_COLUMN)
            threadArgs[trhdNo].endIndex = NUM_NEURONS_PER_COLUMN;
        threadArgs[trhdNo].threadNumber = trhdNo;
        pthread_create(&thrds[trhdNo], NULL, learn_mt_one,
                       (void *) &threadArgs[trhdNo]);
        start += sssize;
        trhdNo++;
    }
}

void batch() {
//    int b = 0;
    int bu = 0;
    int column;
    int isnot;
    int g;
    int tot;
    int led;
    int prevStep = 0;
    static int selectionMask[NUM_EXAMPLES_TRAIN];
    float d;

    numBatchesTested = 0;
    while (1 == 1) {                                          // batch selection
        bcopy(zero, selectionMask, NUM_EXAMPLES_TRAIN * sizeof(int)); // no duplicate (???)
        for (int sampleIdx = 0; sampleIdx < splitSizes[2]; sampleIdx++) { // Go through 1000
            // The value we're placing will be used as a random index into the data
            indexOfSampleInSplit[2][sampleIdx] = rnd(NUM_EXAMPLES_TRAIN);
            while (selectionMask[indexOfSampleInSplit[2][sampleIdx]]) { // While we haven't already selected this sample
                indexOfSampleInSplit[2][sampleIdx] = rnd(NUM_EXAMPLES_TRAIN); // Store the selected index
            }
            selectionMask[indexOfSampleInSplit[2][sampleIdx]] = 1; // Mark that we've selected it
        }

        listNb = 2;
        test();
        numBatchesTested++;

        for (int indexInBatch = 0; indexInBatch < splitSizes[2]; indexInBatch++) {
            int sampleIdx = indexOfSampleInSplit[2][indexInBatch];
            int yTrue = rawLabels[0][sampleIdx];

            if ((float) (err[sampleIdx]) >= positiveQuantVal || numBatchesTested > b /*lim.@start*/) {
                connect(nbNew, sampleIdx, yTrue, THRESHOLD / divNew, g);
                learn(sampleIdx, yTrue, adjp);
                toreload[yTrue] = 1;
                nblearn++;
                b++;
                if (lossw && nblearn % losswNb == 0) {
                    loss(b);
                }
            }

            if (bu > 3 * b / 2)
                continue; // limiter for faster startup.

            tot = 0;
            for (column = 0; column < OUTPUT_SIZE; column++)
                tot += spikeCounts[2][sampleIdx][column];
            for (isnot = 0; isnot < OUTPUT_SIZE; isnot++) {
                if (isnot != yTrue) {
                    d = (float) (spikeCounts[2][sampleIdx][isnot] - (tot - spikeCounts[2][sampleIdx][isnot]) / 9);
                    if (negativeQuantValDeltaUp != 0.0 && d >= negativeQuantVal) {
                        connect(nbNew, sampleIdx, isnot, -THRESHOLD / divNew, g);
                        learn(sampleIdx, isnot, adjn);
                        toreload[isnot] = 1;
                        bu++;
                    }
                }
            }
            if (/*b/briefStep >prevStep*/ b % briefStep == 0 && b != prevStep) {
                printf("testing %d: ", nblearn);
                listNb = 0;
                printf("0: %2.3f  ", (float) test() / 600.0);
                listNb = 1;
                printf("1: %2.2f\n", (float) test() / 100.0);
                prevStep = b /*/briefStep*/;
            }
        }
    }
}

int main() {
    srand(1000);          // time(NULL) for random
    setbuf(stdout, NULL); // unbuffering stdout
    multiThreadedWait.tv_sec = 0;
    multiThreadedWait.tv_nsec = 10;
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
    splitSizes[2] = 200;
    positiveQuantVal = 0.0;
    positiveQuantValDenom = 100;
    positiveQuantValDeltaUp = 19.0;
    negativeQuantVal = 0.0;
    negativeQuantValDenom = 900;
    negativeQuantValDeltaUp = 150.0; // 98.9 : quantDivP=100
    adjp = 500;
    adjn = -500; // 98.9 : 1000 / -1000
    divNew = 10;
    nbNew = 10; // 98.9 : 10 / 20
    lossw = 20;
    losswNb = 100; // 98.9 : 10 / 100

    batch();
}