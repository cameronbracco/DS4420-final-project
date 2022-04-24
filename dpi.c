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
#define NUM_EXAMPLES_BATCH 1000
// size. NUM_CONNECTIONS_PER_NEURON: For 98.9%: 7920. For 98.65% : 792
#define NUM_CONNECTIONS_PER_NEURON 10 // number of connections per neuron.
#define NUM_NEURONS_PER_COLUMN 792 // multiple of CPU_THREAD. Number of neurons per column
// params
#define THRESHOLD 1000000
#define THRESHOLD_MIN 300000
#define THRESHOLD_MAX 1200000
#define MIN_WEIGHT 97000

#define NUM_FILTER_CYCLES 4 // Number of pixel intensity thresholds
unsigned char intensityFilters[NUM_FILTER_CYCLES] = {192, 128, 64, 1};

int indexOfSampleInSplit[3][NUM_EXAMPLES_TRAIN];
int splitSizes[3] = {NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST, NUM_EXAMPLES_BATCH};
int splitToDatasetMap[3] = {0, 1, 0};
int splitSize[2];
unsigned char rawLabels[2][NUM_EXAMPLES_TRAIN]; // Maybe for train/test?
int rawData[2][NUM_EXAMPLES_TRAIN][INPUT_SIZE]; // Don't know why 2 "channels"
int notnuls[NUM_EXAMPLES_TRAIN];
int numConnectionsPerColumn[OUTPUT_SIZE]; // Total number of connections per column (used for growth)
int numConnectionsPerNeuronPerColumn[OUTPUT_SIZE][NUM_NEURONS_PER_COLUMN]; // Yeah...
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
int adjp;
int adjn;
int lossw;
int losswNb;
int maxConnectionsToGrowPerBatch; // Idk why this matters
int divNew;
int briefStep;
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
int multiThreadedMaskLearn[CPU_THREAD];
int multiThreadedMaskTest[CPU_THREAD];
struct ThreadArgs threadArgs[CPU_THREAD];
struct ThreadArgs threadTestArgs[CPU_THREAD];
int threadStartIndices[CPU_THREAD]; // Where thread should start processing
int threadEndIndices[CPU_THREAD]; // Where thread should stop processing
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

    cudaMalloc((void **)&gpu_wd, NUM_FILTER_CYCLES * sizeof(unsigned char));
    for (set = 0; set < 2; set++)
        cudaMalloc((void **)&gpu_in[set], splitSize[set] * INPUT_SIZE * sizeof(unsigned char));
    for (set = 0; set < 3; set++)
        cudaMalloc((void **)&gpu_list[set], splitSizes[set] * sizeof(int));
    cudaMalloc((void **)&gpu_from, OUTPUT_SIZE * NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON * sizeof(short int));
    cudaMalloc((void **)&gpu_weight, OUTPUT_SIZE * NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON * sizeof(int));
    cudaMalloc((void **)&gpu_nsp, splitSize[0] * OUTPUT_SIZE * sizeof(int));
    cudaDeviceSynchronize();

    cudaMemcpy(gpu_wd, intensityFilters, NUM_FILTER_CYCLES * sizeof(unsigned char), cudaMemcpyHostToDevice);
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
        Kernel_Test<<<((OUTPUT_SIZE * NUM_NEURONS_PER_COLUMN + 31)) / 32, 32, 0, streams[startat]>>>(NUM_NEURONS_PER_COLUMN, NUM_CONNECTIONS_PER_NEURON, NUM_FILTER_CYCLES, gpu_from, gpu_weight, gpu_in[set], gpu_wd, gpu_nsp,
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

// Each thread looks at a subset of data and measures spike counts
void *testMultiThreaded(void *args) {
    struct ThreadArgs *arg = (struct ThreadArgs *) args;
    int threadNum = arg->threadNumber;;
    int column;
    int neuron;
    int cycle;
    int connection;

    while (1 == 1) {
        // Thread waits until it is activated
        while (!multiThreadedMaskTest[arg->threadNumber]) {
            nanosleep(&multiThreadedWait, NULL);
        }

        int set = splitToDatasetMap[listNb];
        for (int indexInSplit = threadStartIndices[threadNum]; indexInSplit < threadEndIndices[threadNum]; indexInSplit++) {
            int sampleIdx = indexOfSampleInSplit[listNb][indexInSplit];
            for (column = 0; column < OUTPUT_SIZE; column++) {
                spikeCounts[listNb][sampleIdx][column] = 0;
                for (neuron = 0; neuron < NUM_NEURONS_PER_COLUMN; neuron++) {
                    int cumulativeActivation = 0;
                    for (cycle = 0; cycle < NUM_FILTER_CYCLES; cycle++) {
                        for (connection = 0; connection < NUM_CONNECTIONS_PER_NEURON; connection++) {
                            // Is this connection on for a specific intensity cycle threshold?
                            // DIFFERENCE: ADDING ALL WEIGHTS ACROSS CYCLES, COMPARE AGAINST THRESHOLD
                            //             This allows for adding the weight multiple times
                            if (rawData[set][sampleIdx][receptorIndices[column][neuron][connection]] >= intensityFilters[cycle]) {
                                cumulativeActivation += weights[column][neuron][connection];
                            }
                        }

                        // Theoretically this means you could spike a neuron multiple times per sample
                        // DIFFERENCE: Spiking multiple times (max 1 per cycle) and bringing forward activations across cycles
                        if (cumulativeActivation > THRESHOLD) {
                            spikeCounts[listNb][sampleIdx][column]++;
                            cumulativeActivation = 0;
                        } else {
                            // We bring forward activations, but divide by 2 since this filter is half as
                            // intense as the next
                            cumulativeActivation >>= 1;
                        }
                    }
                }
            }
        }

        // Let the thread manager know that we're done processing
        multiThreadedMaskTest[threadNum] = 0;
    }
}

// Manages the threads for testing
int testMultiThreadedManager() {
    int threadNum;

    int sampleIndex = 0;
    int numSamplesPerCPU = splitSizes[listNb] / CPU_THREAD;
    // There might be some left over...
    int remainingSamples = splitSizes[listNb] - (numSamplesPerCPU * CPU_THREAD);
    for (threadNum = 0; threadNum < CPU_THREAD; threadNum++) {
        threadStartIndices[threadNum] = sampleIndex;
        threadEndIndices[threadNum] = sampleIndex + numSamplesPerCPU;
        // Adds 1 to each of the threads while there are some left
        if (threadNum < remainingSamples) {
            threadEndIndices[threadNum]++;
        }
        sampleIndex = threadEndIndices[threadNum];
        multiThreadedMaskTest[threadNum] = 1;
    }

    int numThreadsFinished = 0;
    while (numThreadsFinished != CPU_THREAD) {
        nanosleep(&multiThreadedWait, NULL);
        numThreadsFinished = 0;
        for (threadNum = 0; threadNum < CPU_THREAD; threadNum++) {
            numThreadsFinished += 1 - multiThreadedMaskTest[threadNum];
        }
    }
}

int test() {
    int set = splitToDatasetMap[listNb];;
    int numCorrectPredictions = 0;

#ifdef GPU
    test_gpu();
#else
    testMultiThreadedManager();
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
        pthread_create(&thrds[trhdNo], NULL, testMultiThreaded,
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
                        numConnectionsPerColumn[a]--;
                        numConnectionsPerNeuronPerColumn[a][n]--;
                    }
                    weights[a][n][s] = w;
                }
            }
        }
        toreload[a] = 1;
    }
}

void connect(int maxConnectionsToGrow, int sampleIdx, int column, int initialWeight) {
    int p;
    int p0;
    int f;
    int f0;
    int connection;
    int ii;
    int pn;
    // short int f0 ;
    static int ps[NUM_EXAMPLES_BATCH];

    // If we don't need to grow any new connections, return early
    if (numConnectionsPerColumn[column] == NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON) {
        return;
    }

    // Limit how many we should grow to how much we have capacity for
    if ((NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON) - numConnectionsPerColumn[column] < maxConnectionsToGrow) {
        maxConnectionsToGrow = (NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON) - numConnectionsPerColumn[column];
    }

    for (int i = 0; i < maxConnectionsToGrow; i++) {
        ii = -1;
        while (i != ii) {
            // WHAT?
            ps[i] = rnd(maxConnectionsToGrow) + 1;
            for (ii = 0; ii < i; ii++) {
                if (ps[ii] == ps[i]) {
                    break;
                }
            }
        }
    }
    qsort(ps, maxConnectionsToGrow, sizeof(int), compinc);

    int neuron = 0;
    connection = 0;
    p0 = 0;

    for (int i = 0; i < maxConnectionsToGrow; i++) {
        p = ps[i];
        for (; neuron < NUM_NEURONS_PER_COLUMN && p0 != p; neuron += (p0 != p)) {
            if (p0 + NUM_CONNECTIONS_PER_NEURON - numConnectionsPerNeuronPerColumn[column][neuron] < p) {
                p0 += NUM_CONNECTIONS_PER_NEURON - numConnectionsPerNeuronPerColumn[column][neuron];
            }
            else {
                for (connection *= (pn == neuron); connection < NUM_CONNECTIONS_PER_NEURON && p0 != p; connection += (p0 != p)) {
                    if (!weights[column][neuron][connection]) {
                        p0++;
                    }
                }
            }
        }

        f = rnd(notnuls[sampleIdx]) + 1;
        for (f0 = 0; f0 < INPUT_SIZE && f; f0 += (f != 0)) {
            if (rawData[0][sampleIdx][f0]) {
                f--;
            }
        }

        receptorIndices[column][neuron][connection] = f0; // Index in INPUT_SIZE to connect to
        weights[column][neuron][connection] = initialWeight;
        numConnectionsPerColumn[column]++;
        numConnectionsPerNeuronPerColumn[column][neuron]++;
        pn = neuron;
    }
}

void *learn_mt_one(void *args) {
    struct ThreadArgs *arg;
    int to, start, trhdNo;
    int a, n, c, s, sample, adj;
    int tot, cnta, mask;

    arg = (struct ThreadArgs *) args;
    while (1 == 1) {
        while (!multiThreadedMaskLearn[arg->threadNumber])
            nanosleep(&multiThreadedWait, NULL);

        for (n = arg->startIndex; n < arg->endIndex; n++) {
            tot = 0;
            cnta = 0;
            for (c = 0; c < NUM_FILTER_CYCLES; c++) {
                mask = 1;
                for (s = 0; s < NUM_CONNECTIONS_PER_NEURON; s++) {
                    if (weights[mta][n][s] && rawData[0][mtsample][receptorIndices[mta][n][s]] >= intensityFilters[c]) {
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
                                        numConnectionsPerNeuronPerColumn[mta][n]--;
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
        multiThreadedMaskLearn[arg->threadNumber] = 0;
    }
}

void learn(int sample, int a, int adj) {
    int trhdNo, done, n;

    mta = a;
    mtadj = adj;
    mtsample = sample;
    for (trhdNo = 0; trhdNo < CPU_THREAD; trhdNo++)
        multiThreadedMaskLearn[trhdNo] = 1;

    done = 0;
    while (done != CPU_THREAD) {
        nanosleep(&multiThreadedWait, NULL);
        done = 0;
        for (trhdNo = 0; trhdNo < CPU_THREAD; trhdNo++)
            done += 1 - multiThreadedMaskLearn[trhdNo];
    }
    numConnectionsPerColumn[mta] = 0;
    for (n = 0; n < NUM_NEURONS_PER_COLUMN; n++)
        numConnectionsPerColumn[mta] += numConnectionsPerNeuronPerColumn[mta][n];
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
    int numSamplesLearnedFrom = 0;
    int bu = 0;
    int column;
    int isnot;
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
        test(); // NOTE: this function will update the quantilizers
        numBatchesTested++;

        for (int indexInBatch = 0; indexInBatch < splitSizes[2]; indexInBatch++) {
            int sampleIdx = indexOfSampleInSplit[2][indexInBatch];
            int yTrue = rawLabels[0][sampleIdx];

            // DIFFERENCE: We always should learn at least once?
            if ((float) (err[sampleIdx]) >= positiveQuantVal || numBatchesTested > numSamplesLearnedFrom /*lim.@start*/) {
                // First step of learning is connecting
                connect(maxConnectionsToGrowPerBatch, sampleIdx, yTrue, THRESHOLD / divNew);
                learn(sampleIdx, yTrue, adjp);
                toreload[yTrue] = 1;
                nblearn++;
                numSamplesLearnedFrom++;
                if (lossw && nblearn % losswNb == 0) {
                    loss(numSamplesLearnedFrom);
                }
            }

            if (bu > 3 * numSamplesLearnedFrom / 2)
                continue; // limiter for faster startup.

            tot = 0;
            for (column = 0; column < OUTPUT_SIZE; column++)
                tot += spikeCounts[2][sampleIdx][column];
            for (isnot = 0; isnot < OUTPUT_SIZE; isnot++) {
                if (isnot != yTrue) {
                    d = (float) (spikeCounts[2][sampleIdx][isnot] - (tot - spikeCounts[2][sampleIdx][isnot]) / 9);
                    if (negativeQuantValDeltaUp != 0.0 && d >= negativeQuantVal) {
                        // DIFFERENCE: Negative initial weights
                        connect(maxConnectionsToGrowPerBatch, sampleIdx, isnot, -THRESHOLD / divNew);
                        learn(sampleIdx, isnot, adjn);
                        toreload[isnot] = 1;
                        bu++;
                    }
                }
            }
            if (/*b/briefStep >prevStep*/ numSamplesLearnedFrom % briefStep == 0 && numSamplesLearnedFrom != prevStep) {
                printf("testing %d: ", nblearn);
                listNb = 0;
                printf("0: %2.3f  ", (float) test() / 600.0);
                listNb = 1;
                printf("1: %2.2f\n", (float) test() / 100.0);
                prevStep = numSamplesLearnedFrom /*/briefStep*/;
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
    maxConnectionsToGrowPerBatch = 10; // 98.9 : 10 / 20
    lossw = 20;
    losswNb = 100; // 98.9 : 10 / 100

    batch();
}