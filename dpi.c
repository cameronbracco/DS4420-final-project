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
#define MAX_CONNECTION_WEIGHT 1000000
#define THRESHOLD_MIN 300000
#define THRESHOLD_MAX 1200000
#define MIN_CONNECTION_WEIGHT 97000

#define NUM_FILTER_CYCLES 4 // Number of pixel intensity thresholds
unsigned char intensityFilters[NUM_FILTER_CYCLES] = {192, 128, 64, 1};

int indexOfSampleInSplit[3][NUM_EXAMPLES_TRAIN];
int splitSizes[3] = {NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST, NUM_EXAMPLES_BATCH};
int splitToDatasetMap[3] = {0, 1, 0};
int splitSize[2];
unsigned char rawLabels[2][NUM_EXAMPLES_TRAIN]; // Maybe for train/test?
int rawData[2][NUM_EXAMPLES_TRAIN][INPUT_SIZE]; // Don't know why 2 "channels"
int nonZeroPixelsCount[NUM_EXAMPLES_TRAIN]; // Count of non-zero pixels for a specific sample in the training set
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
int splitIndex;
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
int multiThreadedAmountToAdjust; // Amount to adjust that all threads use during lead
int multiThreadedSampleIndex; // Sample that all the threads are processing during learn
int multiThreadedColumn; // Column that all the thread are processing during learn

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

/**
 * Perform both positive and negative quantilizer updates when learning from a specific sample
 * @param sampleIdx
 */
void quant(int sampleIdx) {
    int yTrue = rawLabels[0][sampleIdx];

    // Positive quantilizer update
    // err is Delta True
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

    // Sum the total number of spikes across all columns
    int totalSpikeCount = 0; // Total spike count across all columns
    for (int column = 0; column < OUTPUT_SIZE; column++) { // OUTPUT_SIZE = classes (10)
        totalSpikeCount += spikeCounts[splitIndex][sampleIdx][column];
    }

    // Negative quantitlizer update
    for (int colIndex = 0; colIndex < OUTPUT_SIZE; colIndex++) {
        if (colIndex != yTrue) {
            int falseColumnSpikeCount = spikeCounts[splitIndex][sampleIdx][colIndex];
            float allOtherColAvgSpikeCount = (float) (totalSpikeCount - falseColumnSpikeCount) / (OUTPUT_SIZE - 1);


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

                if (tot > MAX_CONNECTION_WEIGHT)
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

    set = splitToDatasetMap[splitIndex];
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

    if (splitIndex == 2)
        cudaMemcpy(gpu_list[splitIndex], indexOfSampleInSplit[splitIndex], splitSizes[splitIndex] * sizeof(int), cudaMemcpyHostToDevice); // upload list
    cudaMemcpy(gpu_nsp, zero, OUTPUT_SIZE * splitSize[0] * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // run
    if (splitIndex == 2)
        bssize = splitSizes[splitIndex];
    else
        bssize = 100;

    for (startat = 0; startat < splitSizes[splitIndex]; startat += bssize)
        Kernel_Test<<<((OUTPUT_SIZE * NUM_NEURONS_PER_COLUMN + 31)) / 32, 32, 0, streams[startat]>>>(NUM_NEURONS_PER_COLUMN, NUM_CONNECTIONS_PER_NEURON, NUM_FILTER_CYCLES, gpu_from, gpu_weight, gpu_in[set], gpu_wd, gpu_nsp,
                                                                              splitIndex, gpu_list[splitIndex], startat, bssize);
    cudaDeviceSynchronize();

    // download nsp
    cudaMemcpy(tmpn, gpu_nsp, splitSizes[splitIndex] * OUTPUT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // dispatch
    for (l = 0; l < splitSizes[splitIndex]; l++)
    {
        if (splitIndex == 2)
            bcopy(tmpn + l * OUTPUT_SIZE, spikeCounts[2][indexOfSampleInSplit[splitIndex][l]], OUTPUT_SIZE * sizeof(int));
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
            nonZeroPixelsCount[n] = 0;
            for (nn = 0; nn < INPUT_SIZE; nn++)
                if (rawData[set][n][nn])
                    nonZeroPixelsCount[n]++;
        }
    }
    close(fdl);
    close(fdi);
}

/**
 * Calculates neuron activation and spike count for all samples in a batch
 * Note this is multithreaded so each thread only deals with a subset of the batch
 * @param args - Threading arguments
 * @return
 */
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

        int set = splitToDatasetMap[splitIndex];
        for (int indexInSplit = threadStartIndices[threadNum]; indexInSplit < threadEndIndices[threadNum]; indexInSplit++) {
            int sampleIdx = indexOfSampleInSplit[splitIndex][indexInSplit];
            for (column = 0; column < OUTPUT_SIZE; column++) {
                spikeCounts[splitIndex][sampleIdx][column] = 0;
                for (neuron = 0; neuron < NUM_NEURONS_PER_COLUMN; neuron++) {
                    int cumulativeActivation = 0;
                    for (cycle = 0; cycle < NUM_FILTER_CYCLES; cycle++) {
                        for (connection = 0; connection < NUM_CONNECTIONS_PER_NEURON; connection++) {
                            // Is this connection on for a specific intensity cycle threshold?
                            // DIFFERENCE: ADDING ALL WEIGHTS ACROSS CYCLES, COMPARE AGAINST MAX_CONNECTION_WEIGHT
                            //             This allows for adding the weight multiple times
                            if (rawData[set][sampleIdx][receptorIndices[column][neuron][connection]] >= intensityFilters[cycle]) {
                                cumulativeActivation += weights[column][neuron][connection];
                            }
                        }

                        // Theoretically this means you could spike a neuron multiple times per sample
                        // DIFFERENCE: Spiking multiple times (max 1 per cycle) and bringing forward activations across cycles
                        if (cumulativeActivation > MAX_CONNECTION_WEIGHT) {
                            spikeCounts[splitIndex][sampleIdx][column]++;
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

/**
 * Manages the multithreaded processing of the activation/spikecounts
 * Turns on the threads and then waits for them to finish
 * @return
 */
int testMultiThreadedManager() {
    int threadNum;

    int sampleIndex = 0;
    int numSamplesPerCPU = splitSizes[splitIndex] / CPU_THREAD;
    // There might be some left over...
    int remainingSamples = splitSizes[splitIndex] - (numSamplesPerCPU * CPU_THREAD);
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
    int set = splitToDatasetMap[splitIndex];;
    int numCorrectPredictions = 0;

#ifdef GPU
    test_gpu();
#else
    testMultiThreadedManager();
#endif

    // At this point we have the spike counts for all samples in the batch
    // Now we need to evaluate how well we did
    for (int indexInSplit = 0; indexInSplit < splitSizes[splitIndex]; indexInSplit++) {
        int sampleIdx = indexOfSampleInSplit[splitIndex][indexInSplit];
        int yTrue = rawLabels[set][sampleIdx];
        int trueSpikeCount = spikeCounts[splitIndex][sampleIdx][yTrue];
        // Find the maximum spike count among all the false columns
        int maxFalseSpikeCount = -1;
        for (int column = 0; column < OUTPUT_SIZE; column++) {
            // If column is not the true column and larger than the previous max false column spike count
            if (column != yTrue && spikeCounts[splitIndex][sampleIdx][column] > maxFalseSpikeCount) {
                maxFalseSpikeCount = spikeCounts[splitIndex][sampleIdx][column];
            }
        }

        // See if we predicted the true column more than any of the false columns
        if (trueSpikeCount > maxFalseSpikeCount) {
            numCorrectPredictions++;
        }

        // If we're learning, error is maxFalseSpikeCount - trueSpikeCount
        if (splitIndex == 2) {
            // Delta True
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
                    if (abs(w) < MIN_CONNECTION_WEIGHT) {
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

/**
 * Grow new connections, if necessary, from a specific
 * @param maxConnectionsToGrow
 * @param sampleIdx
 * @param column
 * @param initialWeight
 */
void connect(int maxConnectionsToGrow, int sampleIdx, int column, int initialWeight) {
    int f;
    int previousNeuron;

    static int ps[NUM_EXAMPLES_BATCH]; // This used to be 1000, which made no sense anyways....

    // If we don't need to grow any new connections, return early
    if (numConnectionsPerColumn[column] == NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON) {
        return;
    }

    // Limit how many we should grow to how much we have capacity for
    // NOTE: We're saying how many connections need to be grown at the column level, not per neuron
    if ((NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON) - numConnectionsPerColumn[column] < maxConnectionsToGrow) {
        maxConnectionsToGrow = (NUM_NEURONS_PER_COLUMN * NUM_CONNECTIONS_PER_NEURON) - numConnectionsPerColumn[column];
    }

    for (int i = 0; i < maxConnectionsToGrow; i++) {
        int ii = -1;
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
    int connection = 0;
    int p0 = 0;

    for (int i = 0; i < maxConnectionsToGrow; i++) {

        // Complex way of selecting a neuron that has an open connection (and select which connection it is)
        for (int p = ps[i]; neuron < NUM_NEURONS_PER_COLUMN && p0 != p; neuron += (p0 != p)) {
            if (p0 + NUM_CONNECTIONS_PER_NEURON - numConnectionsPerNeuronPerColumn[column][neuron] < p) {
                p0 += NUM_CONNECTIONS_PER_NEURON - numConnectionsPerNeuronPerColumn[column][neuron];
            }
            else {
                for (connection *= (previousNeuron == neuron); connection < NUM_CONNECTIONS_PER_NEURON && p0 != p; connection += (p0 != p)) {
                    if (!weights[column][neuron][connection]) {
                        p0++;
                    }
                }
            }
        }

        // This is a very complex way of choosing a random receptor index that is on in the pixel (based on the raw data)
        f = rnd(nonZeroPixelsCount[sampleIdx]) + 1;
        int receptorIndexToGrowTo = 0;
        for (receptorIndexToGrowTo = 0; receptorIndexToGrowTo < INPUT_SIZE && f > 0; receptorIndexToGrowTo += (f != 0)) {
            if (rawData[0][sampleIdx][receptorIndexToGrowTo]) {
                f--;
            }
        }

        // Once we've selected a neuron and a connection,
        receptorIndices[column][neuron][connection] = receptorIndexToGrowTo; // Index in INPUT_SIZE to connect to
        weights[column][neuron][connection] = initialWeight;
        numConnectionsPerColumn[column]++;
        numConnectionsPerNeuronPerColumn[column][neuron]++;
        previousNeuron = neuron;
    }
}

void *learnMultiThreaded(void *args) {
    struct ThreadArgs *arg = (struct ThreadArgs *) args;

    while (1 == 1) {
        // Sleep util this thread gets activated
        while (!multiThreadedMaskLearn[arg->threadNumber]) {
            nanosleep(&multiThreadedWait, NULL);
        }

        // Each thread only looks at a subset of the data
        // Note that here we're splitting up the neurons to learn across the threads (vs the samples in the test code)
        for (int neuron = arg->startIndex; neuron < arg->endIndex; neuron++) {
            int sumWeightsActiveConnections = 0; // Total sum of weights of all activated connections across all cycles (also does discounting...)
            int connectionsToUpdateMask = 0; // Which connections we want to update
            for (int cycle = 0; cycle < NUM_FILTER_CYCLES; cycle++) {
                int mask = 1;
                for (int connection = 0; connection < NUM_CONNECTIONS_PER_NEURON; connection++) {
                    // Check two conditions for each connection
                    // 1. If the connection has weight already
                    // 2. The receptor is on for this cycle (it passes the intensity filter)
                    if (weights[multiThreadedColumn][neuron][connection] && rawData[0][multiThreadedSampleIndex][receptorIndices[multiThreadedColumn][neuron][connection]] >= intensityFilters[cycle]) {
                        sumWeightsActiveConnections += weights[multiThreadedColumn][neuron][connection];
                        connectionsToUpdateMask |= mask;
                    }
                    mask <<= 1;
                }

                // ??? We need to have some good connections now, but not too many?
                if (sumWeightsActiveConnections > THRESHOLD_MIN) {
                    if (sumWeightsActiveConnections < THRESHOLD_MAX) {
                        mask = 1;
                        for (int connection = 0; connection < NUM_CONNECTIONS_PER_NEURON; connection++) {
                            // If the connection was active during this cycle (and had non-zero weight)
                            if ((connectionsToUpdateMask & mask) != 0 && weights[multiThreadedColumn][neuron][connection]) {
                                // Then if we won't go above the threshold when we adjust (in absolute)
                                if (abs(weights[multiThreadedColumn][neuron][connection] + multiThreadedAmountToAdjust) < MAX_CONNECTION_WEIGHT) {
                                    // Adjust the weight for this connection by the specified amount
                                    weights[multiThreadedColumn][neuron][connection] += multiThreadedAmountToAdjust;
                                    // If the absolute value of the connection is below a minimum weight, prune it
                                    if (abs(weights[multiThreadedColumn][neuron][connection]) < MIN_CONNECTION_WEIGHT) {
                                        weights[multiThreadedColumn][neuron][connection] = 0;
                                        numConnectionsPerNeuronPerColumn[multiThreadedColumn][neuron]--;
                                    }
                                }
                            }
                            mask <<= 1;
                        }
                    }
                    sumWeightsActiveConnections = 0;
                    connectionsToUpdateMask = 0;
                }
                // Discounting between cycles by dividing by 2
                sumWeightsActiveConnections >>= 1;
            }
        }
        multiThreadedMaskLearn[arg->threadNumber] = 0;
    }
}


void learn(int sampleIndex, int column, int amountToAdjust) {
    int done;
    int n;

    // Setting globals for the threads to use
    multiThreadedColumn = column;
    multiThreadedAmountToAdjust = amountToAdjust;
    multiThreadedSampleIndex = sampleIndex;
    // Activate all the learning-focused threads
    for (int threadNum = 0; threadNum < CPU_THREAD; threadNum++) {
        multiThreadedMaskLearn[threadNum] = 1;
    }

    done = 0;
    while (done != CPU_THREAD) {
        nanosleep(&multiThreadedWait, NULL);
        done = 0;
        for (int threadNum = 0; threadNum < CPU_THREAD; threadNum++) {
            done += 1 - multiThreadedMaskLearn[threadNum];
        }
    }
    numConnectionsPerColumn[multiThreadedColumn] = 0;
    for (n = 0; n < NUM_NEURONS_PER_COLUMN; n++) {
        numConnectionsPerColumn[multiThreadedColumn] += numConnectionsPerNeuronPerColumn[multiThreadedColumn][n];
    }
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
        pthread_create(&thrds[trhdNo], NULL, learnMultiThreaded,
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

        splitIndex = 2;
        test(); // NOTE: this function will update the quantilizers
        numBatchesTested++;


        for (int indexInBatch = 0; indexInBatch < splitSizes[2]; indexInBatch++) {
            int sampleIdx = indexOfSampleInSplit[2][indexInBatch];
            int yTrue = rawLabels[0][sampleIdx];

            // DIFFERENCE: We always should learn at least once?
            // If our delta true is greater than the current quantilizer, we learn (Or some other stuff)
            if ((float) (err[sampleIdx]) >= positiveQuantVal || numBatchesTested > numSamplesLearnedFrom /*lim.@start*/) {
                // First step of learning is connecting
                // Why do we do this in the for loop in the batch?
                connect(maxConnectionsToGrowPerBatch, sampleIdx, yTrue, MAX_CONNECTION_WEIGHT / divNew);
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
                        connect(maxConnectionsToGrowPerBatch, sampleIdx, isnot, -MAX_CONNECTION_WEIGHT / divNew);
                        learn(sampleIdx, isnot, adjn);
                        toreload[isnot] = 1;
                        bu++;
                    }
                }
            }
            if (/*b/briefStep >prevStep*/ numSamplesLearnedFrom % briefStep == 0 && numSamplesLearnedFrom != prevStep) {
                printf("testing %d: ", nblearn);
                splitIndex = 0;
                printf("0: %2.3f  ", (float) test() / 600.0);
                splitIndex = 1;
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