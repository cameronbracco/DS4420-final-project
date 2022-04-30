from Receptor import Receptor
from Column import Column

import numpy as np

# Main class for the entire network
class SONN:

    # Assumes that input_size and output_size are flat
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # Setting up receptors (input signal things which are super verbose right now)
        self.receptors = []
        for i in range(self.input_size):
            receptor = Receptor(i)
            self.receptors.append(receptor)

        # Setting up columns
        self.columns = []
        NUM_NEURONS_PER_COLUMN = 200 # 7920 ideally? TODO
        NUM_CONNECTIONS_PER_NEURON = 10 # I think this is ideal
        SPIKE_THRESHOLD = 1000
        MAX_UPDATE_THRESHOLD = 3000
        for i in range(self.output_size):
            column = Column(i,
                            self.receptors,
                            NUM_NEURONS_PER_COLUMN,
                            NUM_CONNECTIONS_PER_NEURON,
                            SPIKE_THRESHOLD,
                            MAX_UPDATE_THRESHOLD)

            self.columns.append(column)

    # Evaluates the network on a given sample
    def forward(self, sample):
        # Update receptors
        for receptor in self.receptors:
            receptor.update(sample)

        # Grow connections
        for column in self.columns:
            column.grow_connections()

        # Count spikes for each column
        spikeCounts = list(map(lambda column: column.spike_count(), self.columns))

        # Create a prediction
        predictedColumnIdx = np.argmax(spikeCounts)

        # Cleanup stuff for each column
        for column in self.columns:
            column.decay_and_prune()

        return predictedColumnIdx, spikeCounts
        
    def learn(self, sample, trueColumnIdx):
        # First get the model prediction
        predColumnIdx, spikeCounts = self.forward(sample)

        # Calculate the average spike count (used in false learning)
        avgSpikeCount = int(sum(spikeCounts) / len(spikeCounts))
        
        # Then isolate the false columns and find the maximum spike count
        falseColumnSpikeCounts = spikeCounts[:trueColumnIdx] + spikeCounts[trueColumnIdx + 1:]
        maxFalseSpikeCount = np.amax(falseColumnSpikeCounts)

        # Learn (if necessary) on the true column
        trueColumn = self.columns[trueColumnIdx]
        trueColumn.learn_true(maxFalseSpikeCount)

        # Learn (if necessary) on the false columns
        for i in range(self.output_size):
            # We only want "false" columns
            if i != trueColumnIdx:
                falseColumn = self.columns[i]
                falseColumn.learn_false(avgSpikeCount)
        
        # Finally, returning these at the end so we can track progress
        return predColumnIdx, spikeCounts



