from Neuron import Neuron
from Quantilizer import Quantilizer

class Column:

    def __init__(self, colNum, receptors, numNeurons, maxConnections, spikeThrehsold, updateThreshold):
        self.colNum = colNum
        self.neurons = []
        for i in range(numNeurons):
            neuron = Neuron(receptors, maxConnections, spikeThrehsold, updateThreshold)
            self.neurons.append(neuron)

        self.mostRecentSpikeCount = 0

        # TODO: Real values for these???
        self.positive_reinforce_amount = 100
        self.negative_reinforce_amount = 5
        self.decayAmount = 0 # Amount towards zero that each connection automatically decays
        self.pruneWeight = -5 # Weight below which connections wil be automatically pruned
        
        self.positiveQuantilizer = Quantilizer(19, 1)
        self.negativeQuantilizer = Quantilizer(150, 1)
    
    # Grows connections (if necessary) for all the neurons in the column
    def grow_connections(self):
        for neuron in self.neurons:
            neuron.grow()

    # Count the nunber of spiking neurons in this column
    def spike_count(self):
        self.mostRecentSpikeCount = sum(map(lambda neuron: neuron.spike_count(), self.neurons))
        return self.mostRecentSpikeCount

    # Learns, assuming this column is the true label
    def learn_true(self, maxFalseSpike):
        deltaTrue = self.mostRecentSpikeCount - maxFalseSpike

        # Only learn if this is remarkable (the top 5%)
        if self.positiveQuantilizer.check(deltaTrue):
            for neuron in self.neurons:
                neuron.positive_reinforce(self.positive_reinforce_amount)
    
    # Learns, assuming this column is the false label
    # TODO: Do we ever negatively reinforce the true label? Does that make sense?
    def learn_false(self, avgSpikeCount):
        deltaFalse = self.mostRecentSpikeCount - avgSpikeCount

        if self.negativeQuantilizer.check(deltaFalse):
            for neuron in self.neurons:
                neuron.negative_reinforce(self.negative_reinforce_amount)
    
    # Decays all the neurons in this group and prunes unnecessary connections
    def decay_and_prune(self):
        for neuron in self.neurons:
            neuron.decay(self.decayAmount)
            neuron.prune(self.pruneWeight)


