from Receptor import Receptor
from Quantilizer import Quantilizer

import random

class Neuron:

    def __init__(self, receptors, maxConnections, spikeThrehsold, updateThreshold):
        # References to the actual receptors
        self.receptorsRef = receptors

        # Empty connections to start
        self.connections = []

        # Maximum number of connections that this neuron can hold
        self.maxConnections = maxConnections

        # The potential of the neuron must be above this threshold to spike
        self.spikeThrehsold = spikeThrehsold

        # The potential of the neuron must be above this threshold to *learn*
        # The idea is that if the potential is above this threshold then the neuron is
        # already effective and shouldn't need to update
        self.updateThreshold = updateThreshold

        # Storing the potential so we can learn at a later point than when it is calculated
        self.mostRecentPotential = 0
    
    # Decays all connections in this neuron by a set amount
    def decay(self, decayAmount):
        for conn in self.connections:
            conn.decay(decayAmount)

    # Prune any connections below the minimum weight
    def prune(self, minWeight):
        self.connections = list(filter(lambda conn: conn.weight >= minWeight, self.connections))

    # Grow new connections
    def grow(self):
        numToGrow = self.maxConnections - len(self.connections)

        # Each new connection should be to a randomly selected receptor
        for i in range(numToGrow):
            # TODO: Initial connection threshold...
            conn = Connection(random.choice(self.receptorsRef), int(self.spikeThrehsold / 10))
            self.connections.append(conn)
            
    
    # Determines if this neuron is spiking. Returns a 1 if spiking, 0 otherwise
    def spike_count(self):
        # Potential is the sum of all the connections' potentials
        incomingPotential = sum(map(lambda conn: conn.calc_potential(), self.connections))

        self.mostRecentPotential = incomingPotential

        # Determine and return whether or not we should spike
        # print("Neuron spike time:", neuron_spike_end - neuron_spike_start)
        if self.mostRecentPotential > self.spikeThrehsold:
            return 1
        else:
            return 0

    # Positively reinforce all of the connections in this neuron
    def positive_reinforce(self, amount):
        for conn in self.connections:
            conn.positive_reinforce(amount)

    # Negatively reinforce all of the connections in this neuron
    def negative_reinforce(self, amount):
        for conn in self.connections:
            conn.negative_reinforce(amount)



class Connection:
    def __init__(self, receptor, weight):
        self.receptor = receptor
        self.weight = weight
    
    # Increases the weight of this connection by a set amount
    # ONLY FOR ACTIVE RECEPTORS
    def positive_reinforce(self, amount):
        if self.receptor.activation == 1:
            self.weight += amount
    
    # Reduces the weight of this connection by a set amount
    # ONLY FOR ACTIVE RECEPTORS
    def negative_reinforce(self, amount):
        if self.receptor.activation == 1:
            self.weight -= amount
    
    # Decays the weight of this connection towards zero by a set amount
    def decay(self, amount):
        if self.weight < 0:
            self.weight += amount
        else:
            self.weight -= amount
    
    # Calculates the potential from this connect (0 if receptor off, self.weight if receptor on)
    def calc_potential(self):
        return self.receptor.activation * self.weight
    

        