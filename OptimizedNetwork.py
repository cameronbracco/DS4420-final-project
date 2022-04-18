import torch
from Quantilizer import Quantilizer
from timeit import default_timer as timer

class BetterSONN:
    def __init__(self,
                input_size,
                output_size,
                num_neurons_per_column,
                num_connections_per_neuron,
                spike_threshold=1000,
                max_update_threshold=3000,
                initial_connection_weight=100,
                positive_reinforce_amount=100,
                negative_reinforce_amount=5,
                decay_amount=0,
                prune_weight=-5, 
                device='cuda'):

        self.input_size = input_size # aka number of receptors
        self.output_size = output_size # aka numebr of columns
        self.num_neurons_per_column = num_neurons_per_column
        self.num_connections_per_neuron = num_connections_per_neuron

        self.spike_threshold = spike_threshold
        self.max_update_threshold = max_update_threshold # Not used right now
        self.positive_reinforce_amount = positive_reinforce_amount
        self.negative_reinforce_amount = negative_reinforce_amount
        self.decay_amount = decay_amount
        self.prune_weight = prune_weight
        self.initial_connection_weight = initial_connection_weight #???

        # Connection mask is [columns x neurons per column x input size]
        # Because right now we hard limit the number of connections to match the input size
        # So there's no double sampling of input pixels
        self.connection_masks = torch.zeros((self.output_size,
                                            self.num_neurons_per_column,
                                            self.input_size),
                                            dtype=torch.uint8,
                                            device=device)

        # Weights array is also [columns x neurons per column x input size]
        # Because even though they don't always exist, it makes the masking extremely easy                                    
        self.weight_arrays = torch.full((self.output_size,
                                        self.num_neurons_per_column,
                                        self.input_size),
                                        self.initial_connection_weight,
                                        dtype=torch.int32,
                                        device=device)
        
        # Initialize the postiive and negative quantilizers for each column
        self.positive_quantilizers = []
        self.negative_quantilizers = []
        for i in range(output_size):
            self.positive_quantilizers.append(Quantilizer(19, 1))
            self.negative_quantilizers.append(Quantilizer(150, 1))

        
    def learn(self, x, y_true):
        # First, get the model's prediction
        y_hat, spike_counts = self.forward(x)

        # Book keeping what we're learning
        positive_reinforcements = []
        negative_reinforcements = []

        # Calculate the average spike count (used for false learning)
        avg_spike_count = int(torch.sum(spike_counts) / spike_counts.shape[0])

        # Isolate just the false columns and find the maximum spike count
        false_column_spikes = torch.cat((spike_counts[0:y_true], spike_counts[y_true+1:]))
        max_false_spike = torch.amax(false_column_spikes)

        # Learn true (if necessary)
        delta_true = spike_counts[y_true] - max_false_spike
        if self.positive_quantilizers[y_true].check(delta_true):
            # print("positively reinforcing:", y_true)
            # Positively reinforce the weights of each neuron in true column
            positive_reinforcements.append(y_true)
            self.weight_arrays[y_true] += self.positive_reinforce_amount

        # Learn false (if necessary)
        for i in range(self.output_size):
            # We only want "false" columns
            if i != y_true:
               delta_false = spike_counts[i] - avg_spike_count
               if self.negative_quantilizers[i].check(delta_false):
                   # Negatively reinforce the weights of each neuron in this false column
                   negative_reinforcements.append(i)
                   self.weight_arrays[i] -= self.negative_reinforce_amount

        return y_hat, spike_counts, positive_reinforcements, negative_reinforcements
    
    def forward(self, x):
        # ------ Grow connections ------
        self.grow()

        # ------ Count spikes ------
        col_spike_counts = self.count_spikes(x)

        # ------ Select prediction ------
        prediction_idx = torch.argmax(col_spike_counts).item()

        # ------ Decay all connections ------
        self.decay()

        # ------ Prune bad connections ------
        self.prune()

        return prediction_idx, col_spike_counts
    
    def count_spikes(self, x):
        # Calculate potential from active receptors. Will be of size [columns x neurons per column]
        all_neuron_potentials = torch.sum(x * self.connection_masks * self.weight_arrays, axis=2)

        # Whether or not a neuron is spiking. Has shape [columns x neurons per column]
        all_neuron_spikes = torch.where(all_neuron_potentials < self.spike_threshold, 0, 1)

        # Calculate total spikes per column. Has shape [columns]
        col_spike_counts = torch.sum(all_neuron_spikes, axis=1)

        return col_spike_counts
 
    def grow(self):
        # We grow connections by adding ones to the connection mask where we want them
        # Notably, we need to make sure it's a _new_ connection (in the current setup)

        # Calculate the number of connections each neuron currently has ahead of time
        # Shape is [columns x neurons per column]
        curr_connection_counts = torch.sum(self.connection_masks, axis=2)

        # Identify which neurons need to be updated (those with non-full connections)
        # Produces a tensor w/ all the indices [column, neuron] for such neurons 
        neurons_to_update = torch.nonzero(curr_connection_counts != self.num_connections_per_neuron)
        connections_grown = 0

        # loop_start = timer()
        for neuron_idx in neurons_to_update:
            # Identify all the open connections for this neuron (open is when it's 0)
            open_connections = torch.nonzero(self.connection_masks[neuron_idx[0], neuron_idx[1]] == 0)
            rand_indices = torch.randperm(open_connections.shape[0])
            num_to_update = self.num_connections_per_neuron - curr_connection_counts[neuron_idx[0], neuron_idx[1]]
            connections_grown += num_to_update

            # Only select the first `num_to_update` indices to create connection
            for i in range(num_to_update):
                connection_index = rand_indices[i]
                self.connection_masks[neuron_idx[0], neuron_idx[1], connection_index] = 1
                self.weight_arrays[neuron_idx[0], neuron_idx[1], connection_index] = self.initial_connection_weight

    def decay(self):
        # All connections decay by the same amount over time
        self.weight_arrays -= self.decay_amount
    
    def prune(self):
        # If a connection drops below a specific weight, prune it by clearing the mask
        # NOTE: We don't bother resetting the weight array here, since it will be reset
        #       if the connection is ever re-added
        self.connection_masks = torch.where(torch.logical_or(self.weight_arrays < self.prune_weight,
                                            self.connection_masks == 0), 0, 1)
        