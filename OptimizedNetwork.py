from pickle import dump, load
from typing import Tuple, List

import torch
from torch import Tensor

from Quantilizer import Quantilizer
from timeit import default_timer as timer


class BetterSONN:
    def __init__(
            self,
            input_size,
            output_size,
            num_neurons_per_column,
            num_connections_per_neuron,
            spike_threshold=1000,
            max_connection_weight=1000000,
            min_overall_neuron_update_threshold=300000,
            max_overall_neuron_update_threshold=1200000,
            initial_connection_weight=100,
            initial_connection_weight_delta=(0, 1),
            positive_reinforce_amount=100,
            negative_reinforce_amount=5,
            positive_quantilizer_inc_value=19,
            positive_quantilizer_dec_value=1,
            positive_quantilizer_denom=100,
            negative_quantilizer_increment_value=150,
            negative_quantilizer_decrement_value=1,
            negative_quantilizer_denom=900,
            decay_amount=0,
            prune_weight=-5,
            max_neurons_to_grow_from_on_sample=10,
            decay_prune_every_n_samples=100,
            device='cuda'
    ):

        self.input_size = input_size  # aka number of receptors
        self.output_size = output_size  # aka number of columns

        self.num_neurons_per_column = num_neurons_per_column
        self.num_connections_per_neuron = num_connections_per_neuron

        self.spike_threshold = spike_threshold
        self.max_connection_weight = max_connection_weight
        self.min_overall_neuron_update_threshold = min_overall_neuron_update_threshold
        self.max_overall_neuron_update_threshold = max_overall_neuron_update_threshold

        self.positive_reinforce_amount = positive_reinforce_amount
        self.negative_reinforce_amount = negative_reinforce_amount

        self.decay_amount = decay_amount
        self.prune_weight = prune_weight

        self.initial_connection_weight: int = initial_connection_weight
        self.initial_connection_weight_delta: Tuple[int] = initial_connection_weight_delta

        self.max_neurons_to_grow_from_on_sample = max_neurons_to_grow_from_on_sample

        self.device = device

        # Connection mask is [columns x neurons per column x input size]
        # Because right now we hard limit the number of connections to match the input size
        # So there's no double sampling of input pixels
        self.connection_masks = torch.zeros(
            (self.output_size, self.num_neurons_per_column, self.input_size),
            dtype=torch.uint8,
            device=self.device
        )

        # Weights array is also [columns x neurons per column x input size]
        # Because even though they don't always exist, it makes the masking extremely easy                                    
        self.weight_arrays = torch.randint(
            self.initial_connection_weight - self.initial_connection_weight_delta[0],
            self.initial_connection_weight + self.initial_connection_weight_delta[1],
            (self.output_size, self.num_neurons_per_column, self.input_size),
            dtype=torch.int32,
            device=self.device
        )

        # Initialize the positive and negative quantilizers

        self.positive_quantilizer_increment_value = positive_quantilizer_inc_value
        self.positive_quantilizer_decrement_value = positive_quantilizer_dec_value
        self.positive_quantilizer_denom = positive_quantilizer_denom
        self.positive_quantilizer = Quantilizer(
            self.positive_quantilizer_increment_value / self.positive_quantilizer_denom,
            self.positive_quantilizer_decrement_value / self.positive_quantilizer_denom
        )

        self.negative_quantilizer_increment_value = negative_quantilizer_increment_value
        self.negative_quantilizer_decrement_value = negative_quantilizer_decrement_value
        self.negative_quantilizer_denom = negative_quantilizer_denom
        self.negative_quantilizer = Quantilizer(
            self.negative_quantilizer_increment_value / self.negative_quantilizer_denom,
            self.negative_quantilizer_decrement_value / self.negative_quantilizer_denom
        )

        self.decay_prune_every_n_samples = decay_prune_every_n_samples
        self.samples_learned_from = 0

    def quantilize(self, col_spike_counts, y_true) -> Tensor:
        # Bookkeeping what we're learning
        should_learn = torch.full((self.output_size,), False, device=self.device, dtype=torch.bool)

        # Isolate just the false columns and find the maximum spike count
        false_column_spikes = torch.cat((col_spike_counts[0:y_true], col_spike_counts[y_true + 1:]))
        max_false_spike = torch.amax(false_column_spikes)

        # Learn true (if necessary)
        delta_true = max_false_spike - col_spike_counts[y_true]
        if self.positive_quantilizer.check(delta_true):
            should_learn[y_true] = True

        # Learn false (if necessary)
        # Calculate the average spike count (used for false learning)
        total_spike_counts = torch.sum(col_spike_counts, dim=0)
        avg_spike_count_except_col = (total_spike_counts - col_spike_counts) / (col_spike_counts.shape[0] - 1)
        for i in range(self.output_size):
            # We only want "false" columns
            if i != y_true:
                delta_false = col_spike_counts[i] - avg_spike_count_except_col[i]
                if self.negative_quantilizer.check(delta_false):
                    should_learn[i] = True

        return should_learn

    def forward(self, x):
        # ------ Count spikes ------
        return self.count_spikes(x)

    def predict(self, x):
        col_spike_counts = self.forward(x)
        y_hat = torch.argmax(col_spike_counts).item()
        return y_hat

    def update_weights(self, x, y_true, should_learn) -> Tuple[int, int]:
        n_pruned, n_updated = 0, 0
        weights_per_column_for_cycle = torch.zeros((self.output_size, self.num_neurons_per_column), device=self.device).int()
        n_cycles = x.size(1)
        for cycle in range(n_cycles):
            # dims: [columns x neurons x connections]
            # will be nonzero if the connection should be updated (based on whether:
            #       we should learn for the column,
            #       the connection is non-open, and
            #       the intensity of the data passes the current intensity filter
            learnable_connections = self.connection_masks * should_learn.view(should_learn.size(0), 1, 1)
            weights_per_column_per_neuron_for_cycle = learnable_connections * self.weight_arrays * x[:, cycle]

            # dims: [columns x neurons]
            weights_per_column_for_cycle += weights_per_column_per_neuron_for_cycle.sum(dim=2)

            passes_min_threshold = weights_per_column_for_cycle > self.min_overall_neuron_update_threshold
            passes_max_threshold = weights_per_column_for_cycle < self.max_overall_neuron_update_threshold

            # dims: [columns]
            amount_to_adjust_per_column = torch.full((self.output_size,), self.negative_reinforce_amount, device=self.device)
            amount_to_adjust_per_column[y_true] = self.positive_reinforce_amount

            passes_max_connection_weight = torch.abs(self.weight_arrays + amount_to_adjust_per_column.view(self.output_size, 1, 1)) < self.max_connection_weight

            weight_adjustment_mask = (passes_min_threshold * passes_max_threshold).unsqueeze(2) * passes_max_connection_weight
            masked_weight_adjustments = weight_adjustment_mask * amount_to_adjust_per_column.view(self.output_size, 1, 1)

            self.weight_arrays += masked_weight_adjustments
            n_updated += len(torch.nonzero(masked_weight_adjustments))
            n_pruned += self.prune()

            weights_per_column_for_cycle[passes_min_threshold] = 0
            weights_per_column_for_cycle = weights_per_column_for_cycle.bitwise_right_shift(1)  # divides by 2

        return n_pruned, n_updated

    def fit(self, x, x_raw, y_true, override_should_learn=False):  # change this to training only
        col_spike_counts = self.forward(x)

        # ------ Select prediction ------
        y_hat = torch.argmax(col_spike_counts).item()

        should_learn = self.quantilize(col_spike_counts, y_true)
        should_learn[y_true] = should_learn[y_true] or override_should_learn
        if should_learn[y_true]:
            self.samples_learned_from += 1

        # ------ Grow connections ------
        n_grown = self.grow(x_raw, y_true, should_learn)

        # ------ Learn ------
        n_pruned, n_updated = self.update_weights(x, y_true, should_learn)

        if should_learn[y_true] and self.samples_learned_from % self.decay_prune_every_n_samples:
            # ------ Decay all connections ------
            self.decay()  # TODO: if should_we_decay?, based on batches?

            # ------ Prune bad connections ------
            n_pruned += self.prune()

        return y_hat, col_spike_counts, n_grown, n_pruned, n_updated, should_learn

    def count_spikes(self, x):

        # Calculate potential from active receptors. Will be of size [columns x neurons per column]
        all_neuron_potentials = (x * (self.connection_masks * self.weight_arrays).unsqueeze(3)).sum(dim=(2, 3))

        # Whether a neuron is spiking. Has shape [columns x neurons per column]
        all_neuron_spikes = torch.where(all_neuron_potentials < self.spike_threshold, 0, 1)

        # Calculate total spikes per column. Has shape [columns]
        col_spike_counts = torch.sum(all_neuron_spikes, dim=1)

        return col_spike_counts

    def grow(self, x_raw, y_true, should_learn: Tensor):
        # We grow connections by adding ones to the connection mask where we want them
        # Notably, we need to make sure it's a _new_ connection (in the current setup)

        # Calculate the number of connections each neuron currently has ahead of time
        # Shape is [columns x neurons per column]
        curr_connection_counts = torch.sum(self.connection_masks, dim=2)

        # Identify which neurons need to be updated (those with non-full connections)
        # Produces a tensor w/ all the indices [column, neuron] for such neurons 
        neurons_to_update = torch.nonzero(torch.logical_and(
            curr_connection_counts != self.num_connections_per_neuron,
            should_learn.unsqueeze(1)
        ))
        connections_grown = 0

        # loop_start = timer()
        neurons_to_update_idxs = torch.randperm(neurons_to_update.shape[0])[:self.max_neurons_to_grow_from_on_sample]
        for idx in neurons_to_update_idxs:
            column_idx, neuron_idx = neurons_to_update[idx]
            # Identify all the open connections for this neuron (open is when it's 0) ALL ARE OPEN?
            active_connections = x_raw > 0
            growable_connections = torch.nonzero(
                torch.logical_and(
                    self.connection_masks[column_idx, neuron_idx] == 0,  # open connections
                    active_connections
                )
            )
            rand_indices = torch.randperm(growable_connections.shape[0])

            num_to_update = min(
                self.num_connections_per_neuron - curr_connection_counts[column_idx, neuron_idx],
                rand_indices.shape[0]
            )
            connections_grown += num_to_update

            growing_connections = growable_connections[rand_indices[:num_to_update]]

            # Only select the first `num_to_update` indices to create connection
            self.connection_masks[column_idx, neuron_idx, growing_connections] = 1

            # Selecting the initial weight values for the new connections
            # For delta[0] == 0 and delta[1] == 1, values are NOT random, and fixed at the given initial weight
            # Random values will be generated in the integer range:
            #           [initial_connection_weight - delta[0], initial_connection_weight - delta[1])
            initial_weights = torch.randint(
                self.initial_connection_weight - self.initial_connection_weight_delta[0],
                self.initial_connection_weight + self.initial_connection_weight_delta[1],
                (num_to_update,),
                device=self.device
            ).int()

            # The initial_weights should only be positive on column corresponding to the true label of the currently
            # learned sample. For all others, init to NEGATIVE instead
            if column_idx != y_true:
                initial_weights = -initial_weights
            self.weight_arrays[column_idx, neuron_idx, growing_connections] = initial_weights.unsqueeze(1)

        return connections_grown

    def decay(self):
        # All connections decay (abs(weight) -> 0) by the same amount over time
        self.weight_arrays[self.weight_arrays > 0] -= self.decay_amount
        self.weight_arrays[self.weight_arrays < 0] += self.decay_amount

    def prune(self):
        # If a connection drops below a specific weight, prune it by clearing the mask
        count_pruned = torch.logical_and(
            torch.abs(self.weight_arrays) < self.prune_weight,
            self.connection_masks == 1
        ).sum()
        self.connection_masks = torch.where(
            torch.logical_or(
                torch.abs(self.weight_arrays) < self.prune_weight,
                self.connection_masks == 0
            ),
            0, 1
        )
        # We don't NEED to reset the weight array here, since it will be reset if the connection is ever re-added,
        # but we do it anyway for accounting reasons
        self.weight_arrays *= self.connection_masks
        return count_pruned

    def _get_model_weights_group(self, group) -> Tensor:
        if group == 'all':
            return self.weight_arrays[self.weight_arrays != 0].float()
        if group == 'positive':
            return self.weight_arrays[self.weight_arrays > 0].float()
        if group == 'negative':
            return self.weight_arrays[self.weight_arrays < 0].float()

    def get_weights_mean(self, group='all'):
        group_weights = self._get_model_weights_group(group)
        return torch.mean(group_weights) if group_weights.numel() != 0 else 0

    def get_weights_std(self, group='all'):
        group_weights = self._get_model_weights_group(group)
        return torch.std(group_weights) if group_weights.numel() != 0 else 0

    def get_weights_min(self, group='all'):
        group_weights = self._get_model_weights_group(group)
        return torch.min(group_weights) if group_weights.numel() != 0 else 0

    def get_weights_max(self, group='all'):
        group_weights = self._get_model_weights_group(group)
        return torch.max(group_weights) if group_weights.numel() != 0 else 0

    def get_count_total_connections(self):
        return self._get_model_weights_group('all').nelement()

    def get_count_positive_weights(self):
        return self._get_model_weights_group('positive').nelement()

    def get_count_negative_weights(self):
        return self._get_model_weights_group('negative').nelement()

    def save(self, path):
        with open(path, "wb") as f:
            dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return load(f)
