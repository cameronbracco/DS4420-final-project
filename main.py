import logging

import hydra
import os
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch import where, Tensor

import mnist
import random
import numpy as np
import wandb
import torch
from tqdm.auto import tqdm
from timeit import default_timer as timer

from utils import render, validation
from Network import SONN
from OptimizedNetwork import BetterSONN


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    os.environ["WANDB_MODE"] = cfg.WANDB_MODE

    wandb.init(
        project="test-project",
        entity="dpis-disciples",
        config={k: v for k, v in cfg.items()},
        tags=cfg.WANDB_TAGS.split(',')
    )

    print("Initializing...")

    # Loading MNIST data
    print("Loading data...")
    x_train, t_train, x_test, t_test = mnist.load(get_original_cwd())

    x_train: Tensor = torch.from_numpy(x_train)
    t_train: Tensor = torch.from_numpy(t_train)
    x_test: Tensor = torch.from_numpy(x_test)
    t_test: Tensor = torch.from_numpy(t_test)

    NUM_EXAMPLES_TRAIN = cfg.NUM_EXAMPLES_TRAIN if cfg.NUM_EXAMPLES_TRAIN > 0 else len(x_train)
    NUM_EXAMPLES_VAL = cfg.NUM_EXAMPLES_VAL if cfg.NUM_EXAMPLES_VAL > 0 else len(x_test)
    print(f"Data Loaded. Using {NUM_EXAMPLES_TRAIN} training samples and {NUM_EXAMPLES_VAL} validation samples")

    print("Preprocessing data...")
    FILTER_THRESHOLD = cfg.FILTER_THRESHOLD

    # Filters to keep anything at or _above_ the threshold value
    # Not that this turns into binary, so small intensities are equivalent to max intensity
    filtered_x_train = where(x_train[:NUM_EXAMPLES_TRAIN] >= FILTER_THRESHOLD, 1, 0)
    filtered_x_test = where(x_test[:NUM_EXAMPLES_TRAIN] >= FILTER_THRESHOLD, 1, 0)

    print("Preprocessing done.")

    # Hyper parameters
    NUM_EPOCHS = cfg.NUM_EPOCHS

    INPUT_SIZE = cfg.INPUT_SIZE
    OUTPUT_SIZE = cfg.OUTPUT_SIZE
    NUM_NEURONS_PER_COLUMN = cfg.NUM_NEURONS_PER_COLUMN
    NUM_CONNECTIONS_PER_NEURON = cfg.NUM_CONNECTIONS_PER_NEURON
    SPIKE_THRESHOLD = cfg.SPIKE_THRESHOLD
    MAX_UPDATE_THRESHOLD = cfg.MAX_UPDATE_THRESHOLD
    INITIAL_CONNECTION_WEIGHT = SPIKE_THRESHOLD / 10
    POSITIVE_REINFORCE_AMOUNT = cfg.POSITIVE_REINFORCE_AMOUNT
    NEGATIVE_REINFORCE_AMOUNT = cfg.NEGATIVE_REINFORCE_AMOUNT
    DECAY_AMOUNT = cfg.DECAY_AMOUNT
    PRUNE_WEIGHT = cfg.PRUNE_WEIGHT
    LOG_LEVEL = logging._nameToLevel[cfg.LOG_LEVEL]
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg.DEVICE == 'auto' else cfg.DEVICE
    print("RUNNING ON DEVICE:", DEVICE)

    RANDOM_SEED = cfg.RANDOM_SEED

    # Make sure to set the random seed (should propagate to all other imports of random)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Initialing model for MNIST
    model = BetterSONN(
        INPUT_SIZE,
        OUTPUT_SIZE,
        NUM_NEURONS_PER_COLUMN,
        NUM_CONNECTIONS_PER_NEURON,
        spike_threshold=SPIKE_THRESHOLD,
        max_update_threshold=MAX_UPDATE_THRESHOLD,
        initial_connection_weight=INITIAL_CONNECTION_WEIGHT,
        positive_reinforce_amount=POSITIVE_REINFORCE_AMOUNT,
        negative_reinforce_amount=NEGATIVE_REINFORCE_AMOUNT,
        decay_amount=DECAY_AMOUNT,
        prune_weight=PRUNE_WEIGHT,
        device=DEVICE
    )

    wandb.config = {
      "epochs": NUM_EPOCHS,
      "num_examples_train": NUM_EXAMPLES_TRAIN,
      "num_examples_val": NUM_EXAMPLES_VAL,
      "input_size": INPUT_SIZE,
      "output_size": OUTPUT_SIZE,
      "num_neurons_per_column": NUM_NEURONS_PER_COLUMN,
      "num_connections_per_neuron": NUM_CONNECTIONS_PER_NEURON,
      "spike_threshold": SPIKE_THRESHOLD,
      "max_update_threshold": MAX_UPDATE_THRESHOLD,
      "initial_connection_weight": INITIAL_CONNECTION_WEIGHT,
      "positive_reinforce_amount": POSITIVE_REINFORCE_AMOUNT,
      "negative_reinforce_amount": NEGATIVE_REINFORCE_AMOUNT,
      "decay_amount": DECAY_AMOUNT,
      "pruneWeight": PRUNE_WEIGHT,
      "device": DEVICE
    }

    print("Training...")
    for epoch in (epoch_pbar := tqdm(range(NUM_EPOCHS), position=0, leave=True)):

        correct_count_train = 0
        train_start = timer()
        num_pos_reinforces = 0
        num_neg_reinforces = 0
        connections_grown = 0
        predictions = [0] * 10
        for i in range(NUM_EXAMPLES_TRAIN):
            x = filtered_x_train[i,:].to(DEVICE)
            y = t_train[i].item()

            pred, spikes, pos_reinforcements, neg_reinforcements = model.learn(x, y)

            num_pos_reinforces += len(pos_reinforcements)
            num_neg_reinforces += len(neg_reinforcements)
            predictions[pred] += 1

            # Counting how many we get correct
            if pred == y:
                correct_count_train += 1

        train_end = timer()

        train_time = train_end - train_start
        avg_train_time_per_example = train_time / NUM_EXAMPLES_TRAIN

        val_start = timer()
        correct_count_val = validation(model, DEVICE, NUM_EXAMPLES_VAL, filtered_x_test, t_test[:NUM_EXAMPLES_VAL])
        val_end = timer()

        val_time = val_end - val_start
        avg_val_time_per_example = val_time / NUM_EXAMPLES_VAL

        train_accuracy = correct_count_train / NUM_EXAMPLES_TRAIN
        val_accuracy = correct_count_val / NUM_EXAMPLES_VAL

        epoch_pbar.set_description(
            f"Epoch {str(epoch).zfill(len(str(NUM_EPOCHS)))} - train acc: {train_accuracy:.5f} - avg train time: "
            f"{avg_train_time_per_example:.5f} - val acc: {val_accuracy:.5f} - avg val time: {avg_val_time_per_example:.5f}"
            f" - reinforcements: +{num_pos_reinforces}/-{num_neg_reinforces}"
        )
        if LOG_LEVEL == logging.DEBUG:
            print(predictions)

        # Logging to W&B
        wandb.log({
            "epoch": epoch,
            "train_accuracy": correct_count_train / NUM_EXAMPLES_TRAIN,
            "val_accuracy": correct_count_val / NUM_EXAMPLES_VAL,
            "train_time": train_time,
            "val_time": val_time,
            "positive_reinforces": num_pos_reinforces,
            "negative_reinforces": num_neg_reinforces,
        })

    print("Done!")

if __name__ == "__main__":
    main()
