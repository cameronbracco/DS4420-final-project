import logging

import hydra
import os
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import mnist
import random
import numpy as np
import wandb
import torch
from tqdm.auto import tqdm, trange
from timeit import default_timer as timer

from utils import render, validation
from Network import SONN
from OptimizedNetwork import BetterSONN


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    LOG_LEVEL = logging._nameToLevel[cfg.LOG_LEVEL]

    os.environ["WANDB_MODE"] = cfg.WANDB_MODE

    wandb.init(
        project="test-project",
        entity="dpis-disciples",
        config={k: v for k, v in cfg.items()},
        tags=cfg.WANDB_TAGS.split()
    )

    # Loading MNIST data
    x_train, t_train, x_test, t_test = mnist.load(get_original_cwd())

    x_train = torch.from_numpy(x_train)
    t_train = torch.from_numpy(t_train)
    x_test = torch.from_numpy(x_test)
    t_test = torch.from_numpy(t_test)

    num_examples_train = cfg.NUM_EXAMPLES_TRAIN if cfg.NUM_EXAMPLES_TRAIN > 0 else len(x_train)
    num_examples_val = cfg.NUM_EXAMPLES_VAL if cfg.NUM_EXAMPLES_VAL > 0 else len(x_test)

    filter_thresholds = torch.Tensor([int(t) for t in cfg.FILTER_THRESHOLDS.split()])

    sampled_x_train = x_train[:num_examples_train]
    sampled_x_test = x_test[:num_examples_val]

    filtered_x_train = torch.where(sampled_x_train.unsqueeze(2) >= filter_thresholds, 1, 0)
    filtered_x_test = torch.where(sampled_x_test.unsqueeze(2) >= filter_thresholds, 1, 0)

    initial_connection_weight = cfg.INITIAL_CONNECTION_WEIGHT if cfg.INITIAL_CONNECTION_WEIGHT \
        else int(cfg.SPIKE_THRESHOLD / 10)

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg.DEVICE == 'auto' else cfg.DEVICE

    RANDOM_SEED = cfg.RANDOM_SEED

    # Make sure to set the random seed (should propogate to all other imports of random)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Initialize model for MNIST
    model = BetterSONN(
        cfg.INPUT_SIZE,
        cfg.OUTPUT_SIZE,
        cfg.NUM_NEURONS_PER_COLUMN,
        cfg.NUM_CONNECTIONS_PER_NEURON,
        spike_threshold=cfg.SPIKE_THRESHOLD,
        max_update_threshold=cfg.MAX_UPDATE_THRESHOLD,
        initial_connection_weight=initial_connection_weight,
        initial_connection_weight_delta=(
            cfg.INITIAL_CONNECTION_WEIGHT_DELTA_MIN,
            cfg.INITIAL_CONNECTION_WEIGHT_DELTA_MAX
        ),
        positive_reinforce_amount=cfg.POSITIVE_REINFORCE_AMOUNT,
        negative_reinforce_amount=cfg.NEGATIVE_REINFORCE_AMOUNT,
        positive_quantilizer_inc_value=cfg.POSITIVE_QUANTILIZER_INC_VALUE,
        positive_quantilizer_dec_value=cfg.POSITIVE_QUANTILIZER_DEC_VALUE,
        positive_quantilizer_denom=cfg.POSITIVE_QUANTILIZER_DENOM,
        negative_quantilizer_increment_value=cfg.NEGATIVE_QUANTILIZER_INC_VALUE,
        negative_quantilizer_decrement_value=cfg.NEGATIVE_QUANTILIZER_DEC_VALUE,
        negative_quantilizer_denom=cfg.NEGATIVE_QUANTILIZER_DENOM,
        decay_amount=cfg.DECAY_AMOUNT,
        prune_weight=cfg.PRUNE_WEIGHT,
        device=device
    )

    wandb.config = {
        "epochs": cfg.NUM_EPOCHS,
        "num_examples_train": num_examples_train,
        "num_examples_val": num_examples_val,
        "input_size": cfg.INPUT_SIZE,
        "output_size": cfg.OUTPUT_SIZE,
        "num_neurons_per_column": cfg.NUM_NEURONS_PER_COLUMN,
        "num_connections_per_neuron": cfg.NUM_CONNECTIONS_PER_NEURON,
        "spike_threshold": cfg.SPIKE_THRESHOLD,
        "max_update_threshold": cfg.MAX_UPDATE_THRESHOLD,
        "initial_connection_weight": initial_connection_weight,
        "initial_connection_weight_delta_min": cfg.INITIAL_CONNECTION_WEIGHT_DELTA_MIN,
        "initial_connection_weight_delta_max": cfg.INITIAL_CONNECTION_WEIGHT_DELTA_MAX,
        "positive_reinforce_amount": cfg.POSITIVE_REINFORCE_AMOUNT,
        "negative_reinforce_amount": cfg.NEGATIVE_REINFORCE_AMOUNT,
        "positive_quantilizer_inc_value": cfg.POSITIVE_QUANTILIZER_INC_VALUE,
        "positive_quantilizer_dec_value": cfg.POSITIVE_QUANTILIZER_DEC_VALUE,
        "positive_quantilizer_denom": cfg.POSITIVE_QUANTILIZER_DENOM,
        "negative_quantilizer_inc_value": cfg.NEGATIVE_QUANTILIZER_INC_VALUE,
        "negative_quantilizer_dec_value": cfg.NEGATIVE_QUANTILIZER_DEC_VALUE,
        "negative_quantilizer_denom": cfg.NEGATIVE_QUANTILIZER_DENOM,
        "decay_amount": cfg.DECAY_AMOUNT,
        "prune_weight": cfg.PRUNE_WEIGHT,
        "pixel_filter_thresholds": cfg.FILTER_THRESHOLDS,
        "shuffle_batches": cfg.SHUFFLE_BATCHES,
        "device": device
    }

    train_accuracy, avg_train_time_per_example, val_accuracy, avg_val_time_per_example = 0, 0, 0, 0

    update_pbar_every_n = num_examples_train / cfg.UPDATE_PBAR_N_TIMES_PER_EPOCH

    for epoch in (epoch_pbar := tqdm(range(cfg.NUM_EPOCHS), position=0, leave=True)):

        correct_count_train = 0
        train_start = timer()
        num_pos_reinforces = 0
        num_neg_reinforces = 0
        connections_grown = 0
        predictions = [0] * 10
        y_trues = [0] * 10
        correct_count_train_per_class = [0] * 10

        indeces_perm = torch.randperm(num_examples_train) if cfg.SHUFFLE_BATCHES else range(num_examples_train)
        for count, i in enumerate(indeces_perm):
            x = filtered_x_train[i, :].to(device)
            y = t_train[i].item()

            pred, spikes, pos_reinforcements, neg_reinforcements = model.learn(x, y)

            num_pos_reinforces += len(pos_reinforcements)
            num_neg_reinforces += len(neg_reinforcements)
            predictions[pred] += 1
            y_trues[y] += 1

            # Counting how many we get correct
            if pred == y:
                correct_count_train += 1
                correct_count_train_per_class[y] += 1

            if (count + 1) % update_pbar_every_n == 0:
                epoch_pbar.set_description(
                    f"Epoch {str(epoch).zfill(len(str(cfg.NUM_EPOCHS)))} "
                    f"({str(count + 1).zfill(len(str(num_examples_train)))}/{num_examples_train}) - "
                    f"train acc: {train_accuracy:.5f} - "
                    f"avg train time: {avg_train_time_per_example:.5f} - "
                    f"val acc: {val_accuracy:.5f} - "
                    f"avg val time: {avg_val_time_per_example:.5f} - "
                    f"reinforcements: +{num_pos_reinforces}/-{num_neg_reinforces} - "
                    f"predictions: [{', '.join([str(p) + '/' + str(t)  for p, t in zip(predictions, y_trues)])}]"
                )

        train_end = timer()

        train_time = train_end - train_start
        avg_train_time_per_example = train_time / num_examples_train

        val_start = timer()
        correct_count_val = validation(model, device, num_examples_val, filtered_x_test, t_test[:num_examples_val])
        val_end = timer()

        val_time = val_end - val_start
        avg_val_time_per_example = val_time / num_examples_val

        train_accuracy = correct_count_train / num_examples_train
        val_accuracy = correct_count_val / num_examples_val

        epoch_pbar.set_description(
            f"Epoch {str(epoch).zfill(len(str(cfg.NUM_EPOCHS)))} {0}/{num_examples_train} "
            f"({str(0).zfill(len(str(num_examples_train)))}/{num_examples_train}) - "
            f"train acc: {train_accuracy:.5f} - "
            f"avg train time: {avg_train_time_per_example:.5f} - "
            f"val acc: {val_accuracy:.5f} - "
            f"avg val time: {avg_val_time_per_example:.5f} - "
            f"reinforcements: +{num_pos_reinforces}/-{num_neg_reinforces} - "
            f"predictions: [{', '.join([str(p) + '/' + str(t)  for p, t in zip(predictions, y_trues)])}]"
        )
        if LOG_LEVEL == logging.DEBUG:
            print(predictions)

        # Logging to W&B
        wandb.log({
            "epoch": epoch,
            "train_accuracy": correct_count_train / num_examples_train,
            "val_accuracy": correct_count_val / num_examples_val,
            "train_time": train_time,
            "val_time": val_time,
            "positive_reinforces": num_pos_reinforces,
            "negative_reinforces": num_neg_reinforces,
            **{
                f"training_pred_prop/{i}": predictions[i] / y_trues[i] if y_trues[i] > 0 else 0
                for i in range(10)
            },
            **{
                f"training_preds/{i}": predictions[i]
                for i in range(10)
            },
            **{
                f"training_pred_accuracy/{i}": correct_count_train_per_class[i] / y_trues[i] if y_trues[i] > 0 else 0
                for i in range(10)
            }
        })


if __name__ == "__main__":
    main()
