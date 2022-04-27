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
from tqdm.auto import tqdm
from timeit import default_timer as timer

from utils import validation
from OptimizedNetwork import BetterSONN


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    LOG_LEVEL = logging._nameToLevel[cfg.general.log_level]

    os.environ["WANDB_MODE"] = cfg.wandb.mode

    wandb.init(
        project="test-project",
        entity="dpis-disciples",
        config={k: v for k, v in cfg.items()},
        tags=cfg.wandb.tags.split()
    )

    # Loading MNIST data
    x_train, t_train, x_test, t_test = mnist.load(get_original_cwd())

    x_train = torch.from_numpy(x_train)
    t_train = torch.from_numpy(t_train)
    x_test = torch.from_numpy(x_test)
    t_test = torch.from_numpy(t_test)

    num_examples_train = cfg.training.num_examples_train if cfg.training.num_examples_train > 0 else len(x_train)
    num_examples_val = cfg.training.num_examples_Val if cfg.training.num_examples_Val > 0 else len(x_test)

    filter_thresholds = torch.Tensor([int(t) for t in cfg.preprocessing.pixel_intensity_levels.split()])

    sampled_x_train = x_train[:num_examples_train]
    sampled_x_test = x_test[:num_examples_val]

    filtered_x_train = torch.where(sampled_x_train.unsqueeze(2) >= filter_thresholds, 1, 0)
    filtered_x_test = torch.where(sampled_x_test.unsqueeze(2) >= filter_thresholds, 1, 0)

    initial_connection_weight = cfg.network.initial_connection_weight.weight_value \
        if cfg.network.initial_connection_weight.weight_value \
        else int(cfg.network.spike_threshold / 10)

    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if cfg.general.device == 'auto' \
        else cfg.general.device
    print("Using device:", device)

    RANDOM_SEED = cfg.general.random_seed

    # Make sure to set the random seed (should propogate to all other imports of random)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Initialize model for MNIST
    model = BetterSONN(
        cfg.network.input_size,
        cfg.network.output_size,
        cfg.network.num_neurons_per_column,
        cfg.network.num_connections_per_column,
        spike_threshold=cfg.network.spike_threshold,
        max_update_threshold=cfg.network.max_update_threshold,
        initial_connection_weight=initial_connection_weight,
        initial_connection_weight_delta=(
            cfg.network.initial_connection_weight.delta_min,
            cfg.network.initial_connection_weight.delta_max
        ),
        positive_reinforce_amount=cfg.network.positive_reinforce_amount,
        negative_reinforce_amount=cfg.network.negative_reinforce_amount,
        positive_quantilizer_inc_value=cfg.network.positive_quantilizer.inc_value,
        positive_quantilizer_dec_value=cfg.network.positive_quantilizer.dec_value,
        positive_quantilizer_denom=cfg.network.positive_quantilizer.denom,
        negative_quantilizer_increment_value=cfg.network.negative_quantilizer.inc_value,
        negative_quantilizer_decrement_value=cfg.network.negative_quantilizer.dec_value,
        negative_quantilizer_denom=cfg.network.negative_quantilizer.denom,
        decay_amount=cfg.network.decay_amount,
        prune_weight=cfg.network.prune_weight,
        device=device
    )

    wandb.config = {
        "epochs": cfg.training.num_epochs,
        "num_examples_train": num_examples_train,
        "num_examples_val": num_examples_val,
        "input_size": cfg.network.input_size,
        "output_size": cfg.network.output_size,
        "num_neurons_per_column": cfg.network.num_neurons_per_column,
        "num_connections_per_neuron": cfg.network.num_connections_per_column,
        "spike_threshold": cfg.network.spike_threshold,
        "max_update_threshold": cfg.network.max_update_threshold,
        "initial_connection_weight": initial_connection_weight,
        "initial_connection_weight_delta_min": cfg.network.initial_connection_weight.delta_min,
        "initial_connection_weight_delta_max": cfg.network.initial_connection_weight.delta_max,
        "positive_reinforce_amount": cfg.network.positive_reinforce_amount,
        "negative_reinforce_amount": cfg.network.negative_reinforce_amount,
        "positive_quantilizer_inc_value": cfg.network.positive_quantilizer.inc_value,
        "positive_quantilizer_dec_value": cfg.network.positive_quantilizer.dec_value,
        "positive_quantilizer_denom": cfg.network.positive_quantilizer.denom,
        "negative_quantilizer_inc_value": cfg.network.negative_quantilizer.inc_value,
        "negative_quantilizer_dec_value": cfg.network.negative_quantilizer.dec_value,
        "negative_quantilizer_denom": cfg.network.negative_quantilizer.denom,
        "decay_amount": cfg.network.decay_amount,
        "prune_weight": cfg.network.prune_weight,
        "pixel_filter_levels": cfg.preprocessing.pixel_intensity_levels,
        "shuffle_batches": cfg.training.shuffle_batches,
        "must_learn_every_n_iters": cfg.training.must_learn_every_n_iters,
        "device": device
    }

    train_accuracy, avg_train_time_per_example, val_accuracy, avg_val_time_per_example = 0, 0, 0, 0

    update_pbar_every_n = num_examples_train / cfg.general.update_pbar_n_times_per_epoch

    for epoch in (epoch_pbar := tqdm(range(cfg.training.num_epochs), position=0, leave=True)):

        correct_count_train = 0
        train_start = timer()
        num_pos_reinforces = 0
        num_neg_reinforces = 0
        connections_grown = 0
        predictions = [0] * 10
        y_trues = [0] * 10
        correct_count_train_per_class = [0] * 10

        indexes_perm = torch.randperm(num_examples_train) if cfg.training.shuffle_batches else range(num_examples_train)
        for count, i in enumerate(indexes_perm):
            x_raw = sampled_x_train[i, :].to(device)
            x = filtered_x_train[i, :].to(device)
            y = t_train[i].item()

            override_should_learn = (count % cfg.training.must_learn_every_n_iters) == 0
            pred, spikes, pos_reinforcements, neg_reinforcements = model.fit(
                x, x_raw, y,
                override_should_learn=override_should_learn
            )

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
                    f"Epoch {str(epoch).zfill(len(str(cfg.training.num_epochs)))} "
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
            f"Epoch {str(epoch).zfill(len(str(cfg.training.num_epochs)))} {0}/{num_examples_train} "
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
