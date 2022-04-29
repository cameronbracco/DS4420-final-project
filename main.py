import logging

import hydra
import os
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import fmnist
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

    # Loading data
    if cfg.general.dataset_name == 'mnist':
        x_train, y_train, x_test, y_test = mnist.load(get_original_cwd())
    elif cfg.general.dataset_name == 'fmnist':
        x_train, y_train, x_test, y_test = fmnist.load(get_original_cwd())
    else:
        raise ValueError(f"UNKNOWN DATASET NAME: {cfg.general.dataset_name}")

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    num_examples_train = cfg.training.num_examples_train if cfg.training.num_examples_train > 0 else len(x_train)
    num_examples_val = cfg.training.num_examples_val if cfg.training.num_examples_val > 0 else len(x_test)

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
        max_connection_weight=cfg.network.max_connection_weight,
        min_overall_neuron_update_threshold=cfg.network.min_overall_neuron_update_threshold,
        max_overall_neuron_update_threshold=cfg.network.max_overall_neuron_update_threshold,
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
        max_neurons_to_grow_from_on_sample=cfg.network.max_neurons_to_grow_from_on_sample,
        decay_prune_every_n_samples=cfg.network.decay_prune_every_n_samples,
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
        "max_connection_weight": cfg.network.max_connection_weight,
        "min_overall_neuron_update_threshold": cfg.network.min_overall_neuron_update_threshold,
        "max_overall_neuron_update_threshold": cfg.network.max_overall_neuron_update_threshold,
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
        weights_updated, connections_grown, connections_pruned = 0, 0, 0
        pos_should_learn_per_column, neg_should_learn_per_column = torch.zeros(cfg.network.output_size, device=device), torch.zeros(cfg.network.output_size, device=device)
        predictions, y_trues, correct_count_train_per_class = [0] * 10, [0] * 10, [0] * 10

        indexes_perm = torch.randperm(num_examples_train) if cfg.training.shuffle_batches else range(num_examples_train)
        for count, i in enumerate(indexes_perm):
            x_raw = sampled_x_train[i, :].to(device)
            x = filtered_x_train[i, :].to(device)
            y = y_train[i].item()

            override_should_learn = (count % cfg.training.must_learn_every_n_iters) == 0
            pred, spikes, n_grown, n_pruned, n_updated, should_learn = model.fit(
                x, x_raw, y,
                override_should_learn=override_should_learn
            )

            # Accounting stuff, it's FOR SURE correct, I checked it twice, don't worry (ie. this is bAd)
            pos_should_learn_per_column[y] += should_learn[y]
            should_learn[y] = False
            neg_should_learn_per_column += should_learn

            weights_updated += n_updated
            connections_grown += n_grown
            connections_pruned += n_pruned

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
                    f"connections grown: {connections_grown} - "
                    f"connections pruned: {connections_pruned} - "
                    f"weights updated: {weights_updated} - "
                    f"predictions: [{', '.join([str(p) + '/' + str(t)  for p, t in zip(predictions, y_trues)])}]"
                )

        train_end = timer()

        train_time = train_end - train_start
        avg_train_time_per_example = train_time / num_examples_train

        val_start = timer()
        correct_count_val, val_predictions, val_y_trues, correct_count_val_per_class = validation(model, device, num_examples_val, filtered_x_test, y_test[:num_examples_val])
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
            f"connections grown: {connections_grown} - "
            f"connections pruned: {connections_pruned} - "
            f"weights updated: {weights_updated} - "
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
            "connections_grown": connections_grown,
            "connections_pruned": connections_pruned,
            "weights_updated": weights_updated,
            f"num_positive_reinforcements/total": pos_should_learn_per_column.sum().item(),
            f"num_negative_reinforcements/total": neg_should_learn_per_column.sum().item(),
            **{
                f"num_positive_reinforcements/column_{i}": pos_should_learn_per_column[i].item()
                for i in range(cfg.network.output_size)
            },
            **{
                f"num_negative_reinforcements/column_{i}": neg_should_learn_per_column[i].item()
                for i in range(cfg.network.output_size)
            },
            # **{
            #     f"training_pred_prop/{i}": predictions[i] / y_trues[i] if y_trues[i] > 0 else 0
            #     for i in range(cfg.network.output_size)
            # },
            **{
                f"training_preds/{i}": predictions[i]
                for i in range(cfg.network.output_size)
            },
            **{
                f"training_pred_accuracy/{i}": correct_count_train_per_class[i] / y_trues[i] if y_trues[i] > 0 else 0
                for i in range(cfg.network.output_size)
            },
            # **{
            #     f"validation_pred_prop/{i}": val_predictions[i] / val_y_trues[i] if val_y_trues[i] > 0 else 0
            #     for i in range(cfg.network.output_size)
            # },
            **{
                f"validation_preds/{i}": val_predictions[i]
                for i in range(cfg.network.output_size)
            },
            **{
                f"validation_pred_accuracy/{i}": correct_count_val_per_class[i] / val_y_trues[i] if val_y_trues[i] > 0 else 0
                for i in range(cfg.network.output_size)
            },
            "model_weights/positive/mean": model.get_weights_mean(group='positive'),
            "model_weights/negative/mean": model.get_weights_mean(group='negative'),
            "model_weights/all/mean": model.get_weights_mean(group='all'),
            "model_weights/positive/std": model.get_weights_std(group='positive'),
            "model_weights/negative/std": model.get_weights_std(group='negative'),
            "model_weights/all/std": model.get_weights_std(group='all'),
            "model_weights/positive/min": model.get_weights_min(group='positive'),
            "model_weights/negative/min": model.get_weights_min(group='negative'),
            "model_weights/all/min": model.get_weights_min(group='all'),
            "model_weights/positive/max": model.get_weights_max(group='positive'),
            "model_weights/negative/max": model.get_weights_max(group='negative'),
            "model_weights/all/max": model.get_weights_max(group='all'),
            "open_connections/total": model.get_count_total_connections(),
            "open_connections/positive": model.get_count_positive_weights(),
            "open_connections/negative": model.get_count_negative_weights(),
        })

        if epoch % cfg.checkpointing.save_every_n_epochs == 0:
            path = cfg.checkpointing.model_out_path.format(dataset_name=cfg.general.dataset_name)
            model.save(path)
            wandb.log_artifact(
                path,
                f"{cfg.checkpointing.name.format(dataset_name=cfg.general.dataset_name)}_epoch_{epoch}",
                'model'
            )


if __name__ == "__main__":
    main()
