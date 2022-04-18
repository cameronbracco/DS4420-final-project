import mnist
import random
import numpy as np
import wandb
import torch
from tqdm import tqdm, trange
from timeit import default_timer as timer

from utils import render, validation
from Network import SONN
from OptimizedNetwork import BetterSONN

wandb.init(project="test-project", entity="dpis-disciples")

# Loading MNIST data
x_train, t_train, x_test, t_test = mnist.load()

x_train = torch.from_numpy(x_train)
t_train = torch.from_numpy(t_train)
x_test = torch.from_numpy(x_test)
t_test = torch.from_numpy(t_test)

NUM_EXAMPLES_TRAIN = 100
NUM_EXAMPLES_VAL = 100

# Filters to keep anything at or _above_ the threshold value
# Not that this turns into binary, so small intensities are equivalent to max intensity 
# TODO: Why is this so slow to pre-process these? I should definitely pre-load these...
x_train_1 = torch.where(x_train[:NUM_EXAMPLES_TRAIN] >= 1, 1, 0)
x_train_64 = torch.where(x_train[:NUM_EXAMPLES_TRAIN] >= 64, 1, 0)
x_train_128 = torch.where(x_train[:NUM_EXAMPLES_TRAIN] >= 128, 1, 0)
x_train_192 = torch.where(x_train[:NUM_EXAMPLES_TRAIN] >= 192, 1, 0)

x_test_1 = torch.where(x_test[:NUM_EXAMPLES_VAL] >= 1, 1, 0)
x_test_64 = torch.where(x_test[:NUM_EXAMPLES_VAL] >= 64, 1, 0)
x_test_128 = torch.where(x_test[:NUM_EXAMPLES_VAL] >= 128, 1, 0)
x_test_192 = torch.where(x_test[:NUM_EXAMPLES_VAL] >= 192, 1, 0) 

# import pdb; pdb.set_trace()

# Hyper parameters
NUM_EPOCHS = 300

INPUT_SIZE = 784
OUTPUT_SIZE = 10
NUM_NEURONS_PER_COLUMN = 200
NUM_CONNECTIONS_PER_NEURON = 10
SPIKE_THRESHOLD = 1000
MAX_UPDATE_THRESHOLD = 3000
INITIAL_CONNECTION_WEIGHT = SPIKE_THRESHOLD / 10
POSITIVE_REINFORCE_AMOUNT = 100
NEGATIVE_REINFORCE_AMOUNT = 5
DECAY_AMOUNT = 0 # For now...
PRUNE_WEIGHT = -5
DEVICE = 'cuda'
RANDOM_SEED = 42

# Make sure to set the random seed (should propogate to all other imports of random)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Initializg model for MNIST
# model = SONN(784, 10)
model = BetterSONN(INPUT_SIZE,
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
                    device=DEVICE)

# old_model = SONN(784, 10)

# TODO: Make the hyperparameters settable out here, instead of just
# pulling the numbers from the other files (I'm lazy right now...)
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


for epoch in trange(NUM_EPOCHS):
    correctCountTrain = 0
    train_start = timer()
    num_pos_reinforces = 0
    num_neg_reinforces = 0
    connections_grown = 0
    predictions = [0] * 10
    for i in trange(NUM_EXAMPLES_TRAIN):
        x = x_train_128[i,:].to(DEVICE)
        y = t_train[i].item()

        pred, spikes, pos_reinforcements, neg_reinforcements = model.learn(x, y)
        # pred2, spikes2 = old_model.learn(x.cpu().numpy(), y)

        # if pred != 0:
        #     import pdb; pdb.set_trace()

        num_pos_reinforces += len(pos_reinforcements)
        num_neg_reinforces += len(neg_reinforcements)
        predictions[pred] += 1

        # Counting how many we get correct
        if pred == y:
            correctCountTrain += 1
        # import pdb; pdb.set_trace()

        # print("i:", i, "True:", y, "Prediction:", pred, "Spike Counts:", spikes)
    train_end = timer()

    train_time = train_end - train_start
    avg_train_time_per_example = train_time / NUM_EXAMPLES_TRAIN

    val_start = timer()
    correctCountVal = validation(model, DEVICE, NUM_EXAMPLES_VAL, x_test_128, t_test[:NUM_EXAMPLES_VAL])  
    val_end = timer()

    val_time = val_end - val_start
    avg_val_time_per_example = val_time / NUM_EXAMPLES_VAL

    train_accuracy = correctCountTrain / NUM_EXAMPLES_TRAIN
    val_accuracy = correctCountVal / NUM_EXAMPLES_VAL

    print("Epoch", epoch,
        "train accuracy:", train_accuracy,
        "avg train time:", avg_train_time_per_example,
        "val accuracy:", val_accuracy,
        "avg val time:", avg_val_time_per_example,
        "positive reinforces", num_pos_reinforces,
        "negative reinforces", num_neg_reinforces,
        "\n",
        "precictions:", predictions)

    # Logging to W&B
    wandb.log({
        "epoch": epoch,
        "train_accuracy": correctCountTrain / NUM_EXAMPLES_TRAIN,
        "val_accuracy": correctCountVal / NUM_EXAMPLES_VAL,
        "train_time": train_time,
        "val_time": val_time,
        "positive_reinforces": num_pos_reinforces,
        "negative_reinforces": num_neg_reinforces,
    })