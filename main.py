import mnist
import random
import numpy as np
import wandb

from utils import intensityFilter, convertToFullScale, render, validation
from Network import SONN

wandb.init(project="test-project", entity="dpis-disciples")

# Initializg model for MNIST (784 flat input, 10 class output)
model = SONN(784, 10)

# Loading MNIST data
x_train, t_train, x_test, t_test = mnist.load()

# Preprocess the data
vectorizedFilter = np.vectorize(intensityFilter, otypes=[np.uint8])
vectorizedFullScale = np.vectorize(convertToFullScale, otypes=[np.uint8])

NUM_EXAMPLES_TRAIN = 100
NUM_EXAMPLES_VAL = 100

# Filters to keep anything at or _above_ the threshold value
# Not that this turns into binary, so small intensity becomes max possibly 
# TODO: Why is this so slow to pre-process these? I should definitely pre-load these...
x_train_1 = vectorizedFilter(x_train[:NUM_EXAMPLES_TRAIN], 0)
x_train_64 = vectorizedFilter(x_train[:NUM_EXAMPLES_TRAIN], 63)
x_train_128 = vectorizedFilter(x_train[:NUM_EXAMPLES_TRAIN], 127)
x_train_192 = vectorizedFilter(x_train[:NUM_EXAMPLES_TRAIN], 191)

x_test_1 = vectorizedFilter(x_test[:NUM_EXAMPLES_VAL], 0)
x_test_64 = vectorizedFilter(x_test[:NUM_EXAMPLES_VAL], 63)
x_test_128 = vectorizedFilter(x_test[:NUM_EXAMPLES_VAL], 127)
x_test_192 = vectorizedFilter(x_test[:NUM_EXAMPLES_VAL], 191) 

NUM_EPOCHS = 30

# TODO: Make the hyperparameters settable out here, instead of just
# pulling the numbers from the other files (I'm lazy right now...)
wandb.config = {
  "epochs": NUM_EPOCHS,
  "num_examples": NUM_EXAMPLES_TRAIN,
  "num_neurons_per_column": 200,
  "num_connections_per_neuron": 10,
  "spike_threshold": 1000,
  "max_threshold": 3000,
  "inital_connection_weight": 100, # spike_threshold / 10,
  "positive_reinforce_amount": 100,
  "negative_reinforce_amount": 5, # Prob too low...
  "decay_amount": 0, # For now...
  "pruneWeight": -5
}


for epoch in range(NUM_EPOCHS):
    correctCountTrain = 0
    for i in range(NUM_EXAMPLES_TRAIN):
        x = x_train_128[i,:]
        y = t_train[i]
        pred, spikes = model.learn(x, y)
        
        # Counting how many we get correct
        if pred == y:
            correctCountTrain += 1
        
        # print("i:", i, "True:", y, "Prediction:", pred, "Spike Counts:", spikes)


    correctCountVal = validation(model, NUM_EXAMPLES_VAL, x_test_128, t_test[:NUM_EXAMPLES_VAL])  

    train_accuracy = correctCountTrain / NUM_EXAMPLES_TRAIN
    val_accuracy = correctCountVal / NUM_EXAMPLES_VAL

    print("Epoch", epoch, "train:", train_accuracy, "val:", val_accuracy)

    # Logging to W&B
    wandb.log({
        "epoch": epoch,
        "train_accuracy": correctCountTrain / NUM_EXAMPLES_TRAIN,
        "val_accuracy": correctCountVal / NUM_EXAMPLES_VAL,
    })