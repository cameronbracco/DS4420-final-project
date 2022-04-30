# DS4420-final-project
Final Project for DS4420 - Machine Learning and Data Mining 2


## Data Loading

Upon cloning, run `python3 mnist.py` to download and save a local version of the dataset. 

## Running Stuff

NOTE: Before you run the first time, make sure you have run `pip install wandb` and log in
using `wandb login` 

If you do not want to sync data to the cloud, run `wandb offline`. This can then be toggled again
using `wandb online`. You probably want to run in offline mode. 

To run our model with MNIST, set the dataset name in config.yaml to "mnist".
Run main.py

To run our model with Fashion-MNIST, set the dataset name in config.yaml to "fmnist".
Run main.py

To run our model with CIFAR-10, set the dataset name in config.yaml to "cifar10". 
For CIFAR, look at the cifar config file for the hyper parameter values. 
Run main.py
