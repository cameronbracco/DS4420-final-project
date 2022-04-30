# DS4420-final-project
Final Project for DS4420 - Machine Learning and Data Mining 2

## Report
https://wandb.ai/dpis-disciples/test-project/reports/DS4420-Machine-Learning-2-Validating-and-Investigating-Self-Organizing-Neural-Networks-SONN---VmlldzoxOTIxNzg0?accessToken=k9iqt6gs6ffehgbv4bx9uast9txud8zaoay13x5gst9vx24o2790zaexikl09nn8

## Data Loading

Upon cloning, run `python3 mnist.py` to download and save a local version of the dataset. 

## Running Stuff

NOTE: Before you run the first time, make sure you have run `pip install wandb` and log in
using `wandb login`. Alternatively, you can not log in and run it in offline mode by running `wandb offline`.

To run our model with MNIST, set the dataset name in config.yaml to "mnist".
Run main.py

To run our model with Fashion-MNIST, set the dataset name in config.yaml to "fmnist".
Run main.py

To run our model with CIFAR-10, set the dataset name in config.yaml to "cifar10". 
For CIFAR, look at the cifar config file for the hyper parameter values. 
Run main.py
