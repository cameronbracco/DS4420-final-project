# DS4420-final-project
Final Project for DS4420 - Machine Learning and Data Mining 2


## Data Loading

MNIST is loaded using this extremely helpful script: https://github.com/hsjeong5/MNIST-for-Numpy

Upon cloning, run `python3 mnist.py` to download and save a local version of the dataset. 
One initialized (it will be saved in `mnist.pkl`), you can load the data as follows:
```python
x_train, t_train, x_test, t_test = mnist.load()
```

Note that we need to further apply some filters to this data, but that's easy to handle.
However, it seems like applying those filters takes a long time for some reason, so it
may make more sentence to bake that into the pre-loading (or just figure out how to optimize
the filters because they shouldn't take as long as they do)


## Metrics

I'm doing metrics with Weights & Biases because it makes everything super simple. You
basically just need to create an account, get added to the project, and login using the CLI
to be able to automatically log metrics for each run (also you can disable sync if you're
debugging and we might want to make that even easier...). Here's basic tutorial:
https://docs.wandb.ai/quickstart

Project on W&B lives here: https://wandb.ai/dpis-disciples/test-project


## Model
I first diagramed out the model [here](https://drive.google.com/file/d/1ms71mk4eImrzkVSPj1iwx5C0kbbimDr3/view?usp=sharing)

The current iteration is designed for clarity. I've made a whole bunch of unnecessary classes
(since as we know most of this can just be done with matrices), but I wanted to really
build it all out with explicitly classes and see what I could get it to do to start.

Importantly, it's able to learn something!!! After just 30 iterations of 100 examples
it's able to achieve essentially 50% accuracy on a held out validation set (also of 100 examples).

See this report from the run:
https://wandb.ai/dpis-disciples/test-project/reports/Day-1-Proof-of-Concept--VmlldzoxODUxMjUy


## Running Stuff

NOTE: Before you run the first time, make sure you have run `pip install wandb` and logged in
using `wandb login` 

Once logged in with `wandb` running should be as simple as `python3 main.py`

If you do not want to sync data to the cloud, run `wandb offline`. This can then be toggled again
using `wandb online`. 


## Improvements
Many, many many

1. Moving pre-filter stuff into the pkl or figuring out why it's so slow (so we can run with larger example sizes)

2. One big thing right now is that I set the decay amount to 0 because I don't think my other hyperparameters are set well enough to allow it to get started and really learn otherwise. Hyperparmeters are gonna be hard here.

3. Another is that it's way too slow to even just evaluate the model. I'm assuming this is because of the way I have the neurons/connections/receptors coded. We should figure out another way to do this was is much fast but still clear what's going on. I'm guessing (but haven't confirmed) that the majority of time is spent in the `forward` function. If we can optimize the model that will make doing other stuff much easier.

4. I'd also like to make it so all hyperparameters are set at the absolute top level, instead of the weird mix that I have right now.

5. Actually using other filters besides 128 (and figuring out what he meant by "priming" in one of his posts)

6. Style isn't the greatest...



