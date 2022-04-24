import numpy as np
from PIL import Image

def render(data, isFullScale=False):
    if isFullScale:
        img = Image.fromarray(data.reshape(28,28)) # Assuming already scaled
    else:
        img = Image.fromarray((data * 255).reshape(28,28)) # Scale all 1's to 255
    img.show() # Show the image


def validation(model, device, numExamples, X, Y, n_filter_thresholds):
    correctCountVal = 0
    for i in range(numExamples):
        x = X[i, :].to(device)
        pred, spikeCounts = model.forward(x)

        if pred == Y[i]:
            correctCountVal += 1
    
    return correctCountVal