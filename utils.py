import numpy as np
from PIL import Image

def intensityFilter(data, threshold):
    if data > threshold:
        return 1
    else:
        return 0


def convertToFullScale(data):
    return data * 255


def render(data, isFullScale=False):
    if isFullScale:
        img = Image.fromarray(vectorizedFullScale(data).reshape(28,28)) # Scaling 1 = 255
    else:
        img = Image.fromarray(data.reshape(28,28)) # Assuming already scaled
    img.show() # Show the image


def validation(model, numExamples, X, y):
    correctCountVal = 0
    for i in range(numExamples):
        pred, spikeCounts = model.forward(X[i])

        if pred == y[i]:
            correctCountVal += 1
    
    return correctCountVal