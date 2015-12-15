import numpy as np


ListofPath = [] #these two are what we need to define by our own
newPath = ''

predictions = [np.load(path) for path in ListofPath]
avg_predictions = np.mean(predictions, axis=0)
np.save(newPath, avg_predictions)
