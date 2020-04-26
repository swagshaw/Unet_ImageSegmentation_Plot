from main import *

import numpy as np
import matplotlib as plt


np.savetxt('train_loss.txt', history.history['loss'])
np.savetxt('train_acc.txt', history.history['accuracy'])

