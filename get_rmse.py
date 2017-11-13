import numpy as np
import cPickle as pickle
#import math
#import deepst.metrics as metrics
#from keras import backend as K

cv_name = 'cv_set_4'

# Reading in data
predictions = np.load('PREDICTIONS/%s/historical_average_predictions.npy' % cv_name)  # shape: (1344, 2, 46, 87)
true_y = np.load('PREDICTIONS/%s/true_y.npy' % cv_name)

predictions = predictions.astype(np.float64)

print ((true_y - predictions)**2).mean()**0.5

#timestamps = np.load(data_path + '/timestamps.npy')
mmn = pickle.load(open('PREPROCESS/SG_Preprocess_%s' % cv_name, 'rb'))
predictions = mmn.inverse_transform(predictions)
true_y = mmn.inverse_transform(true_y)

print ((true_y - predictions)**2).mean()**0.5

# x = metrics.rmse(predictions, true_y)
# f = K.function([], [x])
# print f([])[0]
#
#                 # w # h # f # t
# print math.sqrt(sum(sum(sum(sum(((predictions - true_y)**2))))) / (48 * 4 * 7 * 2 * 46 * 87))
