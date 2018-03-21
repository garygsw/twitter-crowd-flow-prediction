'''exptCrowdFlow_SG.py.

Crowd Flow Prediction experiment on the Singapore city.
'''

import logging
import sys
import os
import cPickle as pickle
import time
import numpy as np
import h5py
import warnings
from datetime import datetime
from dataset import load_data


# Input parameters
city_name = 'SG'
ds_name = 'MTCset1' #'VDLset4'  # dataset name
map_height, map_width = 54, 90 # 46, 87  # (23, 44) - 1km, (46, 87) - 500m
len_interval = 60  # 30 minutes per time slot
DATAPATH = 'dataset'
flow_data_fname = '{}_{}_M{}x{}_T{}_InOut.h5'.format(city_name,
                                                     ds_name,
                                                     map_width,
                                                     map_height,
                                                     len_interval)
CACHEDATA = True                                # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE')    # cache path
path_norm = os.path.join(DATAPATH, 'NORM')      # normalization path
T = 24 * 60 / len_interval   # number of time intervals in one day
period_interval = 1          # period interval length (in days)
trend_interval = 7           # period interval length (in days)
nb_flow = 2                  # there are two types of flows: inflow and outflow
days_test = 7 * 4            # number of days from the back as test set
len_test = T * days_test
path_result = 'HIST'                # history path
path_model = 'MODEL'                # model path
path_log = 'LOG'                    # log path
path_predictions = 'PRED'           # predictions path
use_mask = True
warnings.filterwarnings('ignore')

# Make the folders and the respective paths if it does not already exists
if not os.path.isdir(path_result):
    os.mkdir(path_result)
path_results = os.path.join(path_result, ds_name)
if not os.path.isdir(path_result):
    os.mkdir(path_result)
if not os.path.isdir(path_model):
    os.mkdir(path_model)
path_model = os.path.join(path_model, ds_name)
if not os.path.isdir(path_model):
    os.mkdir(path_model)
if not os.path.isdir(path_log):
    os.mkdir(path_log)
path_log = os.path.join(path_log, ds_name)
if not os.path.isdir(path_log):
    os.mkdir(path_log)
if not os.path.isdir(path_predictions):
    os.mkdir(path_predictions)
path_predictions = os.path.join(path_predictions, ds_name)
if not os.path.isdir(path_predictions):
    os.mkdir(path_predictions)
if CACHEDATA:
    if not os.path.isdir(path_cache):
        os.mkdir(path_cache)
    if not os.path.isdir(path_norm):
        os.mkdir(path_norm)
    # Define filename of the cache data file
    mask_info = '_masked' if use_mask else ''
    cache_fname = '{}_{}_M{}x{}_T{}_{}_{}.h5'.format(city_name,
                                                     ds_name,
                                                     map_width,
                                                     map_height,
                                                     len_interval,
                                                     mask_info,
                                                     'baselines')
    cache_fpath = os.path.join(path_cache, cache_fname)
    norm_fname = '{}_{}_Normalizer.pkl'.format(city_name, ds_name)
    norm_fpath = os.path.join(path_norm, norm_fname)

# Define logging parameters
local_time = time.localtime()
log_fname = time.strftime('%Y-%m-%d_%H-%M-%S_', local_time) + 'baselines'
log_fname += '.out'
log_fpath = os.path.join(path_log, log_fname)
fileHandler = logging.FileHandler(log_fpath)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(consoleHandler)


def read_cache(cache_fpath, norm_fpath):
    '''Read the prepared dataset (train and test set prepared).'''
    logging.info('reading %s...' % cache_fpath)
    mmn = pickle.load(open(norm_fpath, 'rb'))
    f = h5py.File(cache_fpath, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in xrange(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    mask = f['mask'].value
    f.close()
    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, \
        timestamp_test, mask


def cache(cache_fpath, X_train, Y_train, X_test, Y_test, external_dim,
          timestamp_train, timestamp_test, mask):
    '''Create cache file for the prepared dataset.'''
    h5 = h5py.File(cache_fpath, 'w')
    h5.create_dataset('num', data=len(X_train))
    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.create_dataset('mask', data=mask)
    h5.close()


def print_elasped(from_ts, title):
    '''helper function to print elasped time.'''
    elasped_seconds = time.time() - from_ts
    logging.info('\nelapsed time (%s): %.3f seconds\n' %
                 (title, elasped_seconds))


def print_header(message):
    '''helper function to print header.'''
    logging.info('=' * 10)
    logging.info(message + '\n')


def main():
    '''main function.'''
    # load data
    print_header('loading data...')
    ts = time.time()
    cache_exists = os.path.exists(cache_fpath)
    norm_exists = os.path.exists(norm_fpath)
    logging.info(os.path.join(DATAPATH, ds_name, flow_data_fname))
    if CACHEDATA and cache_exists and norm_exists:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, \
            timestamp_test, mask = read_cache(cache_fpath, norm_fpath)
        logging.info('loaded %s successfully' % cache_fpath)
    else:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, \
            timestamp_test, mask = load_data(
                datapath=os.path.join(DATAPATH, ds_name),
                flow_data_filename=flow_data_fname,
                T=T,
                len_closeness=1,
                len_period=0,
                len_trend=0,
                period_interval=period_interval,
                trend_interval=trend_interval,
                len_test=len_test,
                norm_name=norm_fpath,
                use_mask=use_mask
            )
        if CACHEDATA:
            cache(cache_fpath, X_train, Y_train, X_test, Y_test, external_dim,
                  timestamp_train, timestamp_test, mask)
    print_elasped(ts, 'loading data')

    if use_mask:
        test_mask = np.tile(mask, [len(Y_test), 1, 1, 1])

    # Persistence model
    print_header("evaluating persistence model...")
    ts = time.time()
    predictions = np.concatenate(([Y_train[-1]], Y_test[:-1]), axis=0)
    logging.info('Predictions shape: ' + str(predictions.shape))
    logging.info('Test shape: ' + str(Y_test.shape))
    predictions_fname = 'persistence_predictions.npy'
    predictions_fpath = os.path.join(path_predictions, predictions_fname)
    np.save(predictions_fpath, predictions)
    if use_mask:
        intermediate = (Y_test[test_mask] - predictions[test_mask]) ** 2
        rmse_norm = intermediate.mean() ** 0.5
    else:
        rmse_norm = ((Y_test - predictions) ** 2).mean() ** 0.5
    logging.info('rmse (norm): %.6f rmse (real): %.6f' %
                 (rmse_norm, rmse_norm * (mmn._max - mmn._min) / 2.))
    print_elasped(ts, 'persitence model prediction')

    # Historical average model
    print_header("evaluating historical average model...")
    ts = time.time()
    def update_historical_averages(timeslots, flow_matrix):
        '''Update the historical averages.'''
        hist_seq = {i: {} for i in range(7)}
        for i, train_t in enumerate(timeslots):
            date, timeslot = train_t.split('_')
            weekday = datetime.strptime(date, '%Y%m%d').weekday()
            train_matrix = flow_matrix[i]
            hist_seq[weekday][timeslot] = hist_seq[weekday].get(timeslot, []) + \
                [train_matrix]
        hist_avg = {i: {} for i in range(7)}
        for weekday, timeslots in hist_seq.iteritems():
            for t, matrix_seq in timeslots.iteritems():
                hist_avg[weekday][t] = np.array(matrix_seq).mean(axis=0)
        return hist_avg
    # enumerate test timestamps and

    predictions = []
    for i, test_t in enumerate(timestamp_test):
        train_timeslots = (timestamp_train[i:], timestamp_test[:i])
        train_timeslots = np.concatenate(train_timeslots, axis=0)
        train_matrix = (Y_train[i:], Y_test[:i])
        train_matrix = np.concatenate(train_matrix, axis=0)
        hist = update_historical_averages(train_timeslots, train_matrix)
        date, timeslot = test_t.split('_')
        weekday = datetime.strptime(date, '%Y%m%d').weekday()
        predictions = predictions + [np.array(hist[weekday][timeslot])]
    predictions = np.array(predictions)
    predictions_fname = 'historical_average_predictions.npy'
    predictions_fpath = os.path.join(path_predictions, predictions_fname)
    np.save(predictions_fpath, predictions)
    logging.info(predictions.shape)
    logging.info(Y_test.shape)
    if use_mask:
        intermediate = (Y_test[test_mask] - predictions[test_mask]) ** 2
        rmse_norm = intermediate.mean() ** 0.5
    else:
        rmse_norm = ((Y_test - predictions) ** 2).mean() ** 0.5
    logging.info('rmse (norm): %.6f rmse (real): %.6f' %
                 (rmse_norm, rmse_norm * (mmn._max - mmn._min) / 2.))
    print_elasped(ts, 'historival average model prediction')

if __name__ == '__main__':
    main()
