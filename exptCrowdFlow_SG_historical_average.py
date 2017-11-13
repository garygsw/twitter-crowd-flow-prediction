# -*- coding: utf-8 -*-
"""
Usage:
    THEANO_FLAGS="device=gpu0" python exptCrowdFlow.py
"""
from __future__ import print_function
import os
import sys
import cPickle as pickle
import time
import numpy as np
import h5py
from datetime import datetime

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics

from dataset import load_data


np.random.seed(1337)  # for reproducibility

# parameters
cv_set_dirs = ['cv_set_1', 'cv_set_2', 'cv_set_3', 'cv_set_4']                # CHANGE: name
map_height, map_width = 46, 87          # CHANGE: grid size (23, 44) - 1km, (46, 87) - 500m
use_meta = True
use_weather = True
use_holidays = True
len_timeslot = 30       # 30 minutes per time slot
DATAPATH = 'dataset'

CACHEDATA = True                                # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE')    # cache path
path_preprocess = os.path.join(DATAPATH, 'PREPROCESS')
nb_epoch = 500          # number of epoch at training stage
nb_epoch_cont = 100     # number of epoch at training (cont) stage
batch_size = 32         # batch size
T = 48                  # number of time intervals in one day
lr = 0.0002             # learning rate
len_closeness = 4       # length of closeness dependent sequence
len_period = 1          # length of peroid dependent sequence
len_trend = 1           # length of trend dependent sequence
nb_residual_unit = 2    # number of residual units
period_interval = 1     # period interval length (in days)
trend_interval = 7      # period interval length (in days)

nb_flow = 2             # there are two types of flows: inflow and outflow
days_test = 7 * 4       # number of days from the back as test set
len_test = T * days_test
path_result = 'RET'             # result path
path_model = 'MODEL'            # model path


# Make the folders of the respective paths if it does not already exists
if not os.path.isdir(path_result):
    os.mkdir(path_result)
if not os.path.isdir(path_model):
    os.mkdir(path_model)
if CACHEDATA and not os.path.isdir(path_cache):
    os.mkdir(path_cache)
if CACHEDATA and not os.path.isdir(path_preprocess):
    os.mkdir(path_preprocess)


def read_cache(flow_fname, preprocess_fname):
    ''' Read the prepared dataset (train and test set prepared).
    '''
    mmn = pickle.load(open(preprocess_fname, 'rb'))
    f = h5py.File(flow_fname, 'r')
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
    f.close()
    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test


def cache(flow_fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    ''' Creates cache file for the prepared dataset.
    '''
    h5 = h5py.File(flow_fname, 'w')
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
    h5.close()


def main():
    for cv_set_name in cv_set_dirs:
        # load data
        print("loading data...")
        flow_data_filename = 'SG_{}_M{}x{}_T{}_InOut.h5'.format(cv_set_name, map_width, map_height, len_timeslot)  # map grid dependent, time dependent
        weather_data_filename = 'SG_{}_T{}_Weather.h5'.format(cv_set_name, len_timeslot)    # map grid independent, time dependent
        holiday_data_filename = 'SG_{}_Holidays.txt'.format(cv_set_name)                    # map grid independent, time independent
        ts = time.time()
        meta_info = []
        if use_meta and use_weather:
            meta_info.append('W')
        if use_meta and use_holidays:
            meta_info.append('H')
        if len(meta_info) > 1:
            meta_info = '_' + '_'.join(meta_info)
        else:
            meta_info = ''
        # Define filename of the data file (for CACHE) based on c, p & t parameters
        flow_fname = os.path.join(DATAPATH, 'CACHE', 'SG_{}_M{}x{}_T{}_C{}_P{}_T{}{}.h5'.format(
            cv_set_name, map_width, map_height, len_timeslot, len_closeness, len_period, len_trend, meta_info)
        )  # map grid dependent, time dependent, param dependent
        preprocess_fname = os.path.join(DATAPATH, 'PREPROCESS', 'SG_Preprocess_{}'.format(cv_set_name))
        if os.path.exists(flow_fname) and os.path.exists(preprocess_fname) and CACHEDATA:
            X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
                flow_fname,
                preprocess_fname
            )
            print("load %s successfully" % flow_fname)
        else:
            X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = load_data(
                datapath=DATAPATH,
                flow_data_filename=flow_data_filename,
                T=T,
                nb_flow=nb_flow,
                len_closeness=len_closeness,
                len_period=len_period,
                len_trend=len_trend,
                period_interval=period_interval,
                trend_interval=trend_interval,
                len_test=len_test,
                preprocess_name=preprocess_fname,
                meta_data=use_meta,
                weather_data=use_weather,
                holiday_data=use_holidays,
                weather_data_filename=weather_data_filename,
                holiday_data_filename=holiday_data_filename
            )
            if CACHEDATA:
                cache(flow_fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test)

        print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
        print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

        # Getting historical averages in training set
        hist_seq = {i: {} for i in range(7)}
        for i, train_t in enumerate(timestamp_train):
            date, timeslot = train_t.split('_')
            weekday = datetime.strptime(date, '%Y%m%d').weekday()
            train_matrix = Y_train[i]
            hist_seq[weekday][timeslot] = hist_seq[weekday].get(timeslot, []) + [train_matrix]
        hist_avg = {i: {} for i in range(7)}
        for weekday, timeslots in hist_seq.iteritems():
            for t, matrix_seq in timeslots.iteritems():
                hist_avg[weekday][t] = np.array(matrix_seq).mean(axis=0)

        predictions = []
        for i, test_t in enumerate(timestamp_test):
            date, timeslot = train_t.split('_')
            weekday = datetime.strptime(date, '%Y%m%d').weekday()
            predictions = predictions + [np.array(hist_avg[weekday][timeslot])]
        predictions = np.array(predictions)
        np.save('PREDICTIONS/' + cv_set_name + '/historical_average_predictions.npy', predictions)

if __name__ == '__main__':
    main()
