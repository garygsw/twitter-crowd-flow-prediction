# -*- coding: utf-8 -*-
"""
Usage:
    THEANO_FLAGS="device=gpu0" python exptCrowdFlow.py
"""
from __future__ import print_function
import os
import cPickle as pickle
import time
import numpy as np
import h5py
from statsmodels.tsa.arima_model import ARIMA
from dataset import load_data
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

np.random.seed(1337)  # for reproducibility

# parameters
cv_set_dirs = ['cv_set_1', 'cv_set_2', 'cv_set_3', 'cv_set_4']                # CHANGE: name
map_height, map_width = 46, 87          # CHANGE: grid size (23, 44) - 1km, (46, 87) - 500m
use_meta = False
use_weather = False
use_holidays = False
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

        order = (0, 1, 0)

        predictions = np.zeros((len(Y_test), 2, map_height, map_width))
        total_test = len(Y_test)
        for i, y_matrix in enumerate(Y_test):
            ts = time.time()
            print('\npredicting for timeslot ({}/{},  {}% completed)'.format(str(i), str(total_test), str(i/float(total_test)*100)))
            training_matrixes = np.concatenate((Y_train[i:], Y_test[:i]), axis=0)
            inflow_matrixes = training_matrixes[:, 0, :, :]
            outflow_matrixes = training_matrixes[:, 1, :, :]
            for i in range(map_height):
                for j in range(map_width):
                    #print('\npredicting for map (%s,%s)' % (i, j))
                    inflow_model = ARIMA(inflow_matrixes[:, i, j], order=order)
                    inflow_model_fit = inflow_model.fit(disp=0)
                    inflow_prediction = inflow_model_fit.forecast()
                    predictions[i][0][i][j] = inflow_prediction[0]

                    outflow_model = ARIMA(outflow_matrixes[:, i, j], order=order)
                    outflow_model_fit = outflow_model.fit(disp=0)
                    outflow_prediction = outflow_model_fit.forecast()
                    predictions[i][1][i][j] = outflow_prediction[0]
            print("\nelapsed time (prediction): %.3f seconds\n" % (time.time() - ts))

        np.save('PREDICTIONS/' + cv_set_name + '/ARIMA_predictions.npy', predictions)

if __name__ == '__main__':
    main()
