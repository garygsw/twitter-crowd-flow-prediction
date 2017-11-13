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

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics

from dataset import load_data


np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = 'dataset'
data_filename = 'SG2011Nov_M45x85_T30_InOut.h5'
CACHEDATA = True                                # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE')    # cache path
nb_epoch = 10           # number of epoch at training stage
nb_epoch_cont = 100     # number of epoch at training (cont) stage
batch_size = 32         # batch size
T = 48                  # number of time intervals in one day
lr = 0.0002             # learning rate
len_closeness = 3       # length of closeness dependent sequence
len_period = 1          # length of peroid dependent sequence
len_trend = 1           # length of trend dependent sequence
nb_residual_unit = 2    # number of residual units
period_interval = 1     # period interval length (in days)
trend_interval = 7      # period interval length (in days)

nb_flow = 2             # there are two types of flows: inflow and outflow
days_test = 7 * 1       # number of days from the back as test set
len_test = T * days_test
map_height, map_width = 45, 85  # grid size
path_result = 'RET'             # result path
path_model = 'MODEL'            # model path


# Make the folders of the respective paths if it does not already exists
if not os.path.isdir(path_result):
    os.mkdir(path_result)
if not os.path.isdir(path_model):
    os.mkdir(path_model)
if CACHEDATA and not os.path.isdir(path_cache):
    os.mkdir(path_cache)


def build_model(external_dim):
    ''' Define the model configuration and optimizer, and compiles it.
    '''
    c_conf = (len_closeness, nb_flow, map_height, map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height, map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height, map_width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf,
                     p_conf=p_conf,
                     t_conf=t_conf,
                     external_dim=external_dim,
                     nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


def read_cache(fname):
    ''' Read the prepared dataset (train and test set prepared).
    '''
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))
    f = h5py.File(fname, 'r')
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


def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    ''' Creates cache file for the prepared dataset.
    '''
    h5 = h5py.File(fname, 'w')
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
    # load data
    print("loading data...")
    ts = time.time()
    # Define filename of the data file based on c, p & t parameters
    fname = os.path.join(DATAPATH, 'CACHE', 'SG_C{}_P{}_T{}.h5'.format(len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(fname)
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = load_data(
            datapath=os.path.join(DATAPATH, data_filename),
            T=T,
            nb_flow=nb_flow,
            len_closeness=len_closeness,
            len_period=len_period,
            len_trend=len_trend,
            period_interval=period_interval,
            trend_interval=trend_interval,
            len_test=len_test,
            preprocess_name='preprocessing.pkl',
            meta_data=False,
            meteorol_data=False,
            holiday_data=False
        )
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print("**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

    ts = time.time()
    model = build_model(external_dim)
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    model.load_weights(fname_param)

    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], mmn.inverse_transform(score[1]))) #  * (mmn._max - mmn._min) / 2.))
    ts = time.time()
    score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], mmn.inverse_transform(score[1])))
    print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))

if __name__ == '__main__':
    main()
