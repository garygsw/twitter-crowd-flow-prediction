import os
import time
import logging
import cPickle as pickle
from copy import copy
import pandas as pd
import numpy as np
import h5py
from datetime import datetime
import scipy.sparse
from preprocessing import MinMaxNormalization, timestamp2vec, string2timestamp


class STMatrix(object):
    """
    """

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i-1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i-1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            logging.debug(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        ''' Returns the matrix given a timestamp.
        '''
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        ''' Check if the time stamps in depends in valid (in the dataset).
        '''
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7,
                       len_period=3, PeriodInterval=1, len_tweetcount=1,
                       use_tweet_counts=False, aggregate_counts=False):
        """current version
        """
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []

        if use_tweet_counts and aggregate_counts:
            len_tweetcount = 1

        # Generate a list of all look back time stamps
        depends = [range(1, len_tweetcount + 1),
                   range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j
                    for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j
                    for j in range(1, len_trend + 1)]]

        # Finding maximum look back in time per step (for period and trend only)
        i = max(self.T * TrendInterval * len_trend,
                self.T * PeriodInterval * len_period,
                len_closeness)

        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame
                                      for j in depend])

            if Flag is False:  # roll forward until the first prediciton time step
                i += 1
                continue

            x_tc = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                    for j in depends[0]]
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[1]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[2]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                   for j in depends[3]]

            if use_tweet_counts and len_tweetcount > 0:
                x_tc = [matrix[2:, :, :] for matrix in x_tc]
                # remove tweet counts in flows
                x_c = [matrix[:2, :, :] for matrix in x_c]
                x_p = [matrix[:2, :, :] for matrix in x_p]
                x_t = [matrix[:2, :, :] for matrix in x_t]
            if len_closeness > 0:
                if use_tweet_counts and len_tweetcount > 0:
                    x_c += x_tc
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))

            y = self.get_matrix(self.pd_timestamps[i])
            if len_tweetcount > 0:  # remove tweet counts in Y
                y = y[:2, :, :]
            Y.append(y)

            timestamps_Y.append(self.timestamps[i])
            i += 1

        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        return XC, XP, XT, Y, timestamps_Y


class TweetMatrix(STMatrix):
    ''' Class to prepare tweets rolling horizons indexes in the required sequences.
    '''

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        assert len(data) == len(timestamps)
        self.data = data[:-1]              # throw away last one
        self.timestamps = timestamps[1:]   # throw away lag timestamp
        self.T = T
        self.pd_timestamps = string2timestamp(self.timestamps)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7,
                       len_period=3, PeriodInterval=1, len_tweets=1):
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        TI = []
        timestamps_Y = []

        # Generate a list of all look back time stamps
        depend = range(1, len_tweets + 1)

        # Finding maximum look back in time per step
        i = max(self.T * TrendInterval * len_trend,
                self.T * PeriodInterval * len_period,
                len_closeness)

        while i < len(self.pd_timestamps):
            Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame
                                  for j in depend])
            if Flag is False:
                i += 1
                continue

            t_ti = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame)
                    for j in depend]

            if len_closeness > 0:
                if len_tweets > 0:
                    TI.append(scipy.sparse.vstack(t_ti))

            timestamps_Y.append(self.timestamps[i])
            i += 1

        TI = np.asarray(TI)
        return TI, timestamps_Y, TI[0].shape


def stat(fname, T):
    def get_nb_timeslot(f):
        s = f['date'][0]   # start date
        e = f['date'][-1]  # end date
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = ((time.mktime(te) - time.mktime(ts)) / 86400 + 1) * T
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    with h5py.File(fname) as f:
        nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / T)
        mmax = f['data'].value.max()
        mmin = f['data'].value.min()
        single_mask = f['mask'].value
        if single_mask is not None:
            inflow_cells = np.sum(single_mask[0])
            outflow_cells = np.sum(single_mask[1])
            mask_info = '# of valid inflow cells: %i\n' % inflow_cells + \
                        '# of valid outflow cells: %i\n' % outflow_cells
        else:
            mask_info = ''
        stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + mask_info + \
               '=' * 5 + 'stat' + '=' * 5 + '\n'
        logging.info(stat)


def load_holiday(timeslots, datapath):
    f = open(datapath, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    logging.info("total number of holidays/weekends: " + str(H.sum()))
    return H[:, None]


def load_weather(timeslots, datapath):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot.
    Instead, we use the meteoral at previous timeslots
    i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    h5 file, which has four following subsets:
    date: a list of timeslots, which is associated the following kinds of data.
    Temperature: a list of continuous value, of which the i'th value is temperature at the timeslot date[i].
    WindSpeed: a list of continuous value, of which the i'th value is wind speed at the timeslot date[i].
    Weather: a 2D matrix, each of which is a one-hot vector (dim=8), showing one of the following weather types:
    sunny = 0
    cloudy = 1
    overcast = 2
    rain = 3
    light rain = 4
    heavy rain = 5
    fog = 6
    haze = 7
    '''
    f = h5py.File(datapath, 'r')
    Timeslot = f['date'].value
    WindSpeed = f['windspeed'].value
    Weather = f['weather'].value
    Temperature = f['temperature'].value
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # windspeed
    WR = []  # weather
    TE = []  # temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    logging.info("windspeed shape: " + str(WS.shape))
    logging.info("weather shape: " + str(WR.shape))
    logging.info("temperature shape: " + str(TE.shape))

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    logging.info('merged shape: ' + str(merge_data.shape))
    return merge_data


def read_count_data(count_name, datapath, counts_norm, flow_timestamps, aggregate_counts,
                    len_lag, len_lead, data_lag, data_lead):
    # Load tweet count data tile
    logging.info('reading counts data for %s...' % count_name)
    if not os.path.exists(datapath):
        raise Exception('counts input data path "%s" does not exists' % datapath)
    f = h5py.File(datapath, 'r')
    assert(flow_timestamps[0] == f['date'].value[data_lag])
    assert(flow_timestamps[-1] == f['date'].value[-data_lead - 1])
    counts_data = f['count'].value
    within_window_counts = counts_data[data_lag - 1:-data_lead - 1]
    assert(len(within_window_counts) == len(flow_timestamps))
    f.close()

    # Aggregate the counts
    logging.info('aggregating counts for %s...' % count_name)
    if aggregate_counts:
        for i, timestamp in enumerate(flow_timestamps):
            # add up the lags
            if len_lag:
                if len_lag < data_lag + i:
                    # now data_lag is always 1 (no need further back)
                    pass  # ignore those with insufficient history data
                elif len_lag > 1:
                    for j in range(1, len_lag):
                        within_window_counts[i] += counts_data[i - j]
            # add up the leads
            if len_lead > 0:
                for j in range(len_lead):
                    within_window_counts[i] += counts_data[i + j + len_lag]

    # Normalize counts
    logging.info('normalizing counts for %s...' % count_name)
    # tweet_count: (# of timeslots, h, w)
    if counts_norm == 'day+time':
        hist_seq = {i: {} for i in range(7)}
        for i, timestamp in enumerate(flow_timestamps):
            date, timeslot = timestamp.split('_')
            weekday = datetime.strptime(date, '%Y%m%d').weekday()
            count_matrix = within_window_counts[i]
            hist_seq[weekday][timeslot] = hist_seq[weekday].get(
                timeslot, []) + [count_matrix]
        hist_max = {i: {} for i in range(7)}
        hist_min = {i: {} for i in range(7)}
        for weekday, timeslots in hist_seq.iteritems():
            for t, matrix_seq in timeslots.iteritems():
                hist_max[weekday][t] = np.array(matrix_seq).max(axis=0)
                hist_min[weekday][t] = np.array(matrix_seq).min(axis=0)
        for i, timestamp in enumerate(flow_timestamps):
            date, timeslot = timestamp.split('_')
            weekday = datetime.strptime(date, '%Y%m%d').weekday()
            numerator = within_window_counts[i] - hist_min[weekday][timeslot]
            denominator = hist_max[weekday][timeslot] - hist_min[weekday][timeslot]
            within_window_counts[i] = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0
            )
    elif counts_norm == 'all':
        max_counts = within_window_counts.max(axis=0)
        min_counts = within_window_counts.min(axis=0)
        denominator = max_counts - min_counts
        for i, timestamp in enumerate(flow_timestamps):
            numerator = within_window_counts[i] - min_counts
            within_window_counts[i] = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0
            )

    return within_window_counts * 2 - 1


def load_data(datapath, flow_data_filename=None, T=48,
              len_closeness=None, len_period=None, len_trend=None,
              len_lag_tweets=None, len_lead_tweets=None,
              period_interval=1, trend_interval=7,
              use_mask=False, len_test=None, norm_name=None, meta_data=False,
              weather_data=False, holiday_data=False,
              tweet_count_data=False, future_count_data=False,
              past_count_data=False, present_count_data=False, positive_count_data=False,
              negative_count_data=False,
              tweet_index_data=False,
              tweet_count_data_filename=None,
              future_count_data_filename=None,
              past_count_data_filename=None,
              present_count_data_filename=None,
              positive_count_data_filename=None,
              negative_count_data_filename=None,
              aggregate_counts=False,
              counts_norm=None, tweet_index_data_filename=None,
              weather_data_filename=None, holiday_data_filename=None,
              tweet_lag=1, tweet_lead=0):
    assert(len_closeness + len_period + len_trend > 0)
    # Load the h5 file and retrieve data
    flow_data_path = os.path.join(datapath, flow_data_filename)
    if not os.path.exists(flow_data_path):
        raise Exception('flow input data path "%s" does not exists' % flow_data_path)
    logging.info('reading flow data...')
    stat(flow_data_path, T)
    f = h5py.File(flow_data_path, 'r')
    flow_data = f['data'].value
    timestamps = f['date'].value
    mask = f['mask'].value if use_mask else None
    f.close()

    # minmax_scale
    data_train = copy(flow_data)[:-len_test]
    logging.info('flows training data shape: ' + str(data_train.shape))
    mmn = MinMaxNormalization()
    logging.info('normalizing flows training data...')
    mmn.fit(data_train)
    data_mmn = np.array([mmn.transform(d) for d in flow_data])

    # Load count
    if tweet_count_data:
        len_tweets = 0
        if len_lag_tweets is not None:
            len_tweets += len_lag_tweets
        if len_lead_tweets is not None:
            len_tweets += len_lead_tweets
        tweet_counts = read_count_data(count_name='tweet_counts',
                                       datapath=os.path.join(datapath, tweet_count_data_filename),
                                       counts_norm=counts_norm,
                                       flow_timestamps=timestamps,
                                       aggregate_counts=aggregate_counts,
                                       len_lag=len_lag_tweets,
                                       len_lead=len_lead_tweets,
                                       data_lag=tweet_lag,
                                       data_lead=tweet_lead)
        # Insert the tweet counts dimension
        data_mmn = np.insert(data_mmn, 2, tweet_counts, axis=1)
    if future_count_data:
        future_counts = read_count_data(count_name='future_counts',
                                        datapath=os.path.join(datapath, future_count_data_filename),
                                        counts_norm=counts_norm,
                                        flow_timestamps=timestamps,
                                        aggregate_counts=aggregate_counts,
                                        len_lag=len_lag_tweets,
                                        len_lead=len_lead_tweets,
                                        data_lag=tweet_lag,
                                        data_lead=tweet_lead)
        # Insert the future counts dimension
        data_mmn = np.insert(data_mmn, 2, future_counts, axis=1)
    if past_count_data:
        past_counts = read_count_data(count_name='past_counts',
                                      datapath=os.path.join(datapath, past_count_data_filename),
                                      counts_norm=counts_norm,
                                      flow_timestamps=timestamps,
                                      aggregate_counts=aggregate_counts,
                                      len_lag=len_lag_tweets,
                                      len_lead=len_lead_tweets,
                                      data_lag=tweet_lag,
                                      data_lead=tweet_lead)
        # Insert the past counts dimension
        data_mmn = np.insert(data_mmn, 2, past_counts, axis=1)
    if present_count_data:
        present_counts = read_count_data(count_name='present_counts',
                                      datapath=os.path.join(datapath, present_count_data_filename),
                                      counts_norm=counts_norm,
                                      flow_timestamps=timestamps,
                                      aggregate_counts=aggregate_counts,
                                      len_lag=len_lag_tweets,
                                      len_lead=len_lead_tweets,
                                      data_lag=tweet_lag,
                                      data_lead=tweet_lead)
        # Insert the present counts dimension
        data_mmn = np.insert(data_mmn, 2, present_counts, axis=1)
    if positive_count_data:
        positive_counts = read_count_data(count_name='positive_counts',
                                          datapath=os.path.join(datapath, positive_count_data_filename),
                                          counts_norm=counts_norm,
                                          flow_timestamps=timestamps,
                                          aggregate_counts=aggregate_counts,
                                          len_lag=len_lag_tweets,
                                          len_lead=len_lead_tweets,
                                          data_lag=tweet_lag,
                                          data_lead=tweet_lead)
        # Insert the positive counts dimension
        data_mmn = np.insert(data_mmn, 2, positive_counts, axis=1)
    if negative_count_data:
        negative_counts = read_count_data(count_name='negative_counts',
                                          datapath=os.path.join(datapath, negative_count_data_filename),
                                          counts_norm=counts_norm,
                                          flow_timestamps=timestamps,
                                          aggregate_counts=aggregate_counts,
                                          len_lag=len_lag_tweets,
                                          len_lead=len_lead_tweets,
                                          data_lag=tweet_lag,
                                          data_lead=tweet_lead)
        # Insert the negative counts dimension
        data_mmn = np.insert(data_mmn, 2, negative_counts, axis=1)

    # save preprocessing stats
    fpkl = open(norm_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    # instance-based dataset --> sequences with format as (X, Y) where X is
    # a sequence of images and Y is an image.
    st = STMatrix(data_mmn, timestamps, T, CheckComplete=False)
    logging.info('creating training and tests flow inputs...')
    XC, XP, XT, Y, timestamps_Y = st.create_dataset(
        len_closeness=len_closeness,
        len_period=len_period,
        len_trend=len_trend,
        len_tweetcount=len_tweets,
        PeriodInterval=period_interval,
        TrendInterval=trend_interval,
        use_tweet_counts=tweet_count_data,
        aggregate_counts=aggregate_counts
    )
    logging.info("XC shape: " + str(XC.shape))
    logging.info("XP shape: " + str(XP.shape))
    logging.info("XT shape: " + str(XT.shape))
    logging.info("Y shape: " + str(Y.shape))

    if tweet_index_data and len_tweets is not None:
        logging.info('reading tweet index data...')
        tweet_index_data_path = os.path.join(datapath, tweet_index_data_filename)
        if not os.path.exists(tweet_index_data_path):
            raise Exception('tweet index input data path "%s" does not exists' % tweet_index_data_path)
        #f = h5py.File(tweet_index_data_path, 'r')
        #index_data = f['index'].value
        #timestamps = f['date'].value
        logging.info('reading tweet index data...')
        index_data = np.load(tweet_index_data_path)
        timestamps = index_data.keys()
        timestamps = sorted(timestamps)
        tweet_index_values = [index_data[k].tolist() for k in timestamps]

        #timestamps, index_data = pickle.load(open(tweet_index_data_path, 'rb'))
        # max_vocab = tweet_index.attrs['vocab_size']
        # m = tweet_index.attrs['m']
        #f.close()

        tm = TweetMatrix(tweet_index_values, timestamps, T, CheckComplete=False)
        TI, T_timestamps, TI_shape, = tm.create_dataset(
            len_closeness=len_closeness,
            len_period=len_period,
            len_trend=len_trend,
            len_tweets=len_tweets,
            PeriodInterval=period_interval,
            TrendInterval=trend_interval
        )
        assert(T_timestamps[0] == timestamps_Y[0])
        assert(T_timestamps[-1] == timestamps_Y[-1])
        logging.info("TI shape: " + str(TI.shape + TI_shape))

    # Segment the training set
    XC_train = XC[:-len_test]
    XP_train = XP[:-len_test]
    XT_train = XT[:-len_test]
    Y_train = Y[:-len_test]
    timestamp_train = timestamps_Y[:-len_test]
    logging.info('train set XC shape: ' + str(XC_train.shape))
    logging.info('train set XP shape: ' + str(XP_train.shape))
    logging.info('train set XT shape: ' + str(XT_train.shape))
    logging.info('train set Y shape' + str(Y_train.shape))
    if tweet_index_data:
        TI_train = TI[:-len_test]
        # TP_train = TP[:-len_test]
        # TT_train = TT[:-len_test]
        logging.info('train set TI shape: ' + str(TI_train.shape + TI_shape))
        #logging.info('train set TP shape: ' + str(TP_train.shape))
        #logging.info('train set TT shape: ' + str(TT_train.shape))

    # Segment the test set
    XC_test = XC[-len_test:]
    XP_test = XP[-len_test:]
    XT_test = XT[-len_test:]
    Y_test = Y[-len_test:]
    timestamp_test = timestamps_Y[-len_test:]
    logging.info('test set XC shape: ' + str(XC_test.shape))
    logging.info('test set XP shape: ' + str(XP_test.shape))
    logging.info('test set XT shape: ' + str(XT_test.shape))
    logging.info('test set Y shape: ' + str(Y_test.shape))
    if tweet_index_data:
        TI_test = TI[-len_test:]
        # TP_test = TP[-len_test:]
        # TT_test = TT[-len_test:]
        # logging.info('test set TC shape: ' + str(TC_test.shape))
        # logging.info('test set TP shape: ' + str(TP_test.shape))
        logging.info('test set TI shape: ' + str(TI_test.shape + TI_shape))

    # Prepare the external component
    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_data_path = os.path.join(datapath, holiday_data_filename)
        holiday_feature = load_holiday(timestamps_Y, holiday_data_path)
        meta_feature.append(holiday_feature)
    if weather_data:
        # load weather data
        weather_data_path = os.path.join(datapath, weather_data_filename)
        weather_feature = load_weather(timestamps_Y, weather_data_path)
        meta_feature.append(weather_feature)
    if len(meta_feature) > 0:
        meta_feature = np.hstack(meta_feature)
    else:
        meta_feature = np.asarray(meta_feature)

    if len(meta_feature.shape) > 1:
        metadata_dim = meta_feature.shape[1]
    else:
        metadata_dim = None
    if metadata_dim < 1:
        metadata_dim = None
    if meta_data:
        logging.info('time feature shape: ' + str(time_feature.shape))
    if meta_data and holiday_data:
        logging.info('holiday feature shape: ' + str(holiday_feature.shape))
    if meta_data and weather_data:
        logging.info('weather feature shape: ' + str(weather_feature.shape))
    logging.info('meta feature shape: ' + str(meta_feature.shape))

    # Combining the datasets into a list
    X_train = []
    X_test = []
    train_datasets_list = zip([len_closeness, len_period, len_trend],
                              [XC_train, XP_train, XT_train])
    test_datasets_list = zip([len_closeness, len_period, len_trend],
                             [XC_test, XP_test, XT_test])

    for l, X_ in train_datasets_list:
        if l > 0:
            X_train.append(X_)
    if tweet_index_data:
        X_train.append(TI_train)

    for l, X_ in test_datasets_list:
        if l > 0:
            X_test.append(X_)
    if tweet_index_data:
        X_test.append(TI_test)

    if metadata_dim is not None:
        meta_feature_train = meta_feature[:-len_test]
        meta_feature_test = meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    # type_map = {0: 'closeness', 1: 'period', 2: 'trend', 3: 'meta'}
    for i, _X in enumerate(X_train):
        logging.info('X train shape at index %s ' % i + ': ' + str(_X.shape))
    for i, _X in enumerate(X_test):
        logging.info('X test shape at index %s ' % i + ': ' + str(_X.shape))

    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test, mask
