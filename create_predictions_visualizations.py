'''create_predictions_visualizations.py.

Plot visualizations of crowd flows predictions (both inflows and outflows) in
the grid format using a heat map.
'''

import math
import os
import numpy as np
import cPickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

# Input parameters
ds_names = ['VDLset1', 'VDLset2', 'VDLset3', 'VDLset4']
city_name = 'SG'
grid = '500m'
w, h = 87, 46  # for 500m, (44, 23) for 1km
use_log = True
log_base = 10
len_interval = 30  # 30 minutes per time slot
len_test = 7 * 4 * ((60 * 24) // 30)
model_name = 'stressnet'
params = '%s_%s_M%sx%s_T%s_c4.p1.t1_Ext.W.H_masked_resunit2_lr0.0002'
# for city name, ds name, w, h, time interval
input_path = 'PRED'
output_path = 'prediction-visualizations'

# Define file paths
if not os.path.exists(output_path):
    os.mkdir(output_path)
prediction_fname = params + '.' + model_name + '.predictions.npy'
timestamps_fname = 'timestamps_T%s_%s.npy' % (len_interval, len_test)
timestamps_fpath = os.path.join(input_path, '%s', timestamps_fname)
true_y_fname = 'true_y_T%s_%s.npy' % (len_interval, len_test)
true_y_fpath = os.path.join(input_path, '%s', true_y_fname)
norm_path = 'dataset/NORM'
normalizer_fname = '%s_%s_Normalizer.pkl'   # for city name, ds name
normalizer_fpath = os.path.join(norm_path, normalizer_fname)
max_flow_cache_dir = 'max-flows-cache'
if not os.path.exists(max_flow_cache_dir):
    os.mkdir(max_flow_cache_dir)
cache_fname = 'max_flow_%s_%s.pkl'  # for ds name and grid
cache_fpath = os.path.join(max_flow_cache_dir, cache_fname)

# Define color map
cmap = plt.cm.RdYlGn  # Red Yellow Green
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[::-1]
# cmaplist[0] = (0, 0, 0, 0.13)  # force the first color to be white
cmap = cmap.from_list('discrete_RdYlGn', cmaplist, cmap.N)
bin_size = 20


# Creating timestamp blocks
block_size = 6   # 6 * 5 mins = 30 minutes block
all_timestamps = [str(hour).zfill(2) + str(minute).zfill(2)
                  for hour in range(24)
                  for minute in range(0, 60, 5)]
len_timestamps = len(all_timestamps)
timestamps_groups = [all_timestamps[i]
                     for i in range(0, len_timestamps, block_size)]


for ds_name in ds_names:
    prediction_filename = prediction_fname % (city_name,
                                              ds_name,
                                              w,
                                              h,
                                              len_interval)
    prediction_filepath = os.path.join(input_path,
                                       ds_name,
                                       prediction_filename)
    timestamps_filepath = timestamps_fpath % (ds_name)
    normalizer_filepath = normalizer_fpath % (city_name, ds_name)

    # Reading in data
    predictions = np.load(prediction_filepath)  # shape: (1344, 2, 46, 87)
    timestamps = np.load(timestamps_filepath)
    mmn = pickle.load(open(normalizer_filepath, 'rb'))
    predictions = mmn.inverse_transform(predictions)  # inverse to real values

    # create output directories
    output_fpath = os.path.join(output_path, ds_name)
    if not os.path.exists(output_fpath):
        os.mkdir(output_fpath)
    output_fpath = os.path.join(output_fpath, grid)
    if not os.path.exists(output_fpath):
        os.mkdir(output_fpath)
    inflow_output_fpath = os.path.join(output_fpath, 'inflow')
    if not os.path.exists(inflow_output_fpath):
        os.mkdir(inflow_output_fpath)
    outflow_output_fpath = os.path.join(output_fpath, 'outflow')
    if not os.path.exists(outflow_output_fpath):
        os.mkdir(outflow_output_fpath)

    # If use log scale, find max inflows and max outflows per day level
    if use_log:
        cache_filepath = cache_fpath % (ds_name, grid)
        if os.path.exists(cache_filepath):
            print 'reading %s ...' % cache_filepath
            max_inflows, max_outflows = pickle.load(open(cache_filepath, "rb"))
            print 'successfuly read cached file.'
        else:
            print 'finding max daily inflows and outflows...'
            day_max_inflows = {}
            day_max_outflows = {}
            seen = set()
            seen_add = seen.add
            all_dates = [x.split('_')[0] for x in timestamps]
            all_dates = [x for x in all_dates
                         if not (x in seen or seen_add(x))]

            for date in all_dates:
                max_inflow = 0
                max_outflow = 0
                for timegroup in timestamps_groups:
                    idx = np.where(timestamps == date + '_' + timegroup)[0][0]
                    prediction_matrix = predictions[idx]
                    if prediction_matrix[0].max() > max_inflow:
                        max_inflow = prediction_matrix[0].max()
                    if prediction_matrix[1].max() > max_outflow:
                        max_outflow = prediction_matrix[1].max()
                print 'date: ', date, 'max inflow: ', max_inflow, \
                    'max outflow: ', max_outflow
                day_max_inflows[date] = max_inflow
                day_max_outflows[date] = max_outflow
            pickle.dump((day_max_inflows, day_max_outflows),
                        open(cache_filepath, "wb"))

    for i in range(len(predictions)):
        timeslot = timestamps[i]
        output_filename = timeslot + '.png'
        print 'creating visualization for %s...' % timeslot
        date = timeslot.split('_')[0]

        # Retrieve max inflow and outflow for this day
        if use_log:
            max_inflow = day_max_inflows[date]
            max_outflow = day_max_outflows[date]

            log_max_inflow = math.log(max_inflow, log_base)
            log_max_outflow = math.log(max_outflow, log_base)

            inflow_bounds = np.logspace(0, log_max_inflow, bin_size)
            inflow_norm = mpl.colors.BoundaryNorm(inflow_bounds, cmap.N)
            outflow_bounds = np.logspace(0, log_max_outflow, bin_size)
            outflow_norm = mpl.colors.BoundaryNorm(outflow_bounds, cmap.N)

        # Load the prediction data for the specific timeslot
        prediction_matrix = predictions[i]

        # Marking invalid regions
        # (based on mask; for future implementation)
        # data[data == 0] = np.nan
        # for i in range(h):
        #     for j in range(w):
        #         inflow, outflow = data[i][j][0], data[i][j][1]
        #         if inflow == 0 and outflow == 0:
        #             data[i][j] = np.nan

        # Obtaining inflow and outflow matrix
        inflow_data = prediction_matrix[0]
        outflow_data = prediction_matrix[1]

        # Plotting inflow image
        fig = plt.figure()
        if use_log:
            inflow_img = plt.imshow(inflow_data,
                                    interpolation='nearest',
                                    cmap=cmap,
                                    origin='upper',
                                    norm=inflow_norm)
        else:
            inflow_img = plt.imshow(inflow_data,
                                    interpolation='nearest',
                                    cmap=cmap,
                                    origin='upper')
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, w, 1))
        ax.set_yticks(np.arange(0.5, h, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='grey', linestyle='-', linewidth=0.5)

        #               [left, bottom, width, height]
        ax2 = fig.add_axes([0.92, 0.23, 0.02, 0.54])
        cb = mpl.colorbar.ColorbarBase(ax2,
                                       cmap=cmap,
                                       norm=inflow_norm,
                                       boundaries=inflow_bounds)

        plt.savefig(os.path.join(inflow_output_fpath, output_filename),
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)

        # Plotting outflow image
        fig = plt.figure()
        if use_log:
            outflow_img = plt.imshow(outflow_data,
                                     interpolation='nearest',
                                     cmap=cmap,
                                     origin='upper',
                                     norm=outflow_norm)
        else:
            outflow_img = plt.imshow(outflow_data,
                                     interpolation='nearest',
                                     cmap=cmap,
                                     origin='upper')
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, w, 1))
        ax.set_yticks(np.arange(0.5, h, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='grey', linestyle='-', linewidth=0.5)

        #               [left, bottom, width, height]
        ax2 = fig.add_axes([0.92, 0.23, 0.02, 0.54])
        cb = mpl.colorbar.ColorbarBase(ax2,
                                       cmap=cmap,
                                       norm=outflow_norm,
                                       boundaries=outflow_bounds)

        plt.savefig(os.path.join(outflow_output_fpath, output_filename),
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)
