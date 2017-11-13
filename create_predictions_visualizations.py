import math
import os
import colorsys
import numpy as np
import cPickle as pickle
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# for now all 500m
w, h = 87, 46  # 500m
#grid_size = '500m'
#w, h = 44, 23  # 1km
#grid_size = '1km'
use_log = True

max_flow_cache_dir = 'max_flow_cache'

cv_dirs = [
    'cv_set_1',
    #'cv_set_2',
    #'cv_set_3',
    #'cv_set_4',
]

# Color map scale
#cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', ['limegreen', 'yellowgreen', 'darkred'], 256)
cmap = plt.cm.RdYlGn
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[::-1]
#cmaplist[0] = (0, 0, 0, 0.13)   # force the first color to be white
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bin_size = 20

for cv_dir in cv_dirs:
    data_path = 'PREDICTIONS/%s/' % cv_dir

    # Reading in data
    predictions = np.load(data_path + '/stresnet_predictions.npy')  # shape: (1344, 2, 46, 87)
    #true_y = np.load(data_path + '/true_y.npy')
    timestamps = np.load(data_path + '/timestamps.npy')
    mmn = pickle.load(open('PREPROCESS/SG_Preprocess_%s' % cv_dir, 'rb'))
    predictions = mmn.inverse_transform(predictions)

    # create output directories
    visualizations_output_dir = 'prediction_visualizations/' + cv_dir
    if not os.path.exists(visualizations_output_dir):
        os.mkdir(visualizations_output_dir)
    #visualizations_output_dir = visualizations_output_dir + '/' + grid_size
    if not os.path.exists(visualizations_output_dir):
        os.mkdir(visualizations_output_dir)
    if not os.path.exists(visualizations_output_dir + '/inflow'):
        os.mkdir(visualizations_output_dir + '/inflow')
    if not os.path.exists(visualizations_output_dir + '/outflow'):
        os.mkdir(visualizations_output_dir + '/outflow')

    # If use log scale, find max inflows and max outflows per day level
    cache_fname = max_flow_cache_dir + '/max_flow_%s.pkl'
    if use_log:
        cache_path = cache_fname % cv_dir
        if os.path.exists(cache_path):
            day_max_inflows, day_max_outflows = pickle.load(open(cache_path, "rb"))
        else:
            if not os.path.exists(max_flow_cache_dir):
                os.mkdir(max_flow_cache_dir)
            # Creating timestamp blocks
            block_size = 6   # 6 * 5 mins = 30 minutes block
            all_timestamps = [''.join([str(hour).zfill(2), str(minute).zfill(2)])
                              for hour in range(24) for minute in range(0, 60, 5)]
            timestamps_groups = [all_timestamps[i] for i in range(0, len(all_timestamps), block_size)]

            # Finding factor for each day
            print 'initializing daily factors...'
            day_max_inflows = {}
            day_max_outflows = {}
            seen = set()
            seen_add = seen.add
            all_dates = [x.split('_')[0] for x in timestamps]
            all_dates = [x for x in all_dates if not (x in seen or seen_add(x))]

            for date in all_dates:
                max_inflow = 0
                max_outflow = 0
                for timegroup in timestamps_groups:
                    index = np.where(timestamps == date + '_' + timegroup)[0][0]
                    prediction_matrix = predictions[index]
                    if prediction_matrix[0].max() > max_inflow:
                        max_inflow = prediction_matrix[0].max()
                    if prediction_matrix[1].max() > max_outflow:
                        max_outflow = prediction_matrix[1].max()
                print 'date: ', date, 'max inflow: ', max_inflow, 'max outflow: ', max_outflow
                day_max_inflows[date] = max_inflow
                day_max_outflows[date] = max_outflow
            pickle.dump((day_max_inflows, day_max_outflows), open(cache_path, "wb"))

    for i in range(len(predictions)):
        timeslot = timestamps[i]
        print 'creating for %s...' % timeslot
        date = timeslot.split('_')[0]

        # Retrieve max inflow and outflow for this day
        if use_log:
            max_inflow = day_max_inflows[date]
            max_outflow = day_max_outflows[date]

            inflow_bounds = np.logspace(0, math.log(max_inflow, 10), bin_size)
            inflow_norm = mpl.colors.BoundaryNorm(inflow_bounds, cmap.N)
            outflow_bounds = np.logspace(0, math.log(max_outflow, 10), bin_size)
            outflow_norm = mpl.colors.BoundaryNorm(outflow_bounds, cmap.N)

        # Load the data
        prediction_matrix = predictions[i]

        # Marking invalid regions
        #data[data == 0] = np.nan
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
                                    norm=inflow_norm) # LogNorm(0.01, max_inflow))
        else:
            inflow_img = plt.imshow(inflow_data,
                                    interpolation='nearest',
                                    cmap=cmap,
                                    origin='upper')
        #plt.colorbar(inflow_img, fraction=0.026, pad=0.01, spacing='proportional', norm=inflow_norm)
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, w, 1))
        ax.set_yticks(np.arange(0.5, h, 1))
        ax.set_xticklabels([]);
        ax.set_yticklabels([]);
        ax.grid(color='grey', linestyle='-', linewidth=0.5)

        ax2 = fig.add_axes([0.92, 0.23, 0.02, 0.54])   # [left, bottom, width, height]
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=inflow_norm, boundaries=inflow_bounds)

        plt.savefig(visualizations_output_dir + '/inflow/' + timeslot + '.png',
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)

        # Plotting outflow image
        fig = plt.figure()
        if use_log:
            outflow_img = plt.imshow(outflow_data,
                                     interpolation='nearest',
                                     cmap=cmap,
                                     origin='upper',
                                     norm=outflow_norm)  #LogNorm(0.01, max_outflow))
        else:
            outflow_img = plt.imshow(outflow_data,
                                     interpolation='nearest',
                                     cmap=cmap,
                                     origin='upper')
        #plt.colorbar(outflow_img, fraction=0.026, pad=0.01)
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, w, 1))
        ax.set_yticks(np.arange(0.5, h, 1))
        ax.set_xticklabels([]);
        ax.set_yticklabels([]);
        ax.grid(color='grey', linestyle='-', linewidth=0.5)

        ax2 = fig.add_axes([0.92, 0.23, 0.02, 0.54])   # [left, bottom, width, height]
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=outflow_norm, boundaries=outflow_bounds)

        plt.savefig(visualizations_output_dir + '/outflow/' + timeslot + '.png',
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)
