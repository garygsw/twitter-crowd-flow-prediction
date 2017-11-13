import os
import numpy as np
import cPickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

w, h = 85, 45

# Reading in data
predictions = np.load('predictions.npy')  # shape: (336, 2, 45, 85)
true_y = np.load('true_y.npy')
timestamps = np.load('timestamps.npy')

# Inverse transforming the data
mmn = pickle.load(open('preprocessing.pkl', 'rb'))
predictions = mmn.inverse_transform(predictions)
true_y = mmn.inverse_transform(true_y)


cache_fname = 'min_max.pkl'
if os.path.exists(cache_fname):
    day_max_inflows, day_max_outflows = pickle.load(open(cache_fname, "rb"))
else:
    # Creating timestamp blocks
    block_size = 6   # 6 * 5 mins = 30 minutes block
    all_timestamps = [''.join([str(hour).zfill(2), str(minute).zfill(2)])
                      for hour in range(24) for minute in range(0, 60, 5)]
    timestamps_groups = [all_timestamps[i] for i in range(0, len(all_timestamps), block_size)]

    # Finding factor for each day
    print 'initializing daily factors...'
    days = [str(i).zfill(2) for i in range(1,28)]
    day_max_inflows = {}
    day_max_outflows = {}
    for day in days:
        max_inflow = 0
        max_outflow = 0
        for timegroup in timestamps_groups:
            data = np.load('/Users/garygsw/SUTD/SMART/Dataset/grid_flow/flow201511' + day + '_' + timegroup + '.npy')
            for i in range(h):
                for j in range(w):
                    inflow, outflow = data[i][j][0], data[i][j][1]
                    if inflow > max_inflow:
                        max_inflow = inflow
                    if outflow > max_outflow:
                        max_outflow = outflow
        print 'day: ', day, 'max inflow: ', max_inflow, 'max outflow: ', max_outflow
        day_max_inflows[day] = max_inflow
        day_max_outflows[day] = max_outflow
    pickle.dump((day_max_inflows, day_max_outflows), open(cache_fname, "wb"))

cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', ['limegreen', 'yellowgreen', 'darkred'], 256)



# Generate images
for i, timestamp in enumerate(timestamps):
    timestamp = timestamps[i]
    day = timestamp[6:8]
    max_inflow = day_max_inflows[day]
    max_outflow = day_max_outflows[day]

    # Read the day's data
    inflow_true = np.copy(true_y)[i][0]
    outflow_true = np.copy(true_y)[i][1]
    inflow_pred = np.copy(predictions)[i][0]
    outflow_pred = np.copy(predictions)[i][1]

    # Finding absolute difference (ratio)
    inflow_diff = abs((inflow_pred - inflow_true)/ (inflow_true))
    outflow_diff = abs((outflow_pred - outflow_true) / (outflow_true))

    # Plotting inflow diff image
    fig = plt.figure()
    inflow_img = plt.imshow(inflow_diff,
                            interpolation='nearest',
                            cmap = cmap,
                            origin='upper',
                            norm=Normalize(vmin=0, vmax=50))
    plt.colorbar(inflow_img, fraction=0.026, pad=0.01)
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, w, 1))
    ax.set_yticks(np.arange(0.5, h, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='grey', linestyle='-', linewidth=0.5)
    plt.savefig('/Users/garygsw/SUTD/Research/Crowd Flow Prediction/error_visualizations/ratio_' + timestamp + '.png',
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)

    # Plotting inflow image
    # fig = plt.figure()
    # inflow_img = plt.imshow(inflow_pred,
    #                         interpolation='nearest',
    #                         cmap = cmap,
    #                         origin='upper',
    #                         norm=LogNorm(0.01, max_inflow))
    # plt.colorbar(inflow_img, fraction=0.026, pad=0.01)
    # ax = plt.gca()
    # ax.set_xticks(np.arange(0.5, w, 1))
    # ax.set_yticks(np.arange(0.5, h, 1))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.grid(color='grey', linestyle='-', linewidth=0.5)
    # plt.savefig('/Users/garygsw/SUTD/Research/Crowd Flow Prediction/predict_inflow_visualizations/' + timestamp + '.png',
    #             bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.close(fig)
    #
    # # Plotting outflow image
    # fig = plt.figure()
    # outflow_img = plt.imshow(outflow_pred,
    #                         interpolation='nearest',
    #                         cmap = cmap,
    #                         origin='upper',
    #                         norm=LogNorm(0.01, max_outflow))
    # plt.colorbar(outflow_img, fraction=0.026, pad=0.01)
    # ax = plt.gca()
    # ax.set_xticks(np.arange(0.5, w, 1))
    # ax.set_yticks(np.arange(0.5, h, 1))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.grid(color='grey', linestyle='-', linewidth=0.5)
    # plt.savefig('/Users/garygsw/SUTD/Research/Crowd Flow Prediction/predict_outflow_visualizations/' + timestamp + '.png',
    #             bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.close(fig)
