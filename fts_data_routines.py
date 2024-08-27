import numpy as np
import math
import matplotlib.pyplot as plt
from so3g.hk import load_range
from sotodlib.flags import get_glitch_flags


def find_time(timestamps, time):
    '''finds closest time index in array timestamps to the inputted time'''
    return (np.abs(timestamps - time)).argmin()


# useful function for analyzing timestreams
def time_zoom(aman, t_min, t_max):
    # returns indices useful for getting a time window
    time = aman.timestamps - aman.timestamps[0]
    inds = np.where((time >= t_min) & (time <= t_max))[0]
    return inds


def get_good_dets(aman, Pxx, freqs, power_threshold=1000, plot=False,
                  chop_freq=8):
    # get the 'good' dets with lots of power at 8Hz
    if (plot):
        plt.figure()

    good_dets = []
    for i in range(0, aman.dets.count):
        if aman.dets.vals[i] in aman.flags.has_cuts(['trends']):
            continue

        index_of_8hz = np.where(freqs <= chop_freq)[0][-1]
        if Pxx[i, index_of_8hz] > (Pxx[i, index_of_8hz + 10] * (
                power_threshold)):
            good_dets.append(i)
            if (plot):
                plt.loglog(freqs, Pxx[i])

    if (plot):
        plt.xlim(6, 10)
        plt.grid()
        plt.xlabel('frequency (hz)')
        plt.ylabel('P(f)')
        plt.title('power spectra of detectors with good 8hz power')
        plt.show()

    good_dets = np.array(good_dets)
    return good_dets


def get_fts_ind_ranges(fts_position_inds):
    time_interval = 200
    ind_ranges = []
    for i, inds in enumerate(fts_position_inds):
        ind_start, ind_end = inds[0], inds[-1]
        if len(inds) > 2:
            ind_start, ind_end = inds[0], inds[-2]
        # If there's only one housekeeping index (happens rarely with >1s
        # integration unless it skips a data point), get the previous one
        # which is further away and integrate in that direction.
        if ind_start == ind_end:
            next_ind = fts_position_inds[i + 1][0]
            prev_ind = fts_position_inds[i - 1][-1]
            if (next_ind - ind_start) > (ind_start - prev_ind):
                ind_end = int(ind_start + time_interval / 2)
            else:
                ind_start = int(ind_end - time_interval / 2)

        # Integrate between these
        ind_range = np.arange(ind_start, ind_end + 1)
        ind_ranges.append(ind_range)
    return ind_ranges


def get_integration_indices_optimized(fts_ind_ranges, glitch_mask):
    total_non_glitch_inds = []
    for ind_range in fts_ind_ranges:
        mask = glitch_mask[np.where((glitch_mask >= ind_range[0]) & (
            glitch_mask <= ind_range[-1]))]
        non_glitch_inds = np.setdiff1d(ind_range, mask)
        total_non_glitch_inds.append(non_glitch_inds)
    return total_non_glitch_inds


def get_integration_indices(fts_position_inds, glitch_mask):
    # So we just need to integrate between all of our times and discount the
    # glitches
    # First we need to make sure that the glitches exist
    time_interval = 200
    total_non_glitch_inds = []
    for i, inds in enumerate(fts_position_inds):
        ind_start, ind_end = inds[0], inds[-1]
        if ind_start == ind_end:
            # get the previous one which is further away and integrate in that
            # direction.
            next_ind = fts_position_inds[i + 1][0]
            prev_ind = fts_position_inds[i - 1][-1]
            if (next_ind - ind_start) > (ind_start - prev_ind):
                ind_end = int(ind_start + time_interval / 2)
            else:
                ind_start = int(ind_end - time_interval / 2)

        # Integrate between these
        ind_range = np.arange(ind_start, ind_end + 1)
        non_glitch_inds = np.setdiff1d(ind_range, glitch_mask)
        total_non_glitch_inds.append(non_glitch_inds)
    return total_non_glitch_inds


def integrate_signal(signal, total_non_glitch_inds):
    return np.array([np.mean(signal[inds]) for inds in total_non_glitch_inds])


def load_fts_range(aman, resolution=.15):
    hk_data = load_range(
        float(aman.timestamps[0]), float(aman.timestamps[-1]),
        config='/data/users/kmharrin/smurf_context/hk_config_202104.yaml')

    max_position = -1 * np.round(np.min(hk_data['fts_mirror'][1]), 2)
    expected_fts_mirror_positions = np.round(np.linspace(
        -1 * max_position, max_position,
        int(2 * max_position / resolution) + 1), 6)
    hk_mirror_positions = hk_data['fts_mirror'][1]
    # now take out the initial data chunk
    last_max_index = np.where(
        np.abs(hk_mirror_positions - (-1 * max_position)) <= .01)[0][-2]
    hk_mirror_positions = hk_mirror_positions[last_max_index:]
    hk_times = hk_data['fts_mirror'][0][last_max_index:]
    hk_mirror_slice = []
    hk_time_slice = []
    for pos in expected_fts_mirror_positions:
        hk_inds = np.where(np.abs(hk_mirror_positions - pos) <= .01)[0]
        if len(hk_inds) == 0:
            print(f"no housekeeping data at fts position {pos}. "
                  "Using data from previous position")
            hk_position = hk_mirror_slice[-1]
            hk_time = hk_time_slice[-1]
        else:
            hk_index = hk_inds[0]
            hk_position = hk_mirror_positions[hk_index]
            hk_time = hk_times[hk_index]
        hk_mirror_slice.append(hk_position)
        hk_time_slice.append(hk_time)

    #assert (np.abs(hk_mirror_slice - expected_fts_mirror_positions) <= .01).all()

    aman_fts_position_timeslice = np.array(
        [find_time(aman.timestamps, time) for time in hk_time_slice])
    return aman_fts_position_timeslice, np.array(hk_mirror_slice)


def load_fts_range_bounds(aman, resolution=.15, max_position=None):
    hk_data = load_range(
        float(aman.timestamps[0]), float(aman.timestamps[-1]),
        config='/data/users/kmharrin/smurf_context/hk_config_202104.yaml')

    if max_position is None:
        max_position = -1 * np.round(np.min(hk_data['fts_mirror'][1]), 2)
    expected_fts_mirror_positions = np.round(np.linspace(
        -1 * max_position, max_position, int(
            2 * max_position / resolution) + 1), 6)
    hk_mirror_positions = hk_data['fts_mirror'][1]
    # now take out the initial data chunk
    # start slightly after the beginning to account for any weird trends
    last_max_index = np.where(
        np.abs(hk_mirror_positions - (-1 * max_position)) <= .01)[0][2]
    # start slightly before the end similarly
    first_right_max_index = np.where(
        np.abs(hk_mirror_positions - max_position) <= .01)[0][-2]
    hk_mirror_positions = hk_mirror_positions[
        last_max_index: first_right_max_index]
    hk_times = hk_data['fts_mirror'][0][last_max_index: first_right_max_index]
    hk_mirror_slice = []
    hk_time_slice = []
    for pos in expected_fts_mirror_positions:
        hk_inds = np.where(np.abs(hk_mirror_positions - pos) <= .01)[0]
        if len(hk_inds) == 0:
            print(f"no housekeeping data at fts position {pos}. "
                  "Using data from previous position")
            hk_position = hk_mirror_slice[-1]
            hk_time = hk_time_slice[-1]
        else:
            hk_position = hk_mirror_positions[hk_inds][0]
            hk_time = hk_times[hk_inds]
        hk_mirror_slice.append(hk_position)
        hk_time_slice.append(hk_time)

    aman_fts_position_timeslice = [
        [find_time(aman.timestamps, time) for time in s] for s in (
            hk_time_slice)]
    return hk_mirror_slice, aman_fts_position_timeslice


def plot_good_interferograms(aman, good_dets, signal, fts_mirror_positions,
                             figsize=(10, 10)):
    n_bias_groups = np.max(aman.det_info.smurf.bias_group) + 1
    fig, axes = plt.subplots(math.ceil(n_bias_groups / 2), 2, figsize=figsize)
    axes = axes.ravel()

    for group in range(n_bias_groups):
        axes[group].grid(True)
        count = np.sum(aman.det_info.smurf.bias_group[good_dets] == group)
        axes[group].set_title(
            "bias group %s, number of 'good' dets = %s" % (group, count))

    print('number of interferograms in bias group -1: %s' % np.sum(
        aman.det_info.smurf.bias_group[good_dets] == -1))

    for i in range(0, aman.dets.count):
        if aman.dets.vals[i] in aman.flags.has_cuts(['trends']) or np.max(
                signal[i]) > 1:
            continue

        group = aman.det_info.smurf.bias_group[i]
        if (group != -1) and i in good_dets:
            axes[group].plot(fts_mirror_positions, signal[i])
    plt.tight_layout()
    plt.show()


def save_data(aman, n, fts_mirror_positions, signal, folder_name,
              band_channel_map, obs_id):
    fts_x, fts_y = get_fts_position(aman)

    # save the data for loading in from another notebook
    data = np.zeros((len(fts_mirror_positions), len(band_channel_map)))
    bands = np.zeros(len(band_channel_map))
    channels = np.zeros(len(band_channel_map))
    for i in range(aman.dets.count):
        band, channel = aman.det_info.smurf.band[i], aman.det_info.smurf.channel[i]
        band_channel_id = band_channel_map[(band, channel)]
        # print(band, channel)
        if aman.dets.vals[i] in aman.flags.has_cuts(['trends']):
            # just make this data a bunch of zeros
            # adjust this to actually save data-- don't trust trends lol
            data[:, band_channel_id] = signal[i]
            # data[:, band_channel_id] = np.zeros(len(fts_mirror_positions))
        else:
            data[:, band_channel_id] = signal[i]

        bands[band_channel_id] = band
        channels[band_channel_id] = channel

    filename = '%s/run_%s_interferograms.npz' % (folder_name, n)
    with open(filename, 'wb') as f:
        np.savez(f, data=data, xy_position=(fts_x, fts_y),
                 fts_mirror_positions=fts_mirror_positions,
                 bands=bands, channels=channels, obs_id=obs_id)
    print('data saved to location %s' % filename)
    return


def get_fts_position(aman):
    # get the XY stage position
    hk_data = load_range(
        float(aman.timestamps[0]), float(aman.timestamps[-1]),
        config='/data/users/kmharrin/smurf_context/hk_config_202104.yaml')
    assert np.around(np.std(hk_data['xy_stage_x'][1]), 2) == 0
    assert np.around(np.std(hk_data['xy_stage_y'][1]), 2) == 0

    fts_x, fts_y = np.around(np.mean(hk_data['xy_stage_x'][1]), 1), np.round(
        np.mean(hk_data['xy_stage_y'][1]), 1)
    return fts_x, fts_y
