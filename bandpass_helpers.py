import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize, signal, interpolate
from sklearn.linear_model import LinearRegression
from IPython.display import HTML, display

h = 6.626e-34  # planck's constant
k = 1.38e-23   # boltzman constant
c = 3e8     # speed of light

# a function that calcualte the RMS after eliminating outliers
# timeseries: timestream


def smart_rms(timeseries, n_iters, threshold, return_data=False):
    ok = np.where(timeseries == timeseries)  # CAN PROBABLY TAKE THIS OUT
    timeseries = timeseries[ok]
    for _ in range(n_iters):
        rms_tmp = np.std(timeseries)
        mean_tmp = np.mean(timeseries)
        if rms_tmp == 0:  # don't divide or else we'll all be nans...
            return (mean_tmp, rms_tmp)
        good = np.where(np.abs(timeseries - mean_tmp) / rms_tmp < threshold)
        timeseries = timeseries[good]

    if return_data:
        return mean_tmp, rms_tmp, timeseries
    return(mean_tmp, rms_tmp)


def filter_frequency_mask(n_samples, f_ranges, c,
                          fts_stage_step_size, fts_frequency_cal):
    # f_low and f_high assumed to be Ghz!!!!
    # This is half of the Fourier Space
    frequencies = frequency(
        np.ones(n_samples), c, fts_stage_step_size, fts_frequency_cal) / 1e9
    mask = ((frequencies > f_ranges[0][0]) & (frequencies < f_ranges[0][1]))
    # now keep adding more masks if needed
    for (f_low, f_high) in f_ranges[1:]:
        added_mask = ((frequencies > f_low) & (frequencies < f_high))
        mask = (mask | added_mask)
    # we need to add the second half reflected about
    # only reflect after the zero!
    return mask

# eliminates huge spikes in data (assumed not "real")
# uses smart RMS to find outliers, replace them with the average of neighbors


def despike_timeseries(timestream, threshold):
    avg, rms = smart_rms(timestream, 7, threshold)
    timestream = timestream - avg  # remove the mean
    # should we subtract off the mean??
    bads = np.where(np.abs(timestream) > threshold * rms)
    timestream[bads] = 0.5 * (np.roll(timestream, 1)
                              [bads] + np.roll(timestream, -1)[bads])
    return(timestream)


# maybe try adding a stat that also tries other parts of the bands...
# not really sure about this though..
def get_cut_stat_new(data, fourier_band_filter, out_of_band_filter):
    if np.std(data) > 1e-4:
        interferogram = get_cleaned_interferogram(data, 5, 10, 7)

        # Just take a conv between the filter of ones and our data I guess...
        signal_stat = np.mean(interferogram[fourier_band_filter]).real
        noise_stat = np.std(interferogram[out_of_band_filter]).real
        return (signal_stat / noise_stat)

    return 0.


def correct_interferogram(data, n_rms_iters, spike_threshold, poly_order,
                          polyfit=True):
    average, rms = smart_rms(data, n_rms_iters, spike_threshold)
    rms_interferogram = data - average
    # eliminate spikes in the data
    despiked_interferogram = despike_timeseries(
        rms_interferogram, spike_threshold)
    # remove a polynomial baseline
    if (polyfit):
        poly_fitted_interferogram, poly = remove_poly(
            despiked_interferogram, poly_order)
        corrected_interferogram = poly_fitted_interferogram
    else:
        corrected_interferogram = despiked_interferogram
    return corrected_interferogram


def get_cleaned_interferogram(data, n_rms_iters, spike_threshold, poly_order,
                              plot=False, polyfit=False, take_sqrt=True):
    if (take_sqrt):
        data = np.sqrt(data)
    corrected_interferogram = correct_interferogram(
        data, n_rms_iters, spike_threshold, poly_order, polyfit=polyfit)

    # centered_interferogram = center_interferogram(corrected_interferogram,
    #                                               max_ind=None)

    centered_interferogram = corrected_interferogram

    # apply the phase spectrum correction
    window = make_triangle_window(centered_interferogram)
    phase_corrected_passband = np.real(invert_interferogram(
        centered_interferogram, window))

    return phase_corrected_passband[:int(np.ceil(len(
        phase_corrected_passband) / 2))]


def band_location_cuts(total_cut_stats, total_good_channels, array_data,
                       x_locs, y_locs, band_nums, distance_threshold=4):
    # see if the correct channel was 'on'
    print('first cut: channel not "on" in location file: \n ')
    for i, good_channels in enumerate(total_good_channels):
        for band_num in band_nums:
            indices_to_remove = []
            for j, channel in enumerate(good_channels[band_num]):
                if array_data[band_num][channel] == 0:
                    print('removing from good bands: run # %s, channel %s,'
                          ' band %s' % (i, channel, band_num))
                    indices_to_remove.append(j)

            total_good_channels[i][band_num] = np.delete(
                good_channels[band_num], indices_to_remove)

    # check the centroid of our data and see if it matches
    print('\nsecond cut: channel too far from centroid: \n')
    for i, good_channels in enumerate(total_good_channels):
        for band_num in band_nums:
            good_x = x_locs[good_channels[band_num]]
            good_y = y_locs[good_channels[band_num]]
            good_points = np.array([good_x, good_y]).T

            removed_points = []
            while len(good_points) > 2:
                centroid = np.sum(good_points, axis=0) / len(good_points)
                distances = np.sqrt(np.sum(np.square(good_points - centroid),
                                           axis=1))
                if np.max(distances) > distance_threshold:
                    index = np.argmax(distances)
                    print('removing from good bands: run # %s, channel %s,'
                          ' band %s' % (i, good_channels[band_num][index],
                                        band_num))
                    removed_points.append(good_points[index])
                    good_points = np.delete(good_points, index, axis=0)
                    total_good_channels[i][band_num] = np.delete(
                        total_good_channels[i][band_num], index)
                else:
                    break

            removed_points = np.array(removed_points)

            if len(removed_points) == 0:
                continue

            goods = good_channels[band_num]
            s_t_n_resized = np.ones(np.size(x_locs))
            s_t_n_resized[goods] = total_cut_stats[i][band_num][
                good_channels[band_num]] * 10
            plt.scatter(x_locs, y_locs, s=s_t_n_resized,
                        color=['r', 'g'][band_num])
            plt.scatter(centroid[0], centroid[1], color='b', s=30)
            x_removed, y_removed = np.array(removed_points).T
            plt.scatter(x_removed, y_removed, color='black', s=30)
            plt.title('run %s' % i)
            plt.show()


def plot_band_snr_values(band_attrs, band):
    snr_values = band_attrs[:, 3]
    band_centers = band_attrs[:, 1]
    bandwidths = band_attrs[:, 2]
    plt.scatter(band_centers, bandwidths, s=3 * (snr_values),
                c=np.sqrt(snr_values))
    plt.xlabel('band center (Ghz)')
    plt.ylabel('band width (Ghz)')
    plt.grid()
    plt.title('Bandwidths and centers vs SNR for %s Ghz band' % band)
    plt.show()


def plot_colored_hist(data, bins, snr_values, n_rms_iters=7, rms_threshold=5,
                      average=None, devs=None, label=None):

    bin_locs = np.linspace(data.min(), data.max(), bins)
    bin_vals = np.digitize(data, bin_locs)
    width = bin_locs[1] - bin_locs[0]
    # Get color data values so we can normalize
    all_color_values = []
    for bin_ind in range(1, bins + 1):
        bin_positions = np.where(bin_vals == bin_ind)
        if len(bin_positions[0]) > 0:
            color_data_value = np.mean(
                snr_values[bin_positions])  # / np.max(snr_values))
        else:
            color_data_value = np.nan
        all_color_values.append(color_data_value)

    max_color_value = np.nanmax(all_color_values)
    min_color_value = np.nanmin(all_color_values)

    # plot the bars individually now.
    cmap = plt.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(vmin=min_color_value,
                                       vmax=max_color_value)
    for i, bin_ind in enumerate(range(1, bins + 1)):
        bin_positions = np.where(bin_vals == bin_ind)
        color_value = cmap(norm(all_color_values[i]))
        plt.bar(bin_locs[bin_ind - 1], len(bin_positions[0]),
                color=color_value, width=width)

    plt.colorbar(matplotlib.cm.ScalarMappable(norm, cmap=cmap),
                 label='mean SNR per bin')
    plt.grid()

    # s_avg, s_rms = smart_rms(data, n_rms_iters, rms_threshold)
    s_avg, s_rms = weighted_rms(data, snr_values ** 2)

    plt.axvline(s_avg, ls='--', color='black',
                label=r'weighted mean of all measurements: %.1f' % s_avg)
    if (average):
        plt.axvline(average, label='%s of total averaged band: %.1f' % (
            label, average), ls='-', color='black')
    if (devs):
        plt.axvline(s_avg + devs * s_rms, ls='-',
                    label='deviation theshold', color='gray')
        plt.axvline(s_avg - devs * s_rms, ls='-', color='gray')

    plt.legend(loc='upper left', bbox_to_anchor=(0, -.1))
    print("weighted mean: %.1f, weighted rms: %.1f" % (s_avg, s_rms))
    return


def weighted_rms(data, weights):
    average = np.average(data, weights=weights)
    # Fast and numerically precise:
    variance = np.average((data - average)**2, weights=weights)
    return (average, np.sqrt(variance))


def band_hist_cuts(passbands, band_attrs, attr_index, attr_label, band_label,
                   n_rms_iters=7, rms_threshold=5, dev_threshold=4):
    indiv_attrs = band_attrs[:, attr_index]
    # s_avg, s_rms = smart_rms(indiv_attrs, n_rms_iters, rms_threshold)
    s_avg, s_rms = weighted_rms(indiv_attrs, band_attrs[:, 3] ** 2)
    # plt.hist(indiv_attrs, bins=50)
    # plt.grid()
    # plt.title('%s %s' % (attr_label, band_label))
    # plt.show()

    plot_colored_hist(indiv_attrs, 21, band_attrs[:, 3], devs=dev_threshold)
    plt.title('%s %s Ghz' % (attr_label, band_label))

    # print("smart mean: %.3g, smart_rms: %.3g" % (s_avg, s_rms))
    num_deviations = np.abs(indiv_attrs - s_avg) / s_rms
    large_devs = np.where(num_deviations > dev_threshold)
    small_devs = np.where(num_deviations <= dev_threshold)
    if np.size(large_devs[0]) > 0:
        print('removing channels %s with %s values of %s from passbands: # '
              'weighted deviations from weighted mean = %s' % (
                  band_attrs[:, 0][large_devs], attr_label,
                  np.round(indiv_attrs[large_devs], 3),
                  np.round(num_deviations[large_devs], 2)))
    else:
        print('no bands removed.\n')
    plt.show()
    print(70 * '-')
    return passbands[small_devs], band_attrs[small_devs]


def divide_by_nonmax_mean(arr):
    max_ind = np.argmax(arr)
    nonmax_arr = np.delete(arr, max_ind)
    if nonmax_arr.any():
        # if the mean is less than one, don't divide- don't elevant low S/N
        # if the others are low S/N as well
        return arr / np.max([1, np.mean(np.delete(arr, max_ind))])


def find_interferograms_clean(data, fourier_band_filters, fourier_noise_filter,
                              spike_threshold):
    n_chans = np.shape(data)[1]
    n_bands = len(fourier_band_filters)
    total_cut_stats = np.zeros((n_chans, n_bands))

    for i in range(n_chans):
        if (np.std(data[:, i]) > 1e-4):
            # despike the channel under consideration
            # tmp = despike_timeseries(np.sqrt(data[:, i]), spike_threshold)
            # tmp = despike_timeseries(tmp, spike_threshold)
            # filter in fourier space
            cut_stats = [get_cut_stat_new(data[:, i], f, fourier_noise_filter)
                         for f in fourier_band_filters]
            # divide by the mean of the cut stats that didn't make it
            cut_stats = list(divide_by_nonmax_mean(np.array(cut_stats)))
            total_cut_stats[i] = cut_stats
            # print(cut_stats)
            # total_cut_stats.append(cut_stats)

            # cut_stat_combined = np.sqrt(np.sum(cut_stats ** 2))
            # cut_stats_combined.append(cut_stat_combined)

    # normalize each cut statistic relative to itimeseries spike rejected rms
    total_cut_stats = np.array(total_cut_stats)
    # combine all of the cut stats intno one group as well
    cut_stats_combined = np.sqrt(np.sum(total_cut_stats ** 2, axis=1))
    return (total_cut_stats.T, cut_stats_combined)


def band_classifier_new(cut_stats):
    return np.max(cut_stats, axis=0), np.argmax(cut_stats, axis=0)


def power_law(x, k, C0, alpha):
    return k * (x ** (-1 * alpha)) + C0


def poisson(x, k, A, C):
    return A * ((x) ** k) * np.exp(-1 * (x)) + C


def tanh(x, h, A):
    return A * (1 - np.tanh(x - h))


def remove_powerlaw_noise(frequencies, spectrum, end_fit_freq,
                          noise_bounds, plots=False):
    end_fit_index = find_freq(frequencies, end_fit_freq)
    noise_start_index = find_freq(frequencies, noise_bounds[0])
    noise_end_index = find_freq(frequencies, noise_bounds[1])
    frequencies = frequencies / 1e9
    # find maximum index
    max_index = np.argmax(spectrum[:end_fit_index])
    # print(frequencies[max_index])
    y = spectrum[max_index:end_fit_index]
    x = frequencies[max_index:end_fit_index]

    # try adding some initial guesses here maybe..
    popt, pcov = optimize.curve_fit(power_law, x, y, maxfev=10000, bounds=(
        (1e6, -1e10, 1), (1e11, 1e10, 5)))

    # popt, pcov = optimize.curve_fit(power_law, x, y, maxfev=10000, bounds=(
    #     (-np.inf, -np.inf, 1), (np.inf, np.inf, 5)))

    if plots:
        plt.plot(x, y)
        plt.plot(x, power_law(x, *popt))
        print(popt)
        plt.show()

    spectrum_fit = np.zeros(np.size(spectrum))
    # x_whole = frequencies[max_index:]
    spectrum_fit[max_index:end_fit_index] = power_law(x, *popt)

    spectrum_fitted = spectrum - spectrum_fit

    # make sure otherwise that we're centered at zero
    current_mean = np.mean(spectrum_fitted[end_fit_index - 10: end_fit_index])
    old_mean = np.mean(spectrum_fitted[noise_start_index:noise_end_index])

    # Now add the mean back so that we don't suddenly hit go to a zero mean!
    spectrum_fitted[max_index:end_fit_index] += (old_mean - current_mean)

    if plots:
        plt.plot(frequencies[max_index:], spectrum_fit[max_index:])
        plt.plot(frequencies[max_index:], spectrum[max_index:])
        plt.show()
        plt.plot(frequencies[max_index:], spectrum_fitted[max_index:])
        plt.title('spectrum - power law fit')
        plt.xlabel('frequency (Ghz)')
        plt.show()
        plt.plot(frequencies, spectrum_fitted)
        plt.show()

    return spectrum_fitted


# a polynomial filter for the interferograms
def remove_poly(timeseries, order=5):
    x = np.arange(np.size(timeseries))
    poly_params = np.polyfit(x, timeseries, order)
    poly_template = np.polyval(poly_params, x)
    return (timeseries - poly_template, poly_template)


def center_interferogram(interferogram, max_ind=None):
    '''center interferogram about the maximum'''

    # find the maximum point
    if (max_ind is None):
        max_ind = np.argmax(interferogram)

    # now make this interferogram symmetric about the max
    interferogram_size = np.min([len(interferogram) - 1 - max_ind, max_ind])
    centered_interferogram = interferogram[max_ind - interferogram_size: (
        max_ind + interferogram_size + 1)]
    return centered_interferogram
    # return interferogram[max_ind - interferogram_size: (
    #     max_ind + interferogram_size + 1)]

    return interferogram


def fit_parabola(p1, p2, p3):
    x1, y1, x2, y2, x3, y3 = p1 + p2 + p3
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3 ** 2 * (y1 - y2) + x2 ** 2 * (y3 - y1) + x1 ** 2 * (
        y2 - y3)) / denom
    # c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (
    #     x1 - x2) * y3) / denom
    vertex_x = (-b / (2 * a))
    # vertex_y = (c - (b ** 2) / (4 * a))
    # we just want the x-vertex!
    return vertex_x


def triangle_apodization_window(interferogram, l1, l2, beta):
    '''creates a triangular apodization window centered around center_pont for
    going to zero at the closest edge of the interferogram. This goes from -l_1
    to +l_2.
    '''
    # see page 286 of porter and tanner
    # make it so that this triangle goes to zero at the edge
    # l2 = len(interferogram) // 2
    indices = np.arange(-1 * l1, l2 + 1)
    triangle_vals = 1 - (np.abs(indices - beta)) / (l2 - beta)
    # make any that are less than (-l2 + 2 beta) zero
    zero_indices = np.where(indices < (-1 * l2 + 2 * beta))
    triangle_vals[zero_indices] = np.zeros(len(zero_indices[0]))
    return triangle_vals
    # return np.ones(len(interferogram))


def zero_pad_data(interferogram, final_length, max_ind):
    # power_of_2 = int(2 ** np.ceil(np.log2(len(interferogram))))

    # set up array with the first point for which x >= beta as the first array
    # element followed by the other points with x > beta. The remaining data
    # points are put at the top of the array, with the last element being the
    # last point for which x < beta. the center contains many zeros
    # we don't zero pad since modern FFT algorithms are already fast enough!
    begin_data = interferogram[max_ind:]
    end_data = interferogram[:max_ind]
    num_zeros = final_length - len(interferogram)
    zero_padded_data = np.concatenate(
        [begin_data, np.zeros(num_zeros), end_data])
    return zero_padded_data


def linear_phase_fit(freqs, phase, amp, bin_min_freq):

    bin_min = find_freq(freqs, bin_min_freq)

    x_points = freqs[:len(phase) // 2] / np.max(freqs) * np.pi
    y_points = phase[:len(phase) // 2]

    # give the in-band weights massive values
    weights = amp[:len(phase) // 2] ** 5
    weights[:bin_min] = 0

    # Aw = A * np.sqrt(W)[:, np.newaxis]
    # Yw = y_points * np.sqrt(W)

    # fit WLS using sample_weights
    WLS = LinearRegression(fit_intercept=True)
    WLS.fit(x_points[:, np.newaxis], y_points, sample_weight=weights)

    # beta = -1 means that we need to shift the apodization window by 1 sample
    # to the left
    beta = -1 * WLS.coef_[0]  # in units of samples to shift by!

    # score = WLS.score(x_points[:, np.newaxis], y_points,
    #                   sample_weight=weights)
    return beta


def get_ramp(interferogram, l1, l2, beta):
    # this should have the center right at around zero
    # assume the interferogram goes from -L_1 to +L_2
    indices = np.arange(-1 * l1, l2 + 1)
    ramp_vals = (indices + l1) / (2 * (beta + l1))
    # make any that are less than (-l2 + 2 beta) zero
    one_indices = np.where(indices > (l1 + 2 * beta))
    ramp_vals[one_indices] = np.ones(len(one_indices[0]))
    return ramp_vals


def phase_correct_interferogram(
        interferogram, max_ind, bin_min, fts_step_size, n_rms_iters,
        spike_threshold, poly_order, polyfit=True):
    # If we're not on the short side, reverse the data.
    if max_ind > len(interferogram) // 2:
        interferogram = interferogram[::-1]
        max_ind = len(interferogram) - 1 - max_ind
        print('reversing interferogram... %s, %s' % (
            max_ind, np.argmax(interferogram)))
        assert max_ind == np.argmax(interferogram)

    interferogram = correct_interferogram(
        interferogram, n_rms_iters, spike_threshold, poly_order,
        polyfit=polyfit)

    # first fit a parabola to the three points
    # p1, p2, p3 = [[max_ind - 1, interferogram[max_ind - 1]],
    #               [max_ind, interferogram[max_ind]],
    #               [max_ind + 1, interferogram[max_ind + 1]]]
    # vertex_x = fit_parabola(p1, p2, p3)
    # beta = vertex_x - max_ind
    # print('beta : %s' % beta)
    beta = 0
    # assume that beta the max ind is the closest to beta here.
    centered_interferogram = center_interferogram(interferogram, max_ind)
    # new max should just be the length / 2
    center_ind = len(centered_interferogram) // 2

    # makes a triangular apodization window
    l1 = len(centered_interferogram) // 2
    apodization_window = triangle_apodization_window(centered_interferogram,
                                                     l1, l1, beta)
    # apodize the interferogram.
    apodized_interferogram = centered_interferogram * apodization_window
    # zero pad interferogram so that the centered length is the same as the
    # original
    zero_padded_data = zero_pad_data(apodized_interferogram,
                                     len(interferogram), center_ind)

    # set up the M-point array here
    fft_interferogram = np.fft.fft(zero_padded_data)

    c = 3e8
    fft_freqs = c * np.fft.fftfreq(len(zero_padded_data), d=(
        4 * fts_step_size))

    # compute the amplitude spectrum.
    amplitude_spectrum = np.sqrt(np.imag(fft_interferogram) ** 2 + np.real(
        fft_interferogram) ** 2)

    # compute the phase spectrum
    phase_spectrum = np.arctan2(
        np.imag(fft_interferogram), np.real(fft_interferogram))

    phase_spectrum = np.unwrap(phase_spectrum)  # do we want to unwrap?

    beta = linear_phase_fit(fft_freqs, phase_spectrum, amplitude_spectrum,
                            bin_min)
    # return beta, fft_freqs, phase_spectrum, amplitude_spectrum
    if (np.abs(beta) > 1):
        print('error: beta > 1 at %s. Using zero.' % beta)
        beta = 0

    # if the left side is the long side, flip and then flip back after
    # multiplying.
    # if max_ind > len(interferogram) // 2:
    #     interferogram = interferogram[::-1]
    #     beta = -1 * beta
    #     max_ind = len(interferogram) - 1 - max_ind
    #     print(max_ind, np.argmax(interferogram))
    #     assert max_ind == np.argmax(interferogram)

    # get length L_1 and L_2 of this intergerogram
    l1 = max_ind
    l2 = len(interferogram) - max_ind - 1

    # multiply by ramp
    ramp_func = get_ramp(interferogram, l1, l2, beta)
    # apodize
    apodization_window = triangle_apodization_window(
        interferogram, l1, l2, beta)
    # apodize the interferogram.
    apodized_interferogram = interferogram * ramp_func * apodization_window
    zero_padded_data = zero_pad_data(
        apodized_interferogram, len(interferogram), max_ind)

    fft_interferogram = np.fft.fft(zero_padded_data)
    # apply the phase correction with our phase from before
    phase_corrected = fft_interferogram * np.exp(-1j * phase_spectrum)
    # return fft_freqs, phase_corrected, beta
    return phase_corrected


def make_triangle_window(timeseries):
    return signal.triang(np.size(timeseries))


def invert_interferogram(interferogram, window):
    fft_interferogram = np.fft.fft(interferogram * window)
    # calculate the phase spectrum
    phase_spectrum = np.arctan2(
        np.imag(fft_interferogram), np.real(fft_interferogram))
    phase_spectrum = np.unwrap(phase_spectrum)
    # apply the phase correction
    phase_corrected = fft_interferogram * np.exp(-1j * phase_spectrum)
    return phase_corrected


def frequency(raw_passband, c, fts_stage_step_size, fts_frequency_cal):
    # generates frequency range for the passband data
    N_samples_kept = np.size(raw_passband)
    Resolution = c / (fts_stage_step_size * N_samples_kept) / \
        4. * fts_frequency_cal
    frequency = np.arange(N_samples_kept / 2) * Resolution
    return(frequency)


def find_peak(passband, peak_index):
    prev_peak_values = np.diff(passband[:peak_index])
    next_peak_values = np.diff(passband[peak_index:])
    try:
        peak_start = next(filter(lambda i: prev_peak_values[i] < 1e-2 and (
            passband[i] < .2 * passband[peak_index]),
            np.arange(len(prev_peak_values))[::-1]))
        peak_end = next(filter(lambda i: next_peak_values[i] > -1e-2 and (
            passband[i + peak_index] < .2 * passband[peak_index]),
            range(len(next_peak_values))))
    except StopIteration:
        print('no peak found for band...')
        return None, None
    # Returning start + 2 works better to fit in the noise RJ correction..
    # I believe this is correct to do here but it's debatable.
    return (peak_start + 2, peak_index + peak_end)


def slope_function(x, y, start, stop, exp_slope=1, side='right'):
    # make the minimum value a 1
    assert len(x) == len(y)
    good_x = np.where((x >= start) & (x <= stop))
    y = y / np.min(y[good_x])
    # make the normed value a 1
    y_sloped = np.ones(len(y))
    y_sloped[good_x] = y[good_x]
    # add exponential decay after the right edge now
    right_edge = good_x[0][-1]
    y_sloped[(right_edge):] = (np.max(y_sloped) - 1) * np.exp(
        -1 * exp_slope * (x[(right_edge):] - x[right_edge])) + 1

    return y_sloped


def get_passband(interferogram, fts_stage_step_size, fts_frequency_cal,
                 n_rms_iters=5, spike_threshold=10, bin_min_freq=15,
                 lower_bound=22., upper_bound=35., slope_cut=1e-3,
                 poly_order=5, end_fit_freq=23, noise_bounds=(100, None),
                 f_ignore_around_peak=3, fit_noise=True,
                 correction_func=None, correction_params=[],
                 max_ind=None, subtract_mean=True, edge_power_limit=.05,
                 normalize=True, amplitude_transfer_func=None,
                 interp_freqs=None, transfer_func_edges=(None, None)):
    if (max_ind is None):
        max_ind = np.argmax(interferogram)  # results may vary...
    # phase_corrected_passband = phase_correct_interferogram(
    #     interferogram, max_ind, bin_min_freq, fts_stage_step_size, n_rms_iters,
    #     spike_threshold, poly_order, polyfit=True)
    corrected_interferogram = correct_interferogram(
        interferogram, n_rms_iters, spike_threshold, poly_order, polyfit=True)

    window = make_triangle_window(corrected_interferogram)
    phase_corrected_passband = invert_interferogram(corrected_interferogram,
                                                    window)

    # make a frequency axis and apply corrections
    frequency_hz = frequency(phase_corrected_passband,
                             c, fts_stage_step_size, fts_frequency_cal)
    bin_min = find_freq(frequency_hz, bin_min_freq)
    passband = phase_corrected_passband

    passband = phase_corrected_passband[
        0: int(np.ceil(np.size(phase_corrected_passband) / 2))].real

    if noise_bounds[1] is None:
        noise_bounds = (noise_bounds[0], frequency_hz[-1] / 1e9)

    if fit_noise:
        # fit a power law to this to reduce noise at low freqs
        passband = remove_powerlaw_noise(
            frequency_hz, passband, end_fit_freq,
            noise_bounds)

    if (correction_func is not None):  # a multiplicative correction
        correction = correction_func(frequency_hz, *correction_params)
        passband = passband * correction

    if (amplitude_transfer_func is not None):  # a multiplicative correction
        assert transfer_func_edges != [None, None]
        start, stop = transfer_func_edges
        correction = slope_function(
            frequency_hz / 1e9, 1 / amplitude_transfer_func(
                frequency_hz / 1e9), start, stop, exp_slope=1)

        # plt.plot(frequency_hz / 1e9, amplitude_transfer_func(
        #     frequency_hz / 1e9))
        # plt.show()

        # plt.plot(frequency_hz / 1e9, 1 / amplitude_transfer_func(
        #     frequency_hz / 1e9))
        # plt.show()

        # plt.plot(frequency_hz / 1e9, correction)
        # plt.show()

        passband = passband * correction

    if (passband is None):
        print('no passband found...')
        return None, None, None, None, None, None, frequency_hz
    # passband = phase_corrected_passband.real

    # rj_correction = RJ_correction(frequency_hz, h, k, t_lamp, t_ambient)
    # passband = RJ_corrected_band(passband, rj_correction,
    #                              bin_min=bin_min)

    # fit out a noise bias term to this passband
    noise_start_index = find_freq(frequency_hz, noise_bounds[0])
    noise_end_index = find_freq(frequency_hz, noise_bounds[1])
    # subtract a mean baseline from the band
    if (subtract_mean):
        passband = passband - np.mean(passband[
            noise_start_index:noise_end_index])
    if (normalize):
        passband = normalize_passband(passband, lower_index=bin_min)

    # calculate the band center and bandwidth and edges
    cf, bw, lower_edge, upper_edge = get_band_attrs(
        passband, frequency_hz, lower_bound, upper_bound, slope_cut,
        bin_min=bin_min)
    cf = np.floor(cf*100)/100.
    bw = np.floor(bw*100)/100.

    if (lower_edge is None):
        print('lower edge not fitted...')
        return None, None, None, None, None, None, frequency_hz

    # should interpolate after we get the attrs
    if (interp_freqs is not None):
        interpolation_func = interpolate.interp1d(
            frequency_hz, passband, fill_value='extrapolate')

        passband = interpolation_func(interp_freqs)
        frequency_hz = interp_freqs

    # calculate the SNR
    snr = np.around(1 / np.std(passband[noise_start_index:]), decimals=1)
    return passband, cf, bw, snr, lower_edge, upper_edge, frequency_hz


# Sum over a weighted avereage to create the average band
def obtain_average_band(passbands, weights):
    # don't do anything if we don't have any data!
    if passbands.size == 0:
        return None

    # noise_start_index = find_freq(frequencies, noise_start_freq)
    # weights = (1 / np.std(passbands[:, noise_start_index:], axis=1)) ** 2
    average_band = np.sum(weights * passbands.T, axis=1)
    # if (return_weights):
    #    return average_band, weights
    return average_band


def plot_indiv_bands(ax, passbands, band_attrs, frequencies, bin_min_freq,
                     plot_freq_range, desc):

    ax.set_xlabel("frequency [GHz]")
    ax.set_title("S/N FTS channels %s" % desc)
    ax.set_ylim(-.2, 1.2)
    if len(passbands) == 0:
        return
    for attrs, passband in zip(band_attrs, passbands):
        ax.plot(frequencies / 1e9, passband, linewidth=0.5,
                label='%d, %.3g, %.3g, %.3g, %.1f, %.1f' % tuple(attrs))
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.1))
    ax.set_xlim(plot_freq_range[0], plot_freq_range[1])


def plot_band_data(run_num, passbands, band_attrs, average_band, frequencies,
                   weights, bin_min_freq=15, plot_freq_range=None):
    bin_min = find_freq(frequencies, bin_min_freq)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for attrs, passband in zip(band_attrs, passbands):
        axes[0].plot(frequencies / 1e9, passband, linewidth=0.5,
                     label='%d, %.3g, %.3g, %.3g, %.1f, %.1f' % tuple(attrs))

    average_passband = average_band / np.max(average_band[bin_min:])
    # plt.plot(frequency_Hz[bin_min:]/1e9, RJ_corr[bin_min:])
    # Plot all the bands + average
    axes[0].plot(frequencies / 1e9, average_passband, linewidth=1, color="k",
                 label='weighted average')
    axes[0].set_xlabel("frequency [GHz]")
    axes[0].set_ylabel("FTS response, peak normalized")
    axes[0].set_title("All FTS channels for run %s \n key: channel,"
                      " center, width, SNR, low/high edges" % str(run_num))
    # handles, labels = axes[0].get_legend_handles_labels()
    # line = matplotlib.lines.Line2D(
    #     [0], [0], color='black', lw=0,
    #     label='channel, band_center, bandwidth, SNR')
    # handles.insert(0, line)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -.1))
    axes[0].set_ylim(-.2, 1.2)

    if plot_freq_range is None:
        plot_freq_range = (bin_min_freq, np.max(frequencies) / 1e9)
    axes[0].set_xlim(plot_freq_range[0], plot_freq_range[1])
    low_SN = np.where(np.sqrt(weights) < 50)
    high_SN = np.where(np.sqrt(weights) > 50)

    plot_indiv_bands(axes[1], passbands[low_SN], band_attrs[low_SN],
                     frequencies, bin_min_freq, plot_freq_range, '< 50')
    plot_indiv_bands(axes[2], passbands[high_SN], band_attrs[high_SN],
                     frequencies, bin_min_freq, plot_freq_range, '> 50')
    # plt.tight_layout()
    plt.show()

    # Plot the high S/N bands
    return


def obtain_passbands(
        band_num, data_sets, total_good_band_channels, fts_step_size,
        fts_frequency_cal, output_vals=False, bin_min_freq=15,
        noise_bounds=(60, None), plot_freq_range=None, n_rms_iters=5,
        spike_threshold=10, lower_bound=22, upper_bound=35, slope_cut=1e-3,
        poly_order=7, end_fit_freq=23, f_ignore_around_peak=.3, fit_noise=True,
        low_snr_cutoff=15, high_snr_change=80, high_snr_cutoff=1000,
        plots=True, centroid=None, correction_func=None,
        correction_params=[], max_ind=None, subtract_mean=True,
        edge_power_limit=.05, normalize=True, x_positions=None,
        y_positions=None, apply_freq_correction=False,
        apply_amplitude_correction=False, transfer_func_edges=[None, None]):
    average_bands = []
    total_passbands = []
    total_band_attrs = []
    frequencies = None
    for i, data_set in enumerate(data_sets):
        passbands = []
        band_attrs = []

        good_run_channels = total_good_band_channels[i][band_num]
        for channel in good_run_channels:
            interferogram = np.sqrt(data_set[:, channel])

            frequency_calibration_factor = fts_frequency_cal  # OLD FACTOR
            amplitude_transfer_func = None
            common_frequencies = None

            if (apply_freq_correction):
                # only use the centroid calibration if we have enough channels
                # to begin with
                if centroid is None:
                    centroid_to_use = get_centroid(
                        good_run_channels, x_positions, y_positions)
                else:
                    centroid_to_use = centroid

                pixel_position = get_pixel_position(channel, x_positions,
                                                    y_positions)
                frequency_calibration_factor = get_frequency_calibration_factor(
                    centroid_to_use, pixel_position)

                # set a common set of frequencies for this
                common_frequencies = frequency(
                    interferogram, c, fts_step_size, fts_frequency_cal)

                if (apply_amplitude_correction):
                    # ymax = (fts_step_size * len(interferogram)) * 1000
                    # print(ymax)
                    # print('applying amplitude correction:')
                    amplitude_transfer_func = get_amplitude_transfer_func(
                        centroid_to_use, pixel_position)

            # print(frequency_calibration_factor)

            # Since the frequencies for each are changed,  we really need to
            # interpolate so that there is a common set of frequencies...

            # change this to use *args and **kwargs-- getting quite long..
            passband, center_freq, bin_width, snr, low_edge, upper_edge, frequencies = get_passband(
                interferogram, fts_step_size, frequency_calibration_factor,
                n_rms_iters=n_rms_iters,
                spike_threshold=spike_threshold, bin_min_freq=bin_min_freq,
                lower_bound=lower_bound, upper_bound=upper_bound,
                slope_cut=slope_cut, poly_order=poly_order,
                end_fit_freq=end_fit_freq, noise_bounds=noise_bounds,
                f_ignore_around_peak=f_ignore_around_peak, fit_noise=fit_noise,
                correction_func=correction_func,
                correction_params=correction_params,
                max_ind=max_ind, subtract_mean=subtract_mean,
                edge_power_limit=edge_power_limit, normalize=normalize,
                interp_freqs=common_frequencies,
                amplitude_transfer_func=amplitude_transfer_func,
                transfer_func_edges=transfer_func_edges)

            # Add a cut for SNR here.
            if passband is not None and (
                    snr > low_snr_cutoff and snr < high_snr_cutoff):

                # Cut for SNR to make sure we don't have any ridiculous ones
                if (snr > high_snr_change):
                    snr = high_snr_change

                passbands.append(passband)
                band_attrs.append([channel, center_freq, bin_width, snr,
                                   low_edge, upper_edge])

        passbands = np.array(passbands)
        band_attrs = np.array(band_attrs)
        total_passbands.append(passbands)
        total_band_attrs.append(band_attrs)

        if band_attrs != []:
            weights = band_attrs[:, 3] ** 2
        else:
            weights = []

        # Sum over a weighted avereage to create the average band
        if len(passbands) > 0:
            average_band = obtain_average_band(passbands, weights)
            bin_min = find_freq(frequencies, bin_min_freq)
            average_band = average_band / np.max(average_band[bin_min:])
        else:
            average_band = None

        average_bands.append(average_band)

        if output_vals and len(passbands) > 0 and plots:
            # plot the data now
            plot_band_data(i, passbands, band_attrs, average_band, frequencies,
                           weights, bin_min_freq=bin_min_freq,
                           plot_freq_range=plot_freq_range)

    return total_passbands, total_band_attrs, average_bands, frequencies


# returns index of frequency closest to desired value in frequency list
# input frequency in Hz and value in GHz
def find_freq(frequency, value):
    i = (np.abs(frequency / 1e9 - value)).argmin()
    return i

# normalizes a classified (90,150,220) passband so peak is 1
# uses lower and upper indices corresponding to frequency band range, so the highest value in the passband is chosen to normalize


def normalize_passband(passband, lower_index=0, upper_index=None):
    if (upper_index is None):
        upper_index = len(passband)
    return passband / np.max(passband[lower_index: upper_index])


# finds integration limits for a passband based on where the slope of the band begins to increase sharply
# integration limits defined by slope_cut
# input frequency in Hz
# upper_start and lower_start: starting search indices for frequency from which we move inward to find the point at which slope increases
# to plot, enter plot='plot'


def find_integration_limits(passband, frequency, lower_start_frequency,
                            upper_start_frequency, slope_cut, plot='no'):
    lower_start = find_freq(frequency, lower_start_frequency)
    upper_start = find_freq(frequency, upper_start_frequency)
    # print(lower_start, upper_start)
    # if we reach the end, just return without doing anything
    # try:
    lower_start_orig, upper_start_orig = lower_start, upper_start
    while ((passband[int(lower_start+2)]-passband[int(lower_start)])/((frequency[int(lower_start+2)]-frequency[int(lower_start)])/(1e9)) < slope_cut):
        lower_start += 1
    while ((passband[int(upper_start)]-passband[int(upper_start-2)])/((frequency[int(upper_start)]-frequency[int(upper_start-2)])/(1e9)) > -slope_cut):
        upper_start -= 1
    lower, upper = lower_start, upper_start
    # except IndexError:
    #    print('unable to integrate band...')
    # we reached the end:
    # return None, None

    if plot == 'plot':
        # matplotlib.rcParams.update({'font.size': 18})
        plt.figure(figsize=(12, 8))
        plt.plot(frequency/1e9, passband)
        plt.plot(frequency/1e9, passband, ".")
        plt.ylim([-0.2, np.max(passband[50:])])
        plt.xlim([frequency[lower_start_orig] / 1e9,
                  frequency[upper_start_orig] / 1e9])
        plt.axvline(frequency[lower] / 1e9)
        plt.axvline(frequency[upper] / 1e9)
        # plt.plot(np.array(
        #     [frequency[lower]/(1e9), frequency[lower]/(1e9)]), np.array([-0.2, 100]), "black")
        # plt.plot(np.array(
        #     [frequency[upper]/(1e9), frequency[upper]/(1e9)]), np.array([-0.2, 100]), "black")
        plt.plot(np.array([0, 1000]), np.array([0, 0]))
        plt.xlabel('Frequency (GHz)')
        # plt.show()
        print('Integration Limits:',
              frequency[lower]/(1e9), 'GHz,', frequency[upper]/(1e9), 'GHz')
        print(' ')
        print(' ')
    return int(lower_start_orig), int(upper_start_orig)
    return (int(lower), int(upper))

# integrates the portion of the passband data corresponding to the band in
# order to calculate weighted bandpass centers

# spectral index: the spectral dependence of the foreground source
# input frequency_range in Hz
# frequency_range and passband_range must have same dimension

# passband_range must be isolated chunk of full passband, determined by
# find_integration_limits upper and lower indices


def integrate_bands(frequency_range, passband_range, spectral_index):
    top = np.trapz(frequency_range**(-1. + spectral_index)
                   * passband_range, frequency_range)
    bottom = np.trapz(frequency_range**(-2. + spectral_index)
                      * passband_range, frequency_range)
    return top/bottom

# calculates weighted band centers (for synchrotron, free-free, Rayleigh-Jeans,
# and dust sources) for a single passband


def return_cent(passband, frequency, lower_bound_frequency,
                upper_bound_frequency, slope_cut, classification,
                spectral_index, plot):
    lower, upper = find_integration_limits(
        passband, frequency, lower_bound_frequency, upper_bound_frequency,
        slope_cut, plot)
    # max_index = np.argmax(passband[50:]) + 50
    # lower, upper = find_peak(passband, max_index)
    centers = [integrate_bands(frequency[int(lower):int(upper)], passband[int(
        lower):int(upper)], index) for index in spectral_index]
    centers.append(lower)
    centers.append(upper)
    return (centers)

# determines integration limits for band depending on itimeseries
# classification (90, 150, 220 = 1, 2, 3)
# calculates weighted band centers of # passband


def bandwidth(passband, frequency, lower, upper):
    # calculates the bandwidth of a passband
    # input frequency in Hz
    top = (np.trapz(passband[int(lower):int(upper)],
                    frequency[int(lower):int(upper)]))**2.
    bottom = (np.trapz(np.array(passband[int(lower):int(
        upper)])**2., frequency[int(lower):int(upper)]))
    return top/bottom/1e9


def bootstrap_resample(X):
    n = len(X)
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample


def bootstrap_resample(X, Y):
    # returns a resampled data set (of random length) from the original data
    # set
    assert len(X) == len(Y)
    n = len(X)
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample, Y[resample_i]


def bootstrap_plot(average_band, all_bands, frequencies, weights, iterations,
                   band_label, line_color, shade_color, norm_start_freq=20,
                   plot_start_freq=None, confidence_level=.95):
    # creates a bootstrap plot using all the bandpasses (plots average band
    # and 95% confidence limits)
    assert len(all_bands) > 0

    bootstrap_data = all_bands.T
    bootstrap_lower = []
    bootstrap_upper = []
    lower_index = int(((1 - confidence_level) / 2) * iterations)
    upper_index = int((confidence_level + (1 - confidence_level) / 2) *
                      iterations)

    # noise_start_index = find_freq(frequencies, noise_start_freq)
    # weights = np.std(all_bands[:, noise_start_index:], axis=1) ** -2
    # Here we want to apply the weights when taking the final meann..  So
    # however we resample all the data sets, we need to resample the weights
    # too!

    for i in range(len(bootstrap_data)):
        bootstrap_spread = []
        for _ in range(iterations):
            # change this mean to a weighted average!
            resampled_data, resampled_weights = bootstrap_resample(
                bootstrap_data[i], weights)
            weighted_average = np.average(
                resampled_data, weights=resampled_weights)
            bootstrap_spread.append(weighted_average)

        bootstrap_ordered = np.sort(bootstrap_spread)
        bootstrap_lower.append(bootstrap_ordered[lower_index])
        bootstrap_upper.append(bootstrap_ordered[upper_index])

    bootstrap_lower = np.array(bootstrap_lower)
    bootstrap_upper = np.array(bootstrap_upper)
    mean = (bootstrap_lower + bootstrap_upper) / 2.
    norm_start_index = find_freq(frequencies, norm_start_freq)
    # find the norm in the region we're interested in-
    # away from the 1/f early noise
    norm = np.max(mean[norm_start_index:])

    plt.figure(figsize=(8, 5))
    plt.plot(frequencies/1e9, average_band, color=line_color)
    plt.plot(frequencies / 1e9, bootstrap_lower / norm,
             color=shade_color, alpha=0.3)
    plt.plot(frequencies / 1e9, bootstrap_upper / norm,
             color=shade_color, alpha=0.3)
    plt.fill_between(frequencies / 1e9, bootstrap_lower / norm,
                     bootstrap_upper / norm, facecolor=shade_color,
                     alpha=0.3, interpolate='True')
    plt.ylim(-0.2, 1.2)
    if (plot_start_freq is None):
        plot_start_freq = norm_start_freq
    plt.xlim(plot_start_freq, np.max(frequencies / 1e9))
    plt.xlabel('Frequency (GHz)')
    plt.title('%s Ghz Band: Average + %s%% Confidence Limits' % (
        band_label, int(confidence_level * 100)))
    plt.show()
    return bootstrap_upper / norm, bootstrap_lower / norm


def bootstrap_save(filename, lower, upper, average, frequencies):
    with open(filename, 'wb') as f:
        np.savez(f, lower=lower, upper=upper, average=average,
                 frequencies=frequencies)
    return


def bootstrap_attrs(all_bands, all_weights, frequencies, iterations,
                    band_label, lower_bound, upper_bound, slope_cut,
                    plot_hists=False):
    attr_spread = []
    for _ in range(iterations):
        band_sample, weight_sample = bootstrap_resample(all_bands, all_weights)
        weighted_average = np.average(band_sample, weights=weight_sample,
                                      axis=0)
        center, lower_limit, upper_limit = return_cent(
            weighted_average, frequencies, lower_bound, upper_bound,
            slope_cut, -1, [0], 'False')
        center /= 1e9
        width = bandwidth(weighted_average, frequencies, lower_limit,
                          upper_limit)
        attr_spread.append([center, width])

    attr_spread = np.array(attr_spread)

    if (plot_hists):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle('Bootstraping bounds for average band center/width for %s'
                     ' Ghz band ' % band_label)
        axes[0].set_ylabel('counts')
        for i in range(2):
            data = attr_spread[:, i]
            axes[i].hist(data, label=r'%.3g $\pm$ %.3g' %
                         (np.mean(data), np.std(data)), bins=iterations // 50)
            axes[i].set_xlabel(['average center', 'average width'][i])
            axes[i].grid()
            axes[i].legend()

    return np.mean(attr_spread, axis=0), np.std(attr_spread, axis=0)


def bootstrap_integration_limits(
        average_band, frequencies, lower_bound_range, upper_bound_range,
        slope_cut, band_label, iterations=1000, plot=False):
    attr_spread = []
    lower_bounds = np.linspace(lower_bound_range[0], lower_bound_range[1],
                               iterations)
    upper_bounds = np.linspace(upper_bound_range[1], upper_bound_range[0],
                               iterations)
    # do we keep the bounds the same or do we mess with them?  probably best to
    # randomly sample over the given range to see if anything weird happens.
    # just set the slope cut to really small so that we just start where these
    # indices are!

    # since generally as we move the bounds inward /some/ (small) linear trend
    # starts to occur

    # other option is that we actually fit this linear function and interpolate
    # to where the band/peak really 'starts' and 'ends'
    for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
        # print(lower_bound, upper_bound)
        center, lower, upper = return_cent(
            average_band, frequencies, lower_bound, upper_bound, 1e-100, -1,
            [0], 'False')
        # print(lower, upper)
        width = bandwidth(average_band, frequencies, lower, upper)
        attr_spread.append([center / 1e9, width])

    attr_spread = np.array(attr_spread)
    if (plot):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle('Changes of center/width over varying integration limits'
                     ' for %s Ghz band ' % band_label)
        for i in range(2):
            data = attr_spread[:, i]
            axes[i].plot(lower_bounds, data)
            axes[i].set_xlabel('lower bound of integration')
            axes[i].set_ylabel(['centers', 'widths'][i])
            axes[i].grid()

    return attr_spread


def bootstrap_integration_limits_random(
        average_band, frequencies, lower_bound_range, upper_bound_range,
        slope_cut, iterations=1000):
    attr_spread = []
    lower_bounds = np.linspace(lower_bound_range[0], lower_bound_range[1],
                               iterations)
    upper_bounds = np.linspace(upper_bound_range[1], upper_bound_range[0],
                               iterations)
    # do we keep the bounds the same or do we mess with them?  probably best to
    # randomly sample over the given range to see if anything weird happens.
    # just set the slope cut to really small so that we just start where these
    # indices are!

    # since generally as we move the bounds inward /some/ (small) linear trend
    # starts to occur

    # other option is that we actually fit this linear function and interpolate
    # to where the band/peak really 'starts' and 'ends'
    lower_bounds_random = np.random.choice(lower_bounds, 1000)
    upper_bounds_random = np.random.choice(upper_bounds, 1000)

    for lower_bound, upper_bound in zip(lower_bounds_random,
                                        upper_bounds_random):
        center, lower, upper = return_cent(
            average_band, frequencies, lower_bound, upper_bound, 1e-100, -1,
            [0], 'False')
        width = bandwidth(average_band, frequencies, lower, upper)
        attr_spread.append([center / 1e9, width])

    attr_spread = np.array(attr_spread)
    return attr_spread


def get_all_band_items(passbands, band_attrs):
    all_passbands = []
    all_band_attrs = []
    for i in range(len(passbands)):
        assert len(passbands[i]) == len(band_attrs[i])
        for j in range(len(passbands[i])):
            band = passbands[i][j]
            attrs = band_attrs[i][j]
            all_passbands.append(band)
            all_band_attrs.append(attrs)
    return np.array(all_passbands), np.array(all_band_attrs)


def plot_band(frequencies, band, band_center, band_width, bin_min, plot_func,
              lower_bound, upper_bound, lower_edge, upper_edge):
    plot_func((frequencies / 1e9)[bin_min:], band[bin_min:])
    plt.axvline(band_center, color='black', label='center')
    plt.axvline(band_center + band_width / 2, color='gray', label='band edges')
    plt.axvline(band_center - band_width / 2, color='gray')

    plt.axvline(lower_bound, label='integration bounds')
    plt.axvline(upper_bound)

    plt.axvline(lower_edge, color='orange', label='5% band edges')
    plt.axvline(upper_edge, color='orange')

    # plt.axvline(frequencies[lower_limit] / 1e9, label='integration limits',
    #             color='green')
    # plt.axvline(frequencies[upper_limit] / 1e9, color='green')
    plt.legend()
    plt.show()
    print("band center: %s, band width: %s" % (band_center, band_width))
    print(70 * '-')


def run_through_bands(band_label, *passband_args, weight_func=np.square,
                      plots=False, **passband_kwargs):
    passbands, attrs, average_bands, frequencies = obtain_passbands(
        *passband_args, plots=plots, **passband_kwargs)
    all_passbands, all_band_attrs = get_all_band_items(passbands, attrs)

    bin_min = find_freq(frequencies, passband_kwargs['bin_min_freq'])
    lower_bound = passband_kwargs['lower_bound']
    upper_bound = passband_kwargs['upper_bound']
    norm_start_freq = bin_min

    for index, label in zip([1, 2, 4, 5], [
            'band centers', 'band widths', 'band lower edges',
            'band upper edges']):
        all_passbands, all_band_attrs = band_hist_cuts(
            all_passbands, all_band_attrs, index, label, band_label)

    if (plots):
        plot_band_snr_values(all_band_attrs, band_label)

    weights = weight_func(all_band_attrs[:, 3])
    total_average_band = normalize_passband(obtain_average_band(
        all_passbands, weights), lower_index=bin_min)

    average_center_freq, average_bandwidth, average_lower_edge, average_upper_edge = get_band_attrs(
        total_average_band, frequencies, lower_bound, upper_bound, 1e-10)

    if (plots):
        for plot_func in (plt.plot, plt.semilogy):
            plot_band(frequencies, total_average_band, average_center_freq,
                      average_bandwidth, 2, plot_func, lower_bound,
                      upper_bound, average_lower_edge, average_upper_edge)

    spread = bootstrap_integration_limits(
        total_average_band, frequencies, [lower_bound - 10, lower_bound],
        [upper_bound, upper_bound + 10], 1e-10, band_label, iterations=1000,
        plot=plots)
    if (plots):
        plt.show()

    means, stds = bootstrap_attrs(
        all_passbands, weights, frequencies, 1000,
        band_label, lower_bound, upper_bound, 1e-10, plot_hists=plots)
    if (plots):
        plt.show()

    if (plots):
        for i, attr_index in enumerate([1, 2, 4, 5]):
            label = ['center', 'width', 'lower edge', 'upper edge'][i]
            plot_colored_hist(
                all_band_attrs[:, attr_index], 20, all_band_attrs[:, 3],
                average=[average_center_freq, average_bandwidth,
                         average_lower_edge, average_upper_edge][i],
                label=label)
            plt.title('Band %s' % label + 's %s Ghz' % band_label)
            plt.show()

        # plot the SNR values
        fig, ax = plt.subplots(figsize=(7, 4))
        if 'high_snr_change' not in passband_kwargs:
            high_snr_change = 80
        else:
            high_snr_change = passband_kwargs['high_snr_change']
        snr_bins = np.arange(0, high_snr_change + high_snr_change // 20 + 1,
                             high_snr_change // 20)
        plt.hist(all_band_attrs[:, 3], bins=snr_bins)

        labels = list(map(int, ax.get_xticks().tolist()))
        labels[-2] = '%s+' % high_snr_change
        ax.set_xticklabels(labels)

        plt.title('SNR values %s Ghz' % band_label)
        plt.grid()
        plt.show()

    upper, lower = bootstrap_plot(
        total_average_band, all_passbands, frequencies, weights, 1000,
        band_label, 'r', 'lightpink', norm_start_freq=norm_start_freq,
        plot_start_freq=2, confidence_level=.95)

    # package these nicely into dictionaries to read
    total_band_data = {'passbands': all_passbands, 'attrs':  all_band_attrs}
    indiv_run_data = {'passbands': passbands, 'attrs': attrs,
                      'average_bands': average_bands}
    stat_data = {'means': means, 'stds': stds, 'spread': spread,
                 'upper': upper, 'lower': lower}
    return {'stat_data': stat_data, 'total_band_data': total_band_data,
            'individual_run_data': indiv_run_data, 'frequencies': frequencies,
            'total_average_band': total_average_band}


def get_band_attrs(band, frequencies, lower_bound, upper_bound, slope_cut,
                   bin_min=0):
    center_freq, lower_limit, upper_limit = return_cent(
        band, frequencies, lower_bound, upper_bound, slope_cut,
        -1, [0], plot='False')
    center_freq /= 1e9

    width = bandwidth(band, frequencies, lower_limit, upper_limit)
    # normalize the band before getting the edges
    normed_band = normalize_passband(band, lower_index=bin_min)
    lower_edge, upper_edge = get_band_edges(frequencies, normed_band)
    return center_freq, width, lower_edge, upper_edge


def get_repeats(total_good_band_channels, band_num, ch=None):
    # Add all these to a giant list and get the channel with the most counts
    # total_channels = []
    # for data_set in total_good_band_channels:
    #     good_channels = data_set[band_num]
    #     total_channels.extend(good_channels)
    # total_count = np.unique(total_channels, return_counts=True)
    # return total_count

    # Actually hasmap these with channel -> list of datasets
    channel_set_map = {}
    for i, data_set in enumerate(total_good_band_channels):
        for channel in data_set[band_num]:
            if channel not in channel_set_map.keys():
                channel_set_map[channel] = []
            channel_set_map[channel].append(i)

    if (ch is not None):
        return ch, channel_set_map[channel]
    max_channel = max(channel_set_map, key=lambda x: len(channel_set_map[x]))
    return max_channel, channel_set_map[max_channel]


def get_channel_dict(total_good_band_channels, band_num):
    channel_set_map = {}
    for i, data_set in enumerate(total_good_band_channels):
        for channel in data_set[band_num]:
            if channel not in channel_set_map.keys():
                channel_set_map[channel] = []
            channel_set_map[channel].append(i)

    return channel_set_map


def get_AB_channels(total_good_band_channels, array_filename):
    test = np.genfromtxt(array_filename, skip_header=19,
                         usecols=(0, 13), dtype=None)
    channels = []
    ab_vals = []
    for pair in test:
        channels.append(pair[0])
        ab_vals.append(pair[1].decode('utf-8'))
    channels = np.array(channels)
    ab_vals = np.array(ab_vals)

    a_channels = channels[np.where(ab_vals == 'A')]
    b_channels = channels[np.where(ab_vals == 'B')]
    total_good_a_channels = []
    total_good_b_channels = []
    for dataset in total_good_band_channels:
        dataset_a = []
        dataset_b = []
        for band_channels in dataset:
            band_a_channels = np.array(list(filter(
                lambda ch: ch in a_channels, band_channels)))
            dataset_a.append(band_a_channels)

            band_b_channels = np.array(list(filter(
                lambda ch: ch in b_channels, band_channels)))
            dataset_b.append(band_b_channels)
        total_good_a_channels.append(dataset_a)
        total_good_b_channels.append(dataset_b)
    return np.array(total_good_a_channels, dtype='object'), np.array(
        total_good_b_channels, dtype='object')


def get_centroid(total_good_channels, x_positions, y_positions):
    if len(total_good_channels) == 0:
        return None
    good_x = x_positions[total_good_channels]
    good_y = y_positions[total_good_channels]
    good_points = np.array([good_x, good_y]).T
    return np.sum(good_points, axis=0) / len(good_points)


def get_pixel_position(channel, x_positions, y_positions):
    # getting the centroid using only this one pixel will simply yield the
    # pixel.
    return get_centroid([channel], x_positions, y_positions)


# def frequency_calibration_func(x_pos, y_pos):
#     a = .0585
#     b = .050
#     # right now this is unaffected by the x position
#     return a * y_pos + (b + 1)


def frequency_calibration_func(x_pos, y_pos):
    # fitted from mathematica. y_pos is in mm!
    a, b, c, d = (1.055031284080443, 0.0009717425200561042,
                  -0.000010387751028142605,  -6.163499394928664e-7)

    # we only care about the y position for now.
    return (a + b * y_pos + c * y_pos ** 2 + d * y_pos ** 3) * (1.0239 / 1.055)


def find_closest_pixel_pair(x, y, pairs):
    # go smallest in total distance apart
    return min(pairs, key=lambda pair: np.sqrt(
        np.sum(np.subtract(pair, [x, y]) ** 2)))


def amplitude_transfer_func(x_pos, y_pos, order=7):
    # load in the correction file containing all the corrections.
    correction_file = np.load('amp_correction_z_0_35.npz', 'wb')
    # data should be an array who has a (x, y) pair for location, each of which
    # has an amplitude correction.
    amplitudes = correction_file['amplitudes']
    pairs = correction_file['pairs']
    frequencies = correction_file['frequencies']

    # get the xy pair closest to ours
    # print(x_pos, y_pos)
    closest_pair = find_closest_pixel_pair(x_pos, y_pos, pairs)
    # print(closest_pair)
    pair_index = np.where(pairs == closest_pair)[0][0]

    # divide by the maximum to roughly normalize.
    amplitudes[pair_index] /= np.max(amplitudes[pair_index])
    poly_params = np.polyfit(frequencies, amplitudes[pair_index], order)
    poly_fitted_amplitudes = np.polyval(poly_params, frequencies)

    # want a function that will output frequency -> amplitude
    transfer_func = interpolate.interp1d(frequencies, poly_fitted_amplitudes,
                                         fill_value='extrapolate')
    return transfer_func


def get_frequency_calibration_factor(centroid, pixel_position):
    # these are flipped with respect to what we have in the FTS sims!
    x_difference = (pixel_position[1] - centroid[1])
    y_difference = (pixel_position[0] - centroid[0])
    return frequency_calibration_func(x_difference, y_difference)


def get_amplitude_transfer_func(centroid, pixel_position):
    # these are flipped with respect to what we have in the FTS sims!
    y_difference = (pixel_position[0] - centroid[0])
    x_difference = (pixel_position[1] - centroid[1])
    return amplitude_transfer_func(x_difference, y_difference)


def get_amplitudes(data_sets, channels):
    amps = []
    for data_set in data_sets:
        for channel in channels:
            # print(channel)
            interferogram = np.sqrt(data_set[:, channel])
            amps.append(np.max(interferogram))

    # print(np.array(amps))
    test = np.array(amps) / 1e7
    # print(test)
    return test


def get_centroid_response(run_num, band_num, total_good_channels, array_data,
                          x_locs, y_locs, *passband_args,
                          **passband_kwargs):
    good_channels = total_good_channels[run_num][band_num]
    good_x = x_locs[good_channels]
    good_y = y_locs[good_channels]
    good_points = np.array([good_x, good_y]).T
    if len(good_points) == 0:
        return None, None

    centroid = np.sum(good_points, axis=0) / len(good_points)
    # now plot top half, bottom half, left half, right half
    bottom_half = np.where(good_points[:, 1] < centroid[1])
    top_half = np.where(good_points[:, 1] >= centroid[1])

    # print(good_channels[top_half])
    if passband_kwargs['plots']:
        plt.scatter(good_x[top_half], good_y[top_half], c='green',
                    s=get_amplitudes([array_data[run_num]], good_channels[top_half]))
        plt.scatter(good_x[bottom_half], good_y[bottom_half], c='red',
                    s=get_amplitudes([array_data[run_num]], good_channels[bottom_half]))
        plt.show()

    left_half = np.where(good_points[:, 0] < centroid[0])
    right_half = np.where(good_points[:, 0] >= centroid[0])

    if passband_kwargs['plots']:
        plt.scatter(good_x[left_half], good_y[left_half], c='red')
        plt.scatter(good_x[right_half], good_y[right_half], c='green')
        plt.show()

    total_averages = []
    total_attrs = []

    lower_bound = passband_kwargs['lower_bound']
    upper_bound = passband_kwargs['upper_bound']

    # NEED TO MODIFY THIS TO TAKE THE CENTROID INTO ACCOUNT!!!
    for split in [top_half, bottom_half, left_half, right_half]:
        passbands, attrs, average_bands, frequencies = obtain_passbands(
            band_num, [array_data[run_num]],
            [[good_channels[split], good_channels[split]]], *passband_args,
            centroid=centroid, **passband_kwargs)
        # if there's only 1 point on each side, we don't have enough info
        # to make a reasonable estimate.
        if len(passbands[0]) <= 0:
            return None, None

        average_bands[0] /= np.max(average_bands[0])
        total_averages.append(average_bands[0])
        total_attrs.append(attrs)

    labels = ['top half', 'bottom half', 'left half', 'right half']

    average_attrs = []
    for i, (attrs, average) in enumerate(zip(total_attrs, total_averages)):
        # average_center_freq, lower_limit, upper_limit = return_cent(
        #     average, frequencies, lower_bound, upper_bound, 1e-10,
        #     -1, [0], plot='False')
        # average_center_freq /= 1e9

        # average_bandwidth = bandwidth(average, frequencies,
        #                               lower_limit, upper_limit)

        average_center_freq, average_bandwidth, lower_edge, upper_edge = get_band_attrs(average, frequencies, lower_bound, upper_bound,
                                                                                        1e-10)

        average_attrs.append([average_center_freq, average_bandwidth,
                              lower_edge, upper_edge])

        # print(attrs[0][:, 1])
        # snrs = np.array(attrs[0][:, 3]) ** 2
        # average_center_freq = np.average(
        #     np.array(attrs)[0][:, 1], weights=snrs)
        # average_bandwidth = np.average(np.array(attrs)[0][:, 2], weights=snrs)
        # average_low_edge, _ = smart_rms(
        #     np.array(attrs)[0][:, 4], 4, 2)
        # average_upper_edge, _ = smart_rms(
        #     np.array(attrs)[0][:, 5], 4, 2)
        # average_attrs.append([average_center_freq, average_bandwidth,
        #                       average_low_edge, average_upper_edge])

        if passband_kwargs['plots']:
            plt.plot(frequencies / 1e9, average, label='%s: %.1f %.1f' % (
                labels[i], average_center_freq, average_bandwidth))
            print(average_center_freq, average_bandwidth)

    if passband_kwargs['plots']:
        plt.legend()
        plt.title('centers and widths over portions of the array'
                  'for run %s' % (run_num))
        plt.show()
    return np.array(average_attrs), total_attrs


def get_band_edges_manually(passband, limit=.05):
    # go until we hit .05
    current_index = np.argmax(passband)
    # get the low edge
    while (np.abs(passband[current_index]) > limit):
        current_index += 1
    end_index = current_index

    current_index = np.argmax(passband)
    # get the low edge
    while (passband[current_index] > limit):
        current_index -= 1
    start_index = current_index
    return (start_index, end_index)


def cubic(x, a, b, c):
    return a * x ** 3 + b * x ** 2 + c


def fit_band_edge(frequencies, passband, edge_guess_ind, plot=False,
                  limit=.05):
    # go, say, 2 points below and 2 points above to fit this edge
    ind_limit = 4
    fit_x = frequencies[edge_guess_ind - ind_limit: edge_guess_ind + ind_limit]
    fit_y = passband[edge_guess_ind - ind_limit: edge_guess_ind + ind_limit]
    popt, _ = optimize.curve_fit(cubic, fit_x, fit_y)
    # want to find the edge where we have (limit * 100)% power
    a, b, c = popt
    roots = np.roots([a, b, 0, c - limit])
    # take the one closest to our edge this is probably not completely
    # correct... make sure to only look at reals
    # roots = roots[np.iscomplex(np.real_if_close(roots)) == False].real
    roots = roots.real
    closest_root = min(roots, key=lambda x: abs(
        x - frequencies[edge_guess_ind]))
    if (plot):
        plt.plot(fit_x, cubic(fit_x, *popt))
        plt.plot(fit_x, fit_y)
        plt.axvline(closest_root)
        plt.show()
    # if the fitis really off, just return the earlier guess.
    if (closest_root) < 0 or closest_root > 300 or np.abs(
            frequencies[edge_guess_ind] - closest_root) > 4:
        return edge_guess_ind
    return closest_root


def get_band_edges(frequencies, passband, limit=.05, plot=False):
    # make sure this band is normalized.
    if np.max(passband) != 1:
        return None, None

    try:
        start, end = get_band_edges_manually(passband, limit=limit)
        low_edge = fit_band_edge(frequencies / 1e9, passband, start, plot=plot)
        upper_edge = fit_band_edge(frequencies / 1e9, passband, end, plot=plot)
    except IndexError:
        return None, None
    return low_edge, upper_edge


def score_fit(data_portion, expected_portion):
    residual = data_portion - expected_portion
    # minimizethe residual
    score = np.sum(np.abs(residual))
    return score


def get_band_rolloff_frequencies(frequencies, passband, expected_rolloff):
    # take a chunk of the rolloff
    min_score = np.inf  # maybe only do this for a portion
    min_index = None
    for i in range(len(frequencies) - len(expected_rolloff)):
        # get the portion of our data that we want to fit
        data_portion = passband[i: i + len(expected_rolloff)]
        score = score_fit(data_portion, expected_rolloff)
        if (score < min_score):
            min_index = i
            min_score = score
    return (frequencies[min_index: min_index + len(expected_rolloff)],
            passband[min_index: min_index + len(expected_rolloff)], min_score)


def save_center_and_width(
        attrs, array_label, band_label, save_dir='centers_and_widths',
        fname='take1', start_file=True):
    center_mean, center_dev = weighted_rms(
        attrs['total_band_data']['attrs'][:, 1],
        attrs['total_band_data']['attrs'][:, 3] ** 2)

    width_mean, width_dev = weighted_rms(
        attrs['total_band_data']['attrs'][:, 2],
        attrs['total_band_data']['attrs'][:, 3] ** 2)

    open_arg = 'a'
    if (start_file):
        open_arg = 'w'
    with open(save_dir + '/%s.csv' % fname, open_arg) as fd:
        if (start_file):
            fd.write('Array, Band, Center (Ghz), Width (Ghz)')
        fd.write('\n %s, %s, %.1f $\\pm$ %.1f, %.1f $\\pm$ %.1f' % (
            array_label, band_label, center_mean, center_dev, width_mean,
            width_dev))


def plot_chunk_hists(chunk_attrs, attr_labels, write_data=False,
                     start_file=False, save_dir='', fname='', array_label='',
                     band_label='', corrected=False):
    save_data = []
    labels = ['top half', 'bottom half', 'left half', 'right half']
    m, n = (3, 2)
    for value in range(len(attr_labels)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for i, (m, n) in enumerate([[3, 2], [1, 0]]):
            data = np.array(chunk_attrs)[:, m, value] - np.array(chunk_attrs)[
                :, n, value]
            mean, rms, data = smart_rms(data, 2, 3, return_data=True)
            axes[i].hist(data, label=r'%.2f $\pm$ %.2f' % (
                mean, rms))
            axes[i].legend()
            axes[i].set_title('%s %s - %s %s' % (
                labels[m], attr_labels[value], labels[n], attr_labels[value]))
            axes[i].grid()
            # only save the right - left data
            if (m, n) == (3, 2):
                save_data.append([mean, rms])
            # plt.show()

    # now save this data to a file
    if not write_data:
        return
    open_arg = 'a'
    if (start_file):
        open_arg = 'w'
    with open(save_dir + '/%s.csv' % fname, open_arg) as fd:
        if (start_file):
            fd.write('Array, Band, Attr, Mean (Ghz), RMS (Ghz), Corrected?')

        # now loop through and save all of our data.
        for ((mean, rms), attr_label) in zip(save_data, attr_labels):
            fd.write('\n %s, %s, %s, %.2f, %.2f, %s' % (
                array_label, band_label, attr_label, mean, rms, corrected))

    return save_data


def spatial_variation(attr_data, x_positions, y_positions, attr_index):
    # list of channels, centers, widths, snrs, edges, bands itself?
    all_attrs = attr_data['total_band_data']['attrs']
    x_rounded = np.round(x_positions, 2)
    y_rounded = np.round(y_positions, 2)
    data = []
    # now for each x position get a list of channels belonging to that position
    unique_positions = np.unique(np.array([x_rounded, y_rounded]).T, axis=0)
    for pixel_position in unique_positions:
        pixel_channels = np.where((x_rounded == pixel_position[0]) & (
            y_rounded == pixel_position[1]))[0]

        # Now get the bands for these channels and plot them/find the mean
        # center and width, low and upper edge
        common_vals, channel_ind, pixel_ind = np.intersect1d(
            np.asarray(all_attrs[:, 0], dtype='int'), pixel_channels,
            return_indices=True)
        pixel_attrs = all_attrs[channel_ind]
        if (len(pixel_attrs) == 0):
            data.append(np.nan)
            continue
        mean, _ = weighted_rms(pixel_attrs[:, attr_index], pixel_attrs[:, 3])
        data.append(mean)

    return np.array(data), unique_positions


def plot_spatial_variation(attr_data, x_positions, y_positions, attr_index):
    attrs, poses = spatial_variation(attr_data, x_positions, y_positions,
                                     attr_index)
    attr_mean, _ = weighted_rms(
        attr_data['total_band_data']['attrs'][:, attr_index],
        attr_data['total_band_data']['attrs'][:, 3])
    plt.figure(figsize=(8, 5))

    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0)
    plt.scatter(x_positions, y_positions, s=.1, c='black', alpha=.4)
    plt.scatter(poses[:, 0], poses[:, 1], c=100 * (
        attrs - attr_mean) / attr_mean, cmap='RdBu', edgecolors='b', norm=norm)
    plt.colorbar(label='percent shift')
    plt.xlabel('y (mm)')
    plt.ylabel('z (mm)')
    plt.grid(False)


def get_top_repeats(set_map, n=5):
    ch_array = np.zeros((len(set_map), 2), dtype='int')
    # first get an array with ch, # of channels
    for i, ch in enumerate(set_map.keys()):
        ch_array[i] = [ch, len(set_map[ch])]
    return ch_array[np.argsort(ch_array[:, 1])][::-1][:n][:, 0]


def channel_spread(ch, run_indices, all_attrs):
    plt.figure(figsize=(10, 5))
    freqs = all_attrs['frequencies']
    for run in run_indices:
        channel_ind = np.where(
            all_attrs['individual_run_data']['attrs'][run][:, 0] == ch)[0]
        if len(channel_ind) != 1:
            continue
        channel_ind = channel_ind[0]
        passband = all_attrs['individual_run_data']['passbands'][run][channel_ind]
        center, width, snr, low_edge, upper_edge = all_attrs[
            'individual_run_data']['attrs'][run][channel_ind][1:]
        plt.plot(freqs / 1e9, passband, alpha=.8, label='run %s: %.1f %.3g, %.3g %.1f %.1f' % (
            run, center, width, snr, low_edge, upper_edge))

    # Plot total average band
    plt.plot(freqs / 1e9, all_attrs['total_average_band'], '--', color='black',
             label='average band over all detectors', alpha=.8)
    plt.title('Spread of channel %s bands over various runs (PA4)' % ch)
    plt.legend(loc='upper left', bbox_to_anchor=(0, -.2))
    plt.xlabel('frequency (Ghz)')
    plt.ylabel('normalized passband amplitude')
