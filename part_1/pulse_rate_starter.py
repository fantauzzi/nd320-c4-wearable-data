#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt

plt.ion()
from matplotlib import get_backend
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
import glob
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
# from scipy import signal
from scipy.stats import describe


# ## Part 1: Pulse Rate Algorithm
# 
# ### Contents
# Fill out this notebook as part of your final project submission.
# 
# **You will have to complete both the Code and Project Write-up sections.**
# - The [Code](#Code) is where you will write a **pulse rate algorithm** and already includes the starter code.
#    - Imports - These are the imports needed for Part 1 of the final project. 
#      - [glob](https://docs.python.org/3/library/glob.html)
#      - [numpy](https://numpy.org/)
#      - [scipy](https://www.scipy.org/)
# - The [Project Write-up](#Project-Write-up) to describe why you wrote the algorithm for the specific case.
# 
# 
# ### Dataset
# You will be using the **Troika**[1] dataset to build your algorithm. Find the dataset under `datasets/troika/training_data`. The `README` in that folder will tell you how to interpret the data. The starter code contains a function to help load these files.
# 
# 1. Zhilin Zhang, Zhouyue Pi, Benyuan Liu, ‘‘TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type Photoplethysmographic Signals During Intensive Physical Exercise,’’IEEE Trans. on Biomedical Engineering, vol. 62, no. 2, pp. 522-531, February 2015. Link
# 
# -----

# ### Code

# In[ ]:


def load_troika_dataset():
    """
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the 
            reference data for data_fls[5], etc...
    """
    data_dir = "./datasets/troika/training_data"
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls


def load_troika_data_file(data_fl):
    """
    Loads and extracts signals from a troika data file.

    Usage:
        data_fls, ref_fls = LoadTroikaDataset()
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fls[0])

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        numpy arrays for ppg, accx, accy, accz signals.
    """
    data = sp.io.loadmat(data_fl)['sig']
    return data[2:]


def load_troika_ground_truth(ref_fl):
    data = sp.io.loadmat(ref_fl)['BPM0']
    data = np.squeeze(data)
    return data


def aggregate_error_metric(pr_errors, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability. 

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding 
            reference heart rates.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        the MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estimates = pr_errors[confidence_est >= percentile90_confidence]

    # Return the mean absolute error
    return np.mean(np.abs(best_estimates))


def evaluate():
    """
    Top-level function evaluation function.

    Runs the pulse rate algorithm on the Troika dataset and returns an aggregate error metric.

    Returns:
        Pulse rate error on the Troika dataset. See AggregateErrorMetric.
    """
    # Retrieve dataset files
    data_fls, ref_fls = load_troika_dataset()
    errs, confs = [], []
    for data_fl, ref_fl in zip(data_fls, ref_fls):
        # Run the pulse rate algorithm on each trial in the dataset
        errors, confidence = run_pulse_rate_algorithm(data_fl, ref_fl)
        errs.append(errors)
        confs.append(confidence)
        # Compute aggregate error metric
    errs = np.hstack(errs)
    confs = np.hstack(confs)
    return aggregate_error_metric(errs, confs)


def run_pulse_rate_algorithm(data_fl, ref_fl):
    # Load data using LoadTroikaDataFile
    ppg, accx, accy, accz = load_troika_data_file(data_fl)

    # Compute pulse rate estimates and estimation confidence.

    # Return per-estimate mean absolute error and confidence as a 2-tuple of numpy arrays.
    errors, confidence = np.ones(100), np.ones(100)  # Dummy placeholders. Remove
    return errors, confidence


# -----
# ### Project Write-up
# 
# Answer the following prompts to demonstrate understanding of the algorithm you wrote for this specific context.
# 
# > - **Code Description** - Include details so someone unfamiliar with your project will know how to run your code and use your algorithm. 
# > - **Data Description** - Describe the dataset that was used to train and test the algorithm. Include its short-comings and what data would be required to build a more complete dataset.
# > - **Algorithhm Description** will include the following:
# >   - how the algorithm works
# >   - the specific aspects of the physiology that it takes advantage of
# >   - a describtion of the algorithm outputs
# >   - caveats on algorithm outputs 
# >   - common failure modes
# > - **Algorithm Performance** - Detail how performance was computed (eg. using cross-validation or train-test split) and what metrics were optimized for. Include error metrics that would be relevant to users of your algorithm. Caveat your performance numbers by acknowledging how generalizable they may or may not be on different datasets.
# 
# Your write-up goes here...

# -----
# ### Next Steps
# You will now go to **Test Your Algorithm** (back in the Project Classroom) to apply a unit test to confirm that your algorithm met the success criteria. 


def bandpass_filter(the_signal, pass_band, fs):
    # noinspection PyTupleAssignmentBalance
    b, a = sp.signal.butter(5, pass_band, btype='bandpass', fs=fs)  # TODO try with order != 5
    res = sp.signal.filtfilt(b, a, the_signal)
    return res


def get_time_scale(data, freq):
    res = np.linspace(start=0, stop=len(data) / freq / 60, num=len(data), endpoint=True)
    return res


def clear_axis(axis):
    for ax in axis:
        ax.clear()


def wait_for_key():
    while not plt.waitforbuttonpress(timeout=0):
        ...


def plot_windowed_fft(ax, color, data, window_start, window_size, frequency):
    fft = np.fft.rfft(data[window_start:window_start + window_size], n=len(data))
    freq = np.fft.rfftfreq(len(data), 1 / frequency)
    ax.plot(freq, np.abs(fft), color=color)


def main():
    print('Using backend', get_backend())
    fs = 125  # Sampling frequency, Hz
    fgt = .5  # Ground truth sample frequency, Hz
    data_fls, ref_fls = load_troika_dataset()
    assert len(data_fls) == len(ref_fls)
    runs = []
    for data_fl, ref_fl in zip(data_fls, ref_fls):
        data = load_troika_data_file(data_fl)
        gt_data = load_troika_ground_truth(ref_fl)
        run = {'ppg': data[0], 'imu': data[1:4], 'gt': gt_data}
        runs.append(run)

    for run in runs:
        run['ppg_filtered'] = bandpass_filter(run['ppg'], (40 / 60, 240 / 60), fs)
        run['imu_filtered'] = np.empty_like(run['imu'])
        for i in range(3):
            run['imu_filtered'][i] = bandpass_filter(run['imu'][i], (40 / 60, 240 / 60), fs)
        run['imu_magnitude'] = np.sqrt(
            (run['imu_filtered'][0] ** 2 + run['imu_filtered'][1] ** 2 + run['imu_filtered'][2] ** 2))

    colors = ('orange', 'blue', 'purple')

    def update(value):
        gt_idx = int(value * fgt+4*fgt)
        gt_value = run['gt'][gt_idx] / 60 if gt_idx < len(run['gt']) else None
        ax[5].clear()
        plot_windowed_fft(ax=ax[5],
                          color='lime',
                          data=run['ppg_filtered'],
                          window_start=int(value) * fs,
                          window_size=8 * fs,
                          frequency=fs)
        if gt_value is not None:
            ax[5].axvline(x=gt_value, color='black', linestyle='dotted')
        ax[5].set_xlim((0, 5))
        for i in range(3):
            ax[2 + i].clear()
            plot_windowed_fft(ax=ax[2 + i],
                              color=colors[i],
                              data=run['imu_filtered'][i],
                              window_start=int(value) * fs,
                              window_size=8 * fs,
                              frequency=fs)
            if gt_value is not None:
                ax[2 + i].axvline(x=gt_value, color='black', linestyle='dotted')
            ax[2 + i].set_xlim((0, 5))
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(7, 1)

    for run in runs:
        t = len(run['ppg']) / fs
        # Plot the 3 IMU axes
        ts_imu = get_time_scale(run['imu_magnitude'], fs)
        for i in range(3):
            ax[0].plot(ts_imu, run['imu'][i], color=colors[i])
        ts_ppg = get_time_scale(run['ppg_filtered'], fs)
        ax[1].plot(ts_ppg, run['ppg_filtered'], color='green')
        # Plot FT of filtered IMU magnitude
        update(0)
        slider = Slider(ax=ax[-1], label='Time', valmin=0, valmax=int(t - 8), valinit=0, valstep=1)
        slider.on_changed(update)
        wait_for_key()
        clear_axis(ax)

    plt.close(fig)

    # Stitch accel. magnitude and PPG data across runs to easily compute their stats later on
    all_accel_magnitude = np.empty((0,), dtype=float)
    all_ppg = np.empty((0,), dtype=float)

    fig, ax = plt.subplots(3, 1)
    for run in runs:
        # Plot the three axes of the IMU
        all_accel_magnitude = np.concatenate((all_accel_magnitude, run['imu_magnitude']))
        all_ppg = np.concatenate((all_ppg, run['ppg']))
        ts_data = get_time_scale(run['ppg'], fs)
        ts_gt = get_time_scale(run['gt'], fgt)
        colors = ('orange', 'blue', 'purple')
        for i in range(3):
            ax[0].plot(ts_data, run['imu'][i], color=colors[i])
        # Plot PPG
        ax[1].plot(ts_data, run['ppg'], color='green')
        # Plot ground truth (HR)
        ax[2].plot(ts_gt, run['gt'], color='red')
        wait_for_key()
        clear_axis(ax)

    plt.close(fig)

    print('Stats for IMU acc. magnitude', describe(all_accel_magnitude))
    print('Stats for PPG', describe(all_ppg))

    # Hystogram for IMU acc. magnitude and for PPG value
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(all_accel_magnitude, bins=20, log=True)
    ax[0].set_title('Histogram of accel. magnitude')
    ax[1].hist(all_ppg, bins=20, log=True)
    ax[1].set_title('Histogram of PPG data')
    wait_for_key()
    plt.close(fig)

    ''' It looks like PPG data are clipped to their min and max value; also, the max value is probably replacement 
    for no value available (for instance when the PPG temporarily lost contact with the skin). '''
    print(all_ppg[all_ppg == max(all_ppg)].sum())
    print(all_ppg[all_ppg == min(all_ppg)].sum())

    fig, ax = plt.subplots(4, 1)
    for run in runs:
        gt_run_ts = get_time_scale(run['gt'], fgt)
        # Plot filtered vs. non filtered PPG sensor values
        ppg_run_filt = bandpass_filter(run['ppg'], (40 / 60, 240 / 60), fs)
        ppg_run_ts = get_time_scale(run['ppg'], fs)
        ax[0].plot(ppg_run_ts, ppg_run_filt, color='green')
        ax[0].plot(ppg_run_ts, run['ppg'], color='lime')

        # Plot FT of filtered PPG
        fft = np.fft.rfft(ppg_run_filt)
        freq = np.fft.rfftfreq(len(ppg_run_filt), 1 / fs)
        ax[1].plot(freq, np.abs(fft), color='green')

        # Plot HR against a spectrogram of the filtered PPG signal
        # Using an 8 sec window with 6 sec overlap because it is what adopted by the HR ground truth
        t = len(run['ppg']) / fs
        ax[2].specgram(ppg_run_filt, Fs=fs, NFFT=8 * fs, noverlap=6 * fs, xextent=((0, t)))
        gt_run_ts2 = np.linspace(start=0, stop=len(run['gt']) / fgt, num=len(run['gt']), endpoint=True)
        ax[2].plot(gt_run_ts2, run['gt'] / 60, '.', color='red')
        ax[2].set_ylim((0, 5))

        # Plot HR against a spectrogram of the acceleration magnitude
        # Same window and overlap as above
        acc_fft = np.fft.rfft(run['imu_magnitude'])
        # acc_freq = np.fft.rfftfreq(len(accel_filt), 1 / fs)
        ax[3].specgram(acc_fft, Fs=fs, NFFT=8 * fs, noverlap=6 * fs, xextent=((0, t)))
        ax[3].plot(gt_run_ts2, run['gt'] / 60, '.', color='red')
        ax[3].set_ylim((0, 5))

        # ax[3].plot(gt_run_ts, gt_run / 60, color='red')
        wait_for_key()
        clear_axis(ax)

    plt.close(fig)


if __name__ == '__main__':
    main()
