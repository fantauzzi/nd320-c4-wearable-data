#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt

plt.ion()
from matplotlib import get_backend
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


def BandpassFilter(the_signal, pass_band, fs):
    b, a = sp.signal.butter(5, pass_band, btype='bandpass', fs=fs)
    res = sp.signal.filtfilt(b, a, the_signal)
    return res


def get_time_scale(data, freq):
    res = np.linspace(start=0, stop=len(data) / freq / 60, num=len(data), endpoint=True)
    return res


def clear_axis(axis):
    for ax in axis:
        ax.clear()


def main():
    print('Using backend', get_backend())
    fs = 125  # Sampling frequency, Hz
    fgt = .5  # Ground truth sample frequency, Hz
    data_fls, ref_fls = load_troika_dataset()
    assert len(data_fls) == len(ref_fls)
    n_runs = len(data_fls)
    accel = []
    ppg = []
    gt = []
    for data_fl, ref_fl in zip(data_fls, ref_fls):
        data = load_troika_data_file(data_fl)
        gt_data = load_troika_ground_truth(ref_fl)
        accel.append(data[1:4])
        ppg.append(data[0])
        gt.append(gt_data)

    all_accel_magnitude = np.empty((0,), dtype=float)
    all_ppg = np.empty((0,), dtype=float)
    fig, ax = plt.subplots(3, 1)
    for accel_run, ppg_run, gt_run in zip(accel, ppg, gt):
        magnitude = np.sqrt(accel_run[0] ** 2 + accel_run[1] ** 2 + accel_run[2] ** 2)
        all_accel_magnitude = np.concatenate((all_accel_magnitude, magnitude))
        all_ppg = np.concatenate((all_ppg, ppg_run))
        ts_data = get_time_scale(ppg_run, fs)
        ts_gt = get_time_scale(gt_run, fgt)
        colors = ('orange', 'blue', 'purple')
        for i in range(3):
            ax[0].plot(ts_data, accel_run[i], color=colors[i])
        ax[1].plot(ts_data, ppg_run, color='green')
        ax[2].plot(ts_gt, gt_run, color='red')
        plt.waitforbuttonpress(timeout=0)
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
    plt.waitforbuttonpress(timeout=0)
    plt.close(fig)

    ''' It looks like PPG data are clipped to their min and max value; also, the max value is probably replacement 
    for no value available (for instance when the PPG temporarily lost contact with the skin). '''
    print(all_ppg[all_ppg == max(all_ppg)].sum())
    print(all_ppg[all_ppg == min(all_ppg)].sum())

    fig, ax = plt.subplots(4, 1)
    for accel_run, ppg_run, gt_run in zip(accel, ppg, gt):
        ppg_run_filt = BandpassFilter(ppg_run, (40 / 60, 240 / 60), fs)
        fft = np.fft.rfft(ppg_run_filt)
        freq = np.fft.rfftfreq(len(ppg_run_filt), 1 / fs)
        ppg_run_ts = get_time_scale(ppg_run, fs)
        gt_run_ts = get_time_scale(gt_run, fgt)
        t = len(ppg_run) / fs
        ax[0].plot(ppg_run_ts, ppg_run, color='green')
        ax[1].plot(freq, np.abs(fft), color='green')
        ax[2].specgram(ppg_run_filt, Fs=fs, NFFT=1 * fs, noverlap=0, xextent=((0, t)))
        gt_run_ts2 = np.linspace(start=0, stop=len(gt_run) / fgt, num=len(gt_run), endpoint=True)
        ax[2].plot(gt_run_ts2, gt_run / 60, '.', color='red')
        ax[2].set_ylim((0, 4))
        ax[3].plot(gt_run_ts, gt_run / 60, color='red')
        plt.waitforbuttonpress(timeout=0)
        clear_axis(ax)
        
    plt.close(fig)


if __name__ == '__main__':
    main()
