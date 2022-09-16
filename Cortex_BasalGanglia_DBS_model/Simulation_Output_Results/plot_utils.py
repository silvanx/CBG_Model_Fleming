import scipy.io as sio
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Any
from numpy.typing import NDArray


def mat_to_dict(obj: Any) -> dict:
    '''Reads a matlab struct and turns it into a dictionary'''
    return dict(zip((e[0] for e in obj.dtype.descr), obj))


def load_dbs_output(dir: Path) -> tuple[dict, dict]:
    '''Reads DBS current data from a .mat file'''
    dbs_file = sio.loadmat(dir / 'DBS_Signal.mat')
    segments, _, _ = dbs_file['block'][0, 0]
    segment = mat_to_dict(segments[0, 0][0, 0])
    signal0, signal1 = segment['analogsignals'][0]
    dbs = mat_to_dict(signal0[0, 0])
    time = mat_to_dict(signal1[0, 0])

    return time, dbs


def load_stn_lfp(dir: Path, steady_state_time: float, sim_time: float)\
        -> tuple[NDArray[np.float32], dict]:
    '''Reads STN LFP data from a .mat file'''
    lfp_file = sio.loadmat(dir / 'STN_LFP.mat')
    segments, _, _ = lfp_file['block'][0, 0]
    segment = mat_to_dict(segments[0, 0][0, 0])
    lfp = mat_to_dict(segment['analogsignals'][0, 0][0, 0])
    lfp_t = np.linspace(steady_state_time, sim_time, len(lfp['signal']))
    return lfp_t, lfp


def load_controller_data(dir: Path, parameter: str)\
        -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    '''Reads controller data from CSV files'''
    with open(dir / 'controller_sample_times.csv', 'r') as f:
        controller_t = np.array([float(r[0]) for r in csv.reader(f)])
    with open(dir / ('controller_%s_values.csv' % parameter), 'r') as f:
        controller_p = np.array([float(r[0]) for r in csv.reader(f)])
    with open(dir / 'controller_beta_values.csv', 'r') as f:
        controller_b = np.array([float(r[0]) for r in csv.reader(f)])
    return controller_t, controller_p, controller_b


def time_to_sample(tt: np.ndarray, t: float) -> int:
    '''Finds the first index in a time array where time is greater than or
    equal to t.'''
    return np.where(tt >= t)[0][0]


def plot_controller_result(plot_start_t: float, plot_end_t: float,
                           parameter: str, time: dict, dbs: dict,
                           controller_t: np.ndarray, controller_p: np.ndarray,
                           controller_b: np.ndarray,
                           lfp_time, lfp, axs: list[plt.Axes] = None):
    if axs is None:
        fig, axs = plt.subplots(4, 1, figsize=(15, 8))
    s = time_to_sample(time['signal'], plot_start_t)
    e = time_to_sample(time['signal'], plot_end_t) - 1
    axs[0].plot(time['signal'][s:e], dbs['signal'][s: e])
    axs[0].set_xlabel('Time [%s]' % time['signal_units'][0])
    axs[0].set_ylabel('DBS signal [%s]' % dbs['signal_units'][0])

    controller_t = controller_t * 1000
    s = time_to_sample(controller_t, plot_start_t)
    e = time_to_sample(controller_t, plot_end_t) - 1
    axs[1].plot(controller_t[s:e], controller_p[s: e])
    axs[1].set_ylabel(parameter)
    axs[1].set_xlabel('Time [ms]')

    axs[2].plot(controller_t[s:e], controller_b[s: e])
    axs[2].set_ylabel('Beta')
    axs[2].set_xlabel('Time [ms]')

    s = time_to_sample(lfp_time, plot_start_t)
    e = time_to_sample(lfp_time, plot_end_t) - 1
    axs[3].plot(lfp_time[s:e], lfp['signal'][s: e])
    axs[3].set_ylabel('Local field potential [%s]' % lfp['signal_units'][0])
    axs[3].set_xlabel('Time [ms]')
    plt.subplots_adjust(hspace=0.42)
    return axs


def load_and_plot(dirname: Union[str, list[str]], parameter: str,
                  steady_state_time: float, sim_time: float,
                  plot_start_t: float, plot_end_t: float) -> None:
    '''Loads and plots the simulation result from a given directory.

    If dirname is a list of strings, displays all the results in one plot.
    '''
    if isinstance(dirname, str):
        dirname = [dirname]
    axs = None
    for d in dirname:
        directory = Path(d)
        time, dbs = load_dbs_output(directory)
        lfp_time, lfp = load_stn_lfp(directory, steady_state_time, sim_time)
        controller_t, controller_p, controller_b =\
            load_controller_data(directory, parameter)
        axs = plot_controller_result(plot_start_t, plot_end_t, parameter, time,
                                     dbs, controller_t, controller_p,
                                     controller_b, lfp_time, lfp, axs)
