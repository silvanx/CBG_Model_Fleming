import scipy.io as sio
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Any
from numpy.typing import NDArray
from scipy.interpolate import griddata
from matplotlib import cm


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


def load_controller_data(dir: Path, parameter: str = None)\
        -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    '''Reads controller data from CSV files'''
    with open(dir / 'controller_sample_times.csv', 'r') as f:
        controller_t = np.array([float(r[0]) for r in csv.reader(f)])
    if parameter is not None:
        with open(dir / ('controller_%s_values.csv' % parameter), 'r') as f:
            controller_p = np.array([float(r[0]) for r in csv.reader(f)])
    else:
        controller_p = None
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


def compute_mse(lfp_time, lfp, setpoint=1.0414E-4):
    duration = lfp_time[-1] - lfp_time[0]
    error = lfp - setpoint
    mse = np.trapz(error ** 2, lfp_time) / duration
    return mse


def plot_mse_dir(dir, setpoint=1.0414E-4):
    directory = Path(dir)
    for result_dir in directory.iterdir():
        if not result_dir.is_dir():
            continue
        controller_t, _, controller_b =\
            load_controller_data(result_dir, None)
        mse = compute_mse(controller_t, controller_b, setpoint)
        fig = plt.figure(figsize=(20, 10))
        plt.plot(controller_t, controller_b)
        plt.axhline(setpoint, linestyle='--', color='k')
        plt.ylim([0, 4E-4])
        plt.title('%s (MSE = %.2E)' % (result_dir.name, mse))
        fig.savefig(directory / ('%s.png' % result_dir.name),
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)


def plot_mse_pi_params(dir, setpoint=1.0414E-4, three_d=False,
                       cmap=cm.gist_rainbow):
    directory = Path(dir)
    mse_list = []
    for result_dir in directory.iterdir():
        res = dict()
        if not result_dir.is_dir():
            continue
        try:
            controller_t, _, controller_b = load_controller_data(result_dir,
                                                                 None)
        except FileNotFoundError:
            print('Not found: %s' % (result_dir.name))
            continue
        mse = compute_mse(controller_t, controller_b, 1.0414E-4)
        params_string = result_dir.name.split('-')[0].split(',')
        for p in params_string:
            p = p.strip()
            k, v = p.split('=')
            res[k] = float(v)
        res['mse'] = mse
        mse_list.append(res)
    x = []
    y = []
    z = []
    for e in mse_list:
        x.append(e['Kp'])
        y.append(e['Ti'])
        z.append(e['mse'])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    max_value = max(x.max(), y.max()) + 0.01
    xi = yi = np.arange(0, max_value, 0.01)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    if three_d:
        fig, ax = plt.subplots(figsize=(12, 7),
                               subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(xi, yi, zi, cmap=cmap,
                               antialiased=True)
        fig.colorbar(surf)
        ax.scatter(x, y, z, c='k', s=3)
    else:
        fig = plt.figure(figsize=(12, 7))
        contours = plt.contourf(xi, yi, zi, cmap=cmap)
        plt.scatter(x, y, c='k', s=5)
        fig.colorbar(contours)
    plt.xlabel('Kp')
    plt.ylabel('Ti')
    plt.xlim([-0.01, x.max() + 0.01])
    plt.ylim([-0.01, y.max() + 0.01])
    plt.title('Mean Square Error of beta power when using PI controller')
