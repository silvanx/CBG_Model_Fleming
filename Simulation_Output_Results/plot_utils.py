import scipy.io as sio
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from pathlib import Path
from typing import Union, Any
from numpy.typing import NDArray
from scipy.interpolate import griddata
from matplotlib import cm
import re


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
        -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    '''Reads controller data from CSV files'''
    with open(dir / 'controller_sample_times.csv', 'r') as f:
        controller_t = np.array([float(r[0]) for r in csv.reader(f)])
    if parameter is not None:
        with open(dir / ('controller_%s_values.csv' % parameter), 'r') as f:
            controller_p = np.array([float(r[0]) for r in csv.reader(f)])
    else:
        with open(dir / ('controller_values.csv'), 'r') as f:
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
    if controller_p is not None:
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


def compute_cost(lfp: np.ndarray, u: np.ndarray, lam: float,
                 setpoint: float = 1.0414E-4) -> float:
    n = len(lfp)
    cost = np.sum((lfp - setpoint) ** 2 + lam * u ** 2) / (2 * n)
    return cost


def compute_mse(lfp_time: np.ndarray, lfp: np.ndarray,
                setpoint: float = 1.0414E-4,
                tail_length: float = None) -> float:
    if tail_length is not None:
        dt = lfp_time[1] - lfp_time[0]
        num_samples = int(tail_length / dt)
        lfp_time = lfp_time[-num_samples:]
        lfp = lfp[-num_samples:]
    duration = lfp_time[-1] - lfp_time[0]
    error = lfp - setpoint
    mse = np.trapz(error ** 2, lfp_time) / duration
    return mse


def plot_mse_dir(dir: str, setpoint: float = 1.0414E-4) -> None:
    directory = Path(dir)
    for result_dir in directory.iterdir():
        if not result_dir.is_dir():
            continue
        try:
            controller_t, _, controller_b =\
                load_controller_data(result_dir, None)
        except FileNotFoundError:
            print('Not found: %s' % (result_dir.name))
            continue
        mse = compute_mse(controller_t, controller_b, setpoint)
        fig = plt.figure(figsize=(20, 10))
        plt.plot(controller_t, controller_b)
        plt.axhline(setpoint, linestyle='--', color='k')
        plt.ylim([0, 4E-4])
        plt.title('%s (MSE = %.2E)' % (result_dir.name, mse))
        fig.savefig(directory / ('%s.png' % result_dir.name),
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)


def compute_teed(lfp_time, lfp, pulse_width, f_stimulation, impedance):
    recording_length = lfp_time.max() - lfp_time.min()
    power = np.square(lfp) * pulse_width * f_stimulation * impedance
    teed = np.trapz(power, lfp_time) / recording_length
    return teed


def plot_colormap(fig, ax, x_orig, y_orig, x, y, z, xlabel='$K_p$',
                  ylabel='$T_i$', title='', show_xy=False, cmap=cm.RdBu_r):
    fontsize = 20
    plt.sca(ax)
    contours = plt.contourf(x, y, z, cmap=cmap)
    if show_xy:
        ax.scatter(x_orig, y_orig, c='k', s=5)
    cbar = fig.colorbar(contours)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylim([-0.01, y_orig.max() + 0.01])
    ax.set_xlim([-0.01, x_orig.max() + 0.01])
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.offsetText.set_fontsize(fontsize)
    cbar.ax.yaxis.OFFSETTEXTPAD = 15


def plot_fitness_pi_params(pi_fitness_dir, setpoint=1.0414E-4, three_d=False,
                           lam=1e-8, tail_length=6, recalculate=False,
                           cmap=cm.gist_rainbow):
    (
        x,
        y,
        xi,
        yi,
        mse,
        teed,
        mse_zi,
        teed_zi,
        cost_zi
     ) = load_fitness_data(pi_fitness_dir, lam, setpoint,
                           tail_length=tail_length, recalculate=recalculate)
    fitness_zi = compute_fitness(x, y, xi, yi, mse, teed, lam)
    comb_fig, comb_ax = plt.subplots(1, 3, figsize=(33, 7))
    plot_colormap(comb_fig, comb_ax[0], x, y, xi, yi, mse_zi,
                  title='Beta power', show_xy=False, cmap=cmap)
    comb_ax[0].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    comb_ax[0].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    comb_ax[0].text(-0.2, 2.11, '(a)', fontsize=24)
    plot_colormap(comb_fig, comb_ax[1], x, y, xi, yi, teed_zi,
                  title='Stimulation power', show_xy=False, cmap=cmap)
    comb_ax[1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    comb_ax[1].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    comb_ax[1].text(-0.2, 2.11, '(b)', fontsize=24)
    plot_colormap(comb_fig, comb_ax[2], x, y, xi, yi, fitness_zi,
                  title=f'Cost function $\\lambda=${lam}', show_xy=False,
                  cmap=cmap)
    comb_ax[2].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    comb_ax[2].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    comb_ax[2].text(-0.2, 2.11, '(c)', fontsize=24)
    plt.subplots_adjust(wspace=0.15)
    plt.savefig('three_plots.eps', bbox_inches='tight')

    if three_d:
        mse_fig, mse_ax = plt.subplots(figsize=(25, 12),
                                       subplot_kw={"projection": "3d"})
        surf = mse_ax.plot_surface(xi, yi, mse_zi, cmap=cmap,
                                   antialiased=True)
        mse_fig.colorbar(surf)
        mse_ax.scatter(x, 1.01 * y, mse, c='k', s=3)

        teed_fig, teed_ax = plt.subplots(figsize=(25, 12),
                                         subplot_kw={"projection": "3d"})
        surf = teed_ax.plot_surface(xi, yi, teed_zi, cmap=cmap,
                                    antialiased=True)
        teed_fig.colorbar(surf)
        teed_ax.scatter(x, 1.01 * y, teed, c='k', s=3)

        fit_fig, fit_ax = plt.subplots(figsize=(25, 12),
                                       subplot_kw={"projection": "3d"})
        surf = fit_ax.plot_surface(xi, yi, fitness_zi, cmap=cmap,
                                   antialiased=True)
        fit_fig.colorbar(surf)
        fit_ax.scatter(x, y, fitness_zi, c='k', s=3)

        mse_ax.view_init(azim=60, elev=45)
        teed_ax.view_init(azim=80, elev=45)
        fit_ax.view_init(azim=80, elev=45)
    else:
        mse_fig = plt.figure(figsize=(12, 7))
        mse_ax = plt.gca()
        plot_colormap(mse_fig, mse_ax, x, y, xi, yi, mse_zi,
                      title='Beta power', show_xy=True, cmap=cmap)

        teed_fig = plt.figure(figsize=(12, 7))
        teed_ax = plt.gca()
        plot_colormap(teed_fig, teed_ax, x, y, xi, yi, teed_zi,
                      title='Stimulation power', show_xy=True, cmap=cmap)

        fit_fig = plt.figure(figsize=(12, 7))
        fit_ax = plt.gca()
        plot_colormap(fit_fig, fit_ax, x, y, xi, yi, fitness_zi,
                      title='Fitness function', show_xy=True, cmap=cmap)

        def func_to_vectorize(x, y, dx, dy, scaling=5000):
            plt.arrow(x, y, dx*scaling, dy*scaling, fc="k", ec="k", head_width=0.01, head_length=0.01)
        vectorized_arrow_drawing = np.vectorize(func_to_vectorize)

        cost_fig = plt.figure(figsize=(22, 14))
        cost_ax = plt.gca()
        plot_colormap(cost_fig, cost_ax, x, y, xi, yi, cost_zi,
                      title='Cost function', show_xy=True, cmap=cmap)
        a, b = np.gradient(cost_zi)
        vectorized_arrow_drawing(xi[1::10, 1::10], yi[1::10, 1::10], a[1::10, 1::10], b[1::10, 1::10])


def load_fitness_data(pi_fitness_dir, lam, setpoint=1.0414E-4, tail_length=6,
                      recalculate=False):
    directory = Path(pi_fitness_dir)

    if not recalculate:
        try:
            output = np.load(directory / 'output.npy', allow_pickle=True)
            return output
        except OSError:
            print('No output.npy found, recalculating')
    else:
        print('Forcibly recalculating fitness')

    fitness_list = []
    for result_dir in directory.iterdir():
        res = dict()
        if not result_dir.is_dir():
            continue
        try:
            controller_t, controller_p, controller_b = \
                load_controller_data(result_dir)
        except FileNotFoundError:
            print('Not found: %s' % (result_dir.name))
            continue
        mse = compute_mse(controller_t, controller_b, setpoint, tail_length)
        teed = compute_mse(controller_t, controller_p, 0, tail_length)

        tail_samples = int(tail_length / (controller_t[1]-controller_t[0]))
        cost = compute_cost(controller_b[-tail_samples:], controller_p[-tail_samples:], lam)

        if re.match('^Kp=.*', result_dir.name):
            params_string = result_dir.name.split('-')[0].split(',')
            for p in params_string:
                p = p.strip()
                k, v = p.split('=')
                res[k] = float(v)
        else:
            simulation_id = result_dir.name.split('-')[-1]
            with open(Path(pi_fitness_dir) / f'pi_grid_{simulation_id}.sh', 'r') as f:
                for line in f:
                    if re.match('^mpirun', line):
                        kp, ti = re.search('pi_([\.0-9]+)_([\.0-9]+)\.yml', line).groups()
                        res['Kp'] = float(kp)
                        res['Ti'] = float(ti)
        res['mse'] = mse
        res['teed'] = teed
        res['cost'] = cost
        fitness_list.append(res)
    x = []
    y = []
    mse = []
    teed = []
    cost = []
    for e in fitness_list:
        x.append(e['Kp'])
        y.append(e['Ti'])
        mse.append(e['mse'])
        teed.append(e['teed'])
        cost.append(e['cost'])
    x = np.array(x)
    y = np.array(y)
    mse = np.array(mse)
    teed = np.array(teed)
    cost = np.array(cost)
    max_value = max(x.max(), y.max()) + 0.01
    xi = yi = np.arange(0, max_value, 0.01)
    xi, yi = np.meshgrid(xi, yi)
    method = 'linear'
    mse_zi = griddata((x, y), mse, (xi, yi), method=method)
    teed_zi = griddata((x, y), teed, (xi, yi), method=method)
    cost_zi = griddata((x, y), cost, (xi, yi), method=method)
    output = np.array([x, y, xi, yi, mse, teed, mse_zi, teed_zi, cost_zi], dtype=object)
    np.save(directory / 'output.npy', output, allow_pickle=True)
    return output


def compute_fitness(x, y, xi, yi, mse, teed, lam, method='linear'):
    fitness = mse + lam * teed
    fitness_zi = griddata((x, y), fitness, (xi, yi), method=method)
    return fitness_zi


def plot_pi_fitness_function(pi_fitness_dir, fig, ax, setpoint=1.0414E-4,
                             lam=1, cmap=cm.RdBu_r, plot_grid=True, cax=None, zlim_exponent_high=1, zlim_exponent_low=-9):
    (
        x,
        y,
        xi,
        yi,
        mse,
        teed,
        _,
        _,
        _
        ) = load_fitness_data(pi_fitness_dir, setpoint)
    fitness_zi = compute_fitness(x, y, xi, yi, mse, teed, lam, 'linear')
    contours = ax.pcolormesh(
        xi,
        yi,
        fitness_zi,
        cmap=cmap,
        norm=clrs.LogNorm(vmin=(10 ** zlim_exponent_low - 1e-15),
                          vmax=(10 ** zlim_exponent_high + 1e-15)))
    if plot_grid:
        ax.scatter(x, y, c='k', s=5)
    if cax is None:
        fig.colorbar(contours)
    else:
        plt.colorbar(contours, cax=cax)
    ax.set_xlim([-0.01, x.max() + 0.01])
    ax.set_ylim([-0.01, y.max() + 0.01])


def plot_ift_trajectory(pi_fitness_dir, parameters, lam=1, setpoint=1.0414E-4,
                        cmap=cm.RdBu_r):
    fit_fig = plt.figure(figsize=(12, 7))
    fit_ax = plt.gca()
    plot_pi_fitness_function(pi_fitness_dir, fit_fig, fit_ax, setpoint, lam,
                             cmap, plot_grid=True)
    add_arrows_to_plot(fit_ax, parameters)
    fit_ax.set_xlabel('Kp')
    fit_ax.set_ylabel('Ti')
    fit_ax.set_title('Fitness of the PI parameters')


def read_ift_results(dirname):
    result_dir = Path(dirname)
    with open(result_dir / 'controller_sample_times.csv', 'r') as f:
        tt = np.array([float(r) for r in f])

    with open(result_dir / 'controller_iteration_values.csv', 'r') as f:
        iteration_history = np.array([float(r) for r in f])

    with open(result_dir / 'controller_reference_values.csv', 'r') as f:
        reference_history = np.array([float(r) for r in f])

    with open(result_dir / 'controller_error_values.csv', 'r') as f:
        error_history = np.array([float(r) for r in f])

    with open(result_dir / 'controller_beta_values.csv', 'r') as f:
        beta_history = np.array([float(r) for r in f])

    with open(result_dir / 'controller_parameter_values.csv', 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        parameters = np.asarray([[float(r[0]), float(r[1])]
                                 for r in csvreader])

    with open(result_dir / 'controller_integral_term_values.csv', 'r') as f:
        integral_term_history = np.array([float(r) for r in f])

    # time, dbs = load_dbs_output(result_dir)

    shifted_reference_history = np.copy(reference_history)
    shifted_reference_history[iteration_history == 0] = 0
    edges = list(iter(np.where(np.diff(1 * (iteration_history == 1)))[0]))
    n = len(edges)
    for i, s in enumerate(edges[::2]):
        if 2 * i + 1 >= n:
            break
        e = edges[2 * i + 1]
        length = e - s
        if s < length:
            tmp = np.delete(shifted_reference_history, np.s_[:s])
            shifted_reference_history = np.insert(tmp, length + 1, np.zeros(s))
        else:
            tmp = np.delete(shifted_reference_history, np.s_[s-length:s])
            shifted_reference_history = np.insert(
                tmp, e - length + 1, np.zeros(length)
                )

    return (
        tt,
        iteration_history,
        reference_history,
        error_history,
        beta_history,
        parameters,
        integral_term_history,
        )


def add_arrows_to_plot(ax, parameters):
    abase = np.vstack((parameters[np.unique(np.where(np.diff(parameters[:, :], axis=0))[0])],
                       parameters[-1, :]))
    ad = np.diff(abase, axis=0)
    arrows = []
    for i in range(len(ad)):
        a = ax.arrow(abase[i, 0], abase[i, 1], ad[i, 0], ad[i, 1],
                     width=0.01, head_width=0.04)
        arrows.append(a)
    return arrows


def plot_two_trajectories(pi_fitness_dir, dir1, lam1, dir2, lam2,
                          setpoint=1.0414E-4, timestop1=None, timestop2=None):
    x, y, xi, yi, mse, teed, _, _ = load_fitness_data(pi_fitness_dir, setpoint)
    fig, ax = plt.subplots(2, 1, figsize=(10, 17))

    tt, _, _, _, _, params, _ = read_ift_results(dir1)
    fitness_zi = compute_fitness(x, y, xi, yi, mse, teed, lam1)
    plot_colormap(fig, ax[0], x, y, xi, yi, fitness_zi)
    ax[0].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax[0].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    if timestop1 is not None:
        dt = tt[1] - tt[0]
        stop_index = int(timestop1 / dt)
        params = params[:stop_index]
    print(f'Lambda: {lam1}, '
          f'Initial value: ({params[0, 0]:.2f}, {params[0, 1]:.2f}), '
          f'Final value: ({params[-1, 0]:.2f}, {params[-1, 1]:.2f})')
    add_arrows_to_plot(ax[0], params)
    ax[0].text(-0.4, 2.06, '(a)', fontsize=30)
    ax[0].set_title(f'$\\lambda={lam1}$', fontsize=20)

    tt, _, _, _, _, params, _ = read_ift_results(dir2)
    fitness_zi = compute_fitness(x, y, xi, yi, mse, teed, lam2)
    plot_colormap(fig, ax[1], x, y, xi, yi, fitness_zi)
    ax[1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax[1].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    if timestop2 is not None:
        dt = tt[1] - tt[0]
        stop_index = int(timestop2 / dt)
        params = params[:stop_index]
    add_arrows_to_plot(ax[1], params)
    print(f'Lambda: {lam2}, '
          f'Initial value: ({params[0, 0]:.2f}, {params[0, 1]:.2f}), '
          f'Final value: ({params[-1, 0]:.2f}, {params[-1, 1]:.2f})')
    ax[1].text(-0.4, 2.06, '(b)', fontsize=30)
    ax[1].set_title(f'$\\lambda={lam2}$', fontsize=20)
    plt.subplots_adjust(hspace=0.22)


def plot_ift_signals(dirname, axs=None):
    result_dir = Path(dirname)
    with open(result_dir / 'controller_sample_times.csv', 'r') as f:
        tt = np.array([float(r) for r in f])

    with open(result_dir / 'controller_iteration_values.csv', 'r') as f:
        iteration_history = np.array([float(r) for r in f])

    with open(result_dir / 'controller_reference_values.csv', 'r') as f:
        reference_history = np.array([float(r) for r in f])

    with open(result_dir / 'controller_error_values.csv', 'r') as f:
        error_history = np.array([float(r) for r in f])

    with open(result_dir / 'controller_beta_values.csv', 'r') as f:
        beta_history = np.array([float(r) for r in f])

    with open(result_dir / 'controller_parameter_values.csv', 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        parameters = np.asarray([
            [float(row[0]), float(row[1])]for row in csvreader
        ])

    with open(result_dir / 'controller_integral_term_values.csv', 'r') as f:
        integral_term_history = np.array([float(r) for r in f])

    time, dbs = load_dbs_output(result_dir)

    shifted_reference_history = np.copy(reference_history)
    shifted_reference_history[iteration_history == 0] = 0
    edges = list(iter(np.where(np.diff(1 * (iteration_history == 1)))[0]))
    n = len(edges)
    for i, s in enumerate(edges[::2]):
        if 2 * i + 1 >= n:
            break
        e = edges[2 * i + 1]
        length = e - s
        if s < length:
            tmp = np.delete(shifted_reference_history, np.s_[:s])
            shifted_reference_history = np.insert(
                tmp, length + 1, np.zeros(s))
        else:
            tmp = np.delete(shifted_reference_history, np.s_[s-length:s])
            shifted_reference_history = np.insert(
                tmp, e - length + 1, np.zeros(length))

    if axs is None:
        fig = plt.figure(figsize=(12, 10))
        axs = []
        axs.append(fig.add_subplot(3, 1, 1))
        axs.append(fig.add_subplot(3, 1, 2))
        axs.append(fig.add_subplot(3, 1, 3))
    axs[0].plot(tt, beta_history)
    axs[1].plot(time['signal'][:, 0]/1000, -dbs['signal'][:, 0])
    axs[1].plot(tt, error_history)
    axs[1].plot(tt, integral_term_history)
    axs[2].plot(tt, parameters[:, 0])
    axs[2].plot(tt, parameters[:, 1])
    axs[2].plot(tt, iteration_history)
    axs[2].legend(['kp', 'ti'])
    axs[0].plot(tt, shifted_reference_history)

    for a in axs:
        a.set_xlim([min(tt) - 0.1, max(tt) + 0.1])
