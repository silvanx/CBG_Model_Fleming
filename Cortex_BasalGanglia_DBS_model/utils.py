import numpy as np
import scipy.signal as signal
from pyNN.parameters import Sequence


def generate_poisson_spike_times(
    pop_size,
    start_time,
    duration,
    fr,
    timestep,
    random_seed
):
    """generate_population_spike_times generates (N = pop_size) Poisson
    distributed spiketrains with firing rate fr.

    Example inputs:
        pop_size = 10
        start_time = 0.0		# ms
        end_time = 6000.0		# ms
        timestep = 1  			# ms
        fr = 1					# Hz
    """

    # Convert to sec for calculating the spikes matrix
    dt = float(timestep) / 1000.0  # sec
    sim_time = float(((start_time + duration) - start_time) / 1000.0)  # sec
    n_bins = int(np.floor(sim_time / dt))

    spike_matrix = np.where(np.random.uniform(0, 1, (pop_size, n_bins)) < fr * dt)

    # Create time vector - ms
    t_vec = np.arange(start_time, start_time + duration, timestep)

    # Make array of spike times
    for neuron_index in np.arange(pop_size):
        neuron_spike_times = t_vec[
            spike_matrix[1][np.where(spike_matrix[0][:] == neuron_index)]
        ]
        if neuron_index == 0:
            spike_times = Sequence(neuron_spike_times)
        else:
            spike_times = np.vstack((spike_times, Sequence(neuron_spike_times)))

    return spike_times


def burst_txt_to_signal(
        tt: np.ndarray,
        aa: np.ndarray,
        tstart: float,
        tstop: float,
        dt: float
        ) -> tuple[np.ndarray, np.ndarray]:
    """Generates square wave from step amplitudes and times"""
    segments = []
    if tstart < tt[0]:
        initial_segment = aa[0] * np.ones(int(np.floor((tt[0] - tstart) / dt)))
        segments.append(initial_segment)
    for i in range(len(tt)):
        t = tt[i]
        a = aa[i]
        if t > tstop:
            break
        if i == (len(tt) - 1):
            next_t = tstop
        elif tt[i + 1] > tstop:
            next_t = tstop
        else:
            next_t = tt[i + 1]
        segment_length = int(np.floor((next_t - t) / dt))
        seg = a * np.ones(segment_length)
        segments.append(seg)
    signal = np.concatenate(segments)
    return np.linspace(tstart, tstop, len(signal)), signal


def generate_inhomogeneous_poisson_spike_times(
    pop_size,
    tt,
    fr_envelope,
    dt,
    random_seed,
    isi_dither,
):
    spike_times = []
    for neuron_index in np.arange(pop_size):
        neuron_spike_train = np.random.rand(len(fr_envelope)) < fr_envelope * dt / 1000
        neuron_spike_times = tt[np.nonzero(neuron_spike_train)[0]]
        neuron_spike_times += isi_dither * np.random.randn(len(neuron_spike_times))
        spike_times.append(Sequence(neuron_spike_times))
    return spike_times


def make_beta_cheby1_filter(fs, n, rp, low, high):
    """Calculate bandpass filter coefficients (1st Order Chebyshev Filter)"""
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq

    b, a = signal.cheby1(n, rp, [lowcut, highcut], "band")

    return b, a


def calculate_avg_beta_power(lfp_signal, tail_length, beta_b, beta_a):
    """Calculate the average power in the beta-band for the current LFP signal
    window, i.e. beta Average Rectified Value (ARV)

    Inputs:
        lfp_signal          - window of LFP signal (samples)

        tail_length         - tail length which will be discarded due to
                              filtering artifact (samples)

        beta_b, beta_a      - filter coefficients for filtering the beta-band
                              from the signal
    """

    lfp_beta_signal = signal.filtfilt(beta_b, beta_a, lfp_signal)
    lfp_beta_signal_rectified = np.absolute(lfp_beta_signal)
    avg_beta_power = np.mean(lfp_beta_signal_rectified[-2 * tail_length : -tail_length])

    return avg_beta_power
