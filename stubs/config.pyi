from _typeshed import Incomplete

class Config:
    Controller: str
    Modulation: str
    RandomSeed: int
    TimeStep: float
    SteadyStateDuration: float
    RunTime: float
    setpoint: float
    ctx_dc_offset: float
    ctx_dc_offset_std: float
    kp: float
    ti: float
    td: float
    ts: float
    min_value: float
    max_value: float
    stage_length: float
    gamma: float
    lam: float
    min_kp: float
    min_ti: float
    save_stn_voltage: bool
    save_gpe_voltage: bool
    save_ctx_voltage: bool
    save_ctx_lfp: bool
    save_interneuron_voltage: bool
    save_gpi_voltage: bool
    save_thalamus_voltage: bool
    create_new_network: bool
    Pop_size: int
    controller_window_length: float
    controller_window_tail_length: float
    fix_kp: bool
    fix_ti: bool
    stimulation_amplitude: float
    stimulation_frequency: float
    cortical_beta_mechanism: str
    ctx_slow_modulation_amplitude: float
    ctx_slow_modulation_step_count: int
    beta_burst_modulation_scale: float
    ctx_beta_spike_frequency: float
    ctx_beta_synapse_strength: float
    ctx_beta_spike_isi_dither: float
    r_matrix: str
    stage_two_mean: bool
    debug: bool
    normalise_error: bool

    def __init__(self, config_file) -> None: ...

def get_controller_kwargs(config): ...
