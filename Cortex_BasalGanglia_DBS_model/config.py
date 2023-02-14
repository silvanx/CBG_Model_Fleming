import yaml
from cerberus import Validator
from pathlib import Path

zero_schema = dict(
    SetPoint={"type": "float", "coerce": float},
    Ts={"type": "float", "coerce": float},
)

pid_schema = dict(
    SetPoint={"type": "float", "coerce": float},
    Kp={"type": "float", "coerce": float},
    Ti={"type": "float", "coerce": float},
    Td={"type": "float", "coerce": float},
    Ts={"type": "float", "coerce": float},
    MinValue={"type": "float", "coerce": float},
    MaxValue={"type": "float", "coerce": float},
)

ift_schema = dict(
    SetPoint={"type": "float", "coerce": float},
    Kp={"type": "float", "coerce": float},
    Ti={"type": "float", "coerce": float},
    Ts={"type": "float", "coerce": float},
    MinValue={"type": "float", "coerce": float},
    MaxValue={"type": "float", "coerce": float},
    stage_length={"type": "float", "coerce": float},
    gamma={"type": "float", "coerce": float},
    lam={"type": "float", "coerce": float},
    min_kp={"type": "float", "coerce": float},
    min_ti={"type": "float", "coerce": float},
)


class Config(object):
    schema = dict(
        Controller={
            "type": "string",
            "coerce": (str, lambda x: x.upper()),
            "default": "ZERO",
            "allowed": ("ZERO", "PID", "IFT"),
        },
        Modulation={
            "type": "string",
            "coerce": (str, lambda x: x.lower()),
            "default": "none",
            "allowed": ("amplitude", "frequency", "none", ""),
        },
        RandomSeed={"type": "integer", "coerce": int, "default": 3695},
        TimeStep={"type": "float", "coerce": float, "default": 0.01},
        SteadyStateDuration={"type": "float", "coerce": float, "default": 6000.0},
        RunTime={"type": "float", "coerce": float, "default": 32000.0},
        SetPoint={"type": "float", "coerce": float, "default": 0},
        beta_burst_modulation_scale={"type": "float", "coerce": float, "default": 0.02},
        Kp={"type": "float", "coerce": float, "default": 0.23},
        Ti={"type": "float", "coerce": float, "default": 0.2},
        Td={"type": "float", "coerce": float, "default": 0},
        Ts={"type": "float", "coerce": float, "default": 0},
        MinValue={"type": "float", "coerce": float, "default": 0},
        MaxValue={"type": "float", "coerce": float, "default": 3},
        stage_length={"type": "float", "coerce": float, "default": 0},
        gamma={"type": "float", "coerce": float, "default": 0.01},
        lam={"type": "float", "coerce": float, "default": 1e-8},
        min_kp={"type": "float", "coerce": float, "default": 0.01},
        min_ti={"type": "float", "coerce": float, "default": 0.01},
        save_stn_voltage={"type": "boolean", "coerce": bool, "default": True},
    )

    def __init__(self, config_file):

        if config_file:
            self.config_file = Path(config_file).resolve()
            with self.config_file.open("r") as f:
                conf = yaml.safe_load(f)
        else:
            self.config_file = None
            conf = {}

        v = Validator(require_all=False)

        if not v.validate(conf, self.schema):
            raise RuntimeError(f"Invalid configuration file:\n{v.errors}")

        for key, value in v.document.items():
            self.__setattr__(key, value)

    def __str__(self):
        return str(vars(self)).strip("{}").replace(", ", ",\n")


def get_controller_kwargs(config):
    v = Validator(require_all=True, purge_unknown=True)
    controller = config.Controller
    if controller == "ZERO":
        schema = zero_schema
    elif controller == "PID":
        schema = pid_schema
    elif controller == "IFT":
        schema = ift_schema
    else:
        raise RuntimeError("Invalid controller type")

    if not v.validate(vars(config), schema):
        raise RuntimeError(f"Invalid controller options:\n{v.errors}")

    return v.document
