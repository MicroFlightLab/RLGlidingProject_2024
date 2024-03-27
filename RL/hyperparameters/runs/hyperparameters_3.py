import numpy as np

hyper_parameters_dict = {
    # Glider settings #
    "m": 5,  # mass of the glider
    "wingspan": 2.5,
    "aspect_ratio": 16,

    # Environment Settings #
    "flight_duration": 200,  # max duration of flight in simulation
    "w_star": 5,  # the strength of wind
    "thermal_height": 2000,
    "step_size_seconds": 1,
    "horizontal_wind": [3, 0, 0],
    "noise_wind": [0, 0, 0],  # gaussian noise
    "time_for_noise_change": 5,  # if the area is turbulent the time for noise change is low
    "velocity": 10,  # initial velocity of the glider
    "mode": "moving",  # mode of the thermal, the incline of the thermal
    "thermals_settings_list": [],  # for the case of more than one thermal, [] - will be 1 default thermal
    # need to have the keys: "mode", "center", "w_star", "thermal_height", "decay_by_time_factor"
    # add option for adding different types of changing horizontal wind
    "horizontal_wind_functions": {
        "uniform": lambda min_max_wind: [np.random.uniform(min_max_wind[0], min_max_wind[1]), 0, 0],
        "discrete_uniform": lambda options: [np.random.choice(options), 0, 0]
    },
    "horizontal_wind_settings": None,  # if none will use the regular horizontal wind, else {"func_name": , "params": }

    # States Settings #
    "time_back": 8,  # the size of the history - 1 is the minimum
    # all possible parameters are from Simulation.RL_Glider_Simulator_v2 - GliderFlight.get_curr_state_for_learning
    "states": {
        "velocity": {
            "use": True,
            "function": lambda info: info["velocity"],
            "min": -20,
            "max": 20,
            "explanation": "the velocity of the glider",
            "state_serial_number": 0,
            "noise": 0
            # the serial number is to preserve the order of the states when creating environment
        },
        "wind_diff": {
            "use": False,
            "function": lambda info: info["wind_diffs"],  # getting the info with time_back
            "min": -1,
            "max": 1,
            "explanation": "the difference in vz between right and left wing",
            "state_serial_number": 1,
            "noise": 0
        },
        "distance": {
            "use": False,
            "function": lambda info: (info["x"] - info["thermal_center_0"][:, 0]) ** 2 + (
                    info["y"] - info["thermal_center_0"][:, 1]) ** 2,
            "min": -10000,
            "max": 10000,
            "explanation": "the squared distance of the glider - legacy state - showing only the distance from timeback seconds - bug",
            "state_serial_number": 3,
            "noise": 0
        },
        "vz": {
            "use": True,
            "function": lambda info: info["velocity"] * np.sin(np.deg2rad(info["glide_angle"])),
            "min": -30,
            "max": 30,
            "explanation": "velocity in z axis",
            "state_serial_number": 4,
            "noise": 0
        },
        "angle_from_wind": {
            "use": True,
            "function": lambda info: (2 * (info["bank_angle"] >= 0) - 1) * (
                    ((info["side_angle"] - info["wind_angle"]) % 360) - 180),
            "min": -180,
            "max": 180,
            "explanation": "angle from wind (-90 - leeside, 0 - headwind, 90 - windward, 180 - tailwind)",
            "state_serial_number": 5,
            "noise": 0
        },

        "glide_angle": {
            "use": False,
            "function": lambda info: (info["glide_angle"] + 180) % 360 - 180,
            "min": -180,
            "max": 180,
            "explanation": "the glide angle of the glider",
            "state_serial_number": 9,
            "noise": 0
        },
        "distance_from_center_v2": {
            "use": False,
            "function": lambda info: np.sqrt((info["x"] - info["fixed_thermal_center_0"][:, 0]) ** 2 + (
                    info["y"] - info["fixed_thermal_center_0"][:, 1]) ** 2),
            "min": 0,
            "max": 500,
            "explanation": "the distance from center",
            "state_serial_number": 10
        },
        "angle_from_wind_sin": {
            "use": False,
            "function": lambda info: np.sin(np.deg2rad(info["angle_from_wind"])),
            "min": -1,
            "max": 1,
            "explanation": "sin of angle from wind to prevent discontinuity",
            "state_serial_number": 11
        },
        "angle_from_wind_cos": {
            "use": False,
            "function": lambda info: np.cos(np.deg2rad(info["angle_from_wind"])),
            "min": -1,
            "max": 1,
            "explanation": "cos of angle from wind to prevent discontinuity",
            "state_serial_number": 12
        },
        "wind_speed": {
            "use": False,
            "function": lambda info: info["wind_speed"],
            "min": 0,
            "max": 12,
            "explanation": "the wind speed",
            "state_serial_number": 13
        },

        # states that are also control parameters
        "bank_angle": {
            "use": True,
            "function": lambda info: info["bank_angle"],
            "min": -50,
            "max": 50,
            "explanation": "the bank angle of the glider",
            "state_serial_number": 2,  # number is from legacy code - where bank angle was before other variables
            "noise": 0
        },
        "attack_angle": {
            "use": False,
            "function": lambda info: info["attack_angle"],
            "min": -30,
            "max": 30,
            "explanation": "the attack angle of the glider",
            "state_serial_number": 6,
            "noise": 0
        },
        "sideslip_angle": {
            "use": False,
            "function": lambda info: info["sideslip_angle"],
            "min": -50,
            "max": 50,
            "explanation": "the sideslip angle of the glider",
            "state_serial_number": 7,
            "noise": 0
        },
        "wingspan": {
            "use": False,
            "function": lambda info: info["wingspan"],
            "min": 0,
            "max": 3,
            "explanation": "the wingspan of the glider",
            "state_serial_number": 8,
            "noise": 0
        }

    },

    # Actions Settings #
    "actions": {
        "bank_angle": {
            "use": True,
            "max_action": 15,  # the maximum value of action
            "max_value": 50,  # the maximum value of action parameter 20
            "action_serial_number": 0
        },
        "attack_angle": {
            "use": True,
            "max_action": 10,
            "max_value": 30,  # 35, 10
            "action_serial_number": 1
        },
        "sideslip_angle": {
            "use": False,
            "max_action": 3,
            "max_value": 50,  # 35, 10
            "action_serial_number": 2
        },
        "wingspan": {
            "use": False,
            "max_action": 0.5,
            "max_value": 2.5,
            "min_value": 1.5,
            "action_serial_number": 3
        }
    },
    # if "set_action_directly" True, the value will be the control value of the action parameter
    # else the new control value will be the control value plus the action value
    "set_actions_directly": False,

    "rewards": {"vz": lambda info: info["vz"][0],
                "vz_and_center": lambda info: info["vz"][0] + 30 / (
                        10 + ((info["x"][0] - info["fixed_thermal_center_0"][0, 0]) ** 2 + (
                        info["y"][0] - info["fixed_thermal_center_0"][0, 1]) ** 2)),
                "center": lambda info: 50 / (
                        10 + ((info["x"][0] - info["fixed_thermal_center_0"][0, 0]) ** 2 + (
                        info["y"][0] - info["fixed_thermal_center_0"][0, 1]) ** 2)),
                "vz_center_punishment": lambda info: info["vz"][0] - np.sqrt(
                    (info["x"][0] - info["fixed_thermal_center_0"][0, 0]) ** 2 + (
                            info["y"][0] - info["fixed_thermal_center_0"][0, 1]) ** 2) / 50,
                "vz_attack": lambda info: info["vz"][0] - (info["attack_angle"][0] > 15) * (
                        info["attack_angle"][0] - 15) / 200
                },
    # the functions for the reward
    "chosen_reward": "vz",  # the chosen reward function
    "movement_error_punishment_per_sec": 1,  # the punishment for seconds missed because of movement error
    "out_of_bounds_punishment": 1000,  # the punishment for going out of bounds

    # wandb settings #
    "model_save_freq": 50000,
    "animation_save_freq": 20000,
    "video_save_each_n_animation_save": 3,  # every n*animation_save_freq the video will be saved - to save memory
    "gradient_save_freq": 10000,
    "param_calc_freq": 10000,
    "short_description": "trying_hyper_param_file",  # the description of run for name
    "entity": "yoavflato",  # wandb entity
    "project_name": "TuningForPaper",  # the name of the project in wandb
    "remove_history_from_computer": False,  # if True, the history of the run will be deleted from the computer
    "hyper_param_path": None,  # the path to the hyperparameters file
    "verbose": 2,

    # Learning Settings #
    "policy_type": "MlpPolicy",
    "total_timestamps": 2000000,
    "env_name": "gym_glider:glider-v5",
    "learning_rate": 0.001,
    "algorithm_name": "DDPG",  # the algorithm we use
    # "policy_kwargs": dict(net_arch=dict(pi=[64, 128, 128, 32], qf=[64, 128, 128, 32]))
    "policy_kwargs": dict(net_arch=dict(pi=[200, 200], qf=[200, 200])),
    "load_model_file_name": None,  # None if not loading, the file name in wandb such as "model_FFPG_6000000.zip"
    "load_model_path": None,  # None if not loading, the path in wandb such as "yoavflato/TuningForPaper/2q3q2q3"
    # when loading a model, the new parameters to change except from the parameters that has been loaded from wandb
    "load_model_new_params": None  # if want to resume run with some different parameter we can add
    # a dictionary with the new parameters
    # for example if want to resume a run and run it for longer time i can set this param for:
    # {"timestamp": 20000000} and the run will be for 20M more timestamps
}

# dictionary of different hyperparameters for runs, converts option number to hyperparameters
option_runs_hyper_parameters = {
    i: {"horizontal_wind": [i, 0, 0],
        "short_description": f"wind_{i}"} for i in range(4)}

sweep_config = {
    'method': 'bayes',  # grid, random, bayes
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'average_thermal_time'
    },
    'parameters': {
        'net_size': {'max': 500, 'min': 60},
        'num_layers': {'max': 4, 'min': 2},
        'algorithm_name': {'values': ["DDPG", "PPO"]},
        'learning_rate': {'max': 0.1, 'min': 0.0001},
        'total_timestamps': {'value': 1000000},
        'project_name': {'value': "TuneParameters"},
        "time_back": {'value': 8}
    }
}
