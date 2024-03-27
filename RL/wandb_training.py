import os
import shutil
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import PPO
import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from RL import training_utils
from RL import wandb_utils
from RL.wandbIntegration import WandbCallback
from RL.hyperparameters.hyperparameters import *
import numpy as np
import plotly.express as px
import pandas as pd

# for not making warnings
pd.options.mode.chained_assignment = None  # default='warn'

alg_dict = training_utils.alg_dict


def wandb_sweep(params_dict, sweep_config_dict, count_sweep=1, old_sweep_id=None):
    """
    running a sweep of the model
    :param old_sweep_id: in case want to start from a specific sweep
    :param count_sweep: number of times to run the agent in sweep
    :param params_dict: dictionary of relevant params for run
    :param sweep_config_dict: dictionary of sweep params
    :return:
    """
    entity = params_dict["entity"]
    # update the params dict so that every element is now a dictionary - in order to pass threw sweep
    for key, value in params_dict.items():
        params_dict[key] = {"value": value}

    params_dict.update(sweep_config_dict["parameters"])
    sweep_config_dict["parameters"] = params_dict
    project_name = params_dict["project_name"]["value"]
    if old_sweep_id is None:
        sweep_id = wandb.sweep(sweep_config_dict, project=project_name, entity=entity)
    else:
        sweep_id = old_sweep_id

    wandb.agent(sweep_id, function=wandb_train_env, project=project_name, entity=entity, count=count_sweep)


def wandb_train_env(config=None):
    """
    training the model and save it to wandb
    :param config: hyperparameters
    :return:
    """
    resume = False
    if config is None:
        # case of sweep
        run = wandb.init(config=config, sync_tensorboard=True, save_code=True)
        # If called by wandb.agent, as below, this config will be set by Sweep Controller
        config = wandb.config

        # update the config with lambda functions - because wandb cannot save functions
        config = wandb_utils.get_hyper_param_dict_from_config(config)
        # update the policy kwargs by net size and num layers
        if "net_size" in config:
            num_layers = 2
            if "num_layers" in config:
                num_layers = config["num_layers"]
            config["customized_policy_kwargs"] = dict(net_arch=[config["net_size"]] * num_layers)

    else:
        load_model_path = config["load_model_path"]
        load_model_file_name = config["load_model_file_name"]
        total_timestamps = config["total_timestamps"]
        hyper_param_path = config["hyper_param_path"]
        load_model_new_params = {}
        if "load_model_new_params" in config.keys():
            load_model_new_params = config["load_model_new_params"]
            if load_model_new_params is None:
                load_model_new_params = {}

        # in case want to load a project
        if load_model_path is not None:
            run_names = load_model_path.split("/")
            config = wandb_utils.get_model_by_run_id(run_names[1], run_names[2], return_only_config=True)
            config["project_name"] = run_names[1]
            config["load_model_path"] = load_model_path
            config["load_model_file_name"] = load_model_file_name
            config["short_description"] = f"{config['short_description']}_resumed"
            config["total_timestamps"] = total_timestamps  # for running in different timestamps than before
            config["hyper_param_path"] = hyper_param_path  # for saving the hyper_parameters
            # update parameters in the run
            config = training_utils.update(config, load_model_new_params)
            resume = True

        run = wandb.init(
            project=config["project_name"],
            entity=config["entity"],
            config=config,
            name=f"{config['algorithm_name']}_{config['short_description']}",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,  # optional
        )

    # save the hyperparameters.py file
    if config["hyper_param_path"] is not None:
        wandb.save(config["hyper_param_path"], base_path=os.path.dirname(config["hyper_param_path"]))
        if config["verbose"] > 1:
            print("saved hyperparameters.py file")

    def make_env():
        env = gym.make(config["env_name"], hyper_parameters_dict=config)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    plot_env = make_env()
    if resume:
        print(f"resume run {config['load_model_path']} with {config['load_model_file_name']}")
        model_path = training_utils.get_model_path_from_wandb_id(config["load_model_path"],
                                                                 config["load_model_file_name"])
        model = alg_dict[config["algorithm_name"]].load(model_path, env=env, verbose=1,
                                                        tensorboard_log=f"wandb/runs/{run.id}")
    else:
        # case of getting policy kwargs from config
        policy_kwargs = config["policy_kwargs"]
        if "customized_policy_kwargs" in config:
            policy_kwargs = config["customized_policy_kwargs"]

        # create model
        model = alg_dict[config["algorithm_name"]](config["policy_type"], env, verbose=1,
                                                   tensorboard_log=f"wandb/runs/{run.id}",
                                                   policy_kwargs=policy_kwargs,
                                                   learning_rate=config["learning_rate"])

    # backward compatibility section
    if "video_save_each_n_animation_save" not in config.keys():
        config["video_save_each_n_animation_save"] = hyper_parameters_dict["video_save_each_n_animation_save"]

    model.learn(
        total_timesteps=config["total_timestamps"],
        callback=WandbCallback(
            gradient_save_freq=config["gradient_save_freq"],
            model_save_freq=config["model_save_freq"],
            model_save_path=f"wandb/wandb_models/{run.id}",
            verbose=config["verbose"],
            params_functions=
            {
                "average_vz_by_direction_in_thermal": training_utils.average_vz_by_direction_in_thermals,
                "average_bank_angle_in_thermal": training_utils.average_bank_angle_in_thermals,
                "std_bank_angle_in_thermals": training_utils.std_bank_angle_in_thermals,
                "median_distance_from_center": training_utils.median_distance_from_center,
                "bank_by_velocity": training_utils.bank_by_velocity,
                "direction_change": training_utils.direction_change,
                "average_velocity_by_direction": training_utils.average_velocity_by_direction,
                "average_thermal_time": training_utils.average_thermal_time,
                "average_thermal_in_and_out": training_utils.average_thermal_in_and_out,
                "average_vz_in_thermals": training_utils.average_vz_in_thermals,
                "average_vz": training_utils.average_vz,
                "median_distance_from_center_in_thermals": training_utils.median_distance_from_center_in_thermals,
                "median_distance_by_direction": training_utils.median_distance_by_direction,
                "average_attack_angle": training_utils.average_attack_angle,
                "average_bank_angle": training_utils.average_bank_angle,
                "average_vz_by_wind_size": training_utils.average_vz_by_wind_size
            },
            animation_save_freq=config["animation_save_freq"],
            video_save_each_n_animation_save=config["video_save_each_n_animation_save"],
            plot_env=plot_env,
            hyper_params_dict=config,
            param_calc_freq=config["param_calc_freq"]
        ),
    )
    run.finish()

    # the case of loading a model, need to delete it
    if config["load_model_path"] is not None:
        os.remove(model_path)

    if "remove_history_from_computer" in config.keys() and config["remove_history_from_computer"]:
        training_utils.delete_folder("wandb")
