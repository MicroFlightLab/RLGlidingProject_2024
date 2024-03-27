import gym
from tqdm import tqdm
import os
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import PPO
import wandb
import plotly.io as pio
from RL.hyperparameters.hyperparameters import *
from RL import training_utils

pio.renderers.default = "browser"

alg_dict = training_utils.alg_dict


def get_model_by_run_id(project_name, run_id, model_number=-1, model_full_path_in_wandb=None, api=None, log=False,
                        env_name='gym_glider:glider-v5', user_name="yoavflato", return_only_config=False):
    """
    get the model from wandb run id
    :param return_only_config:
    :param user_name:
    :param env_name:
    :param log: whether to log or not
    :param model_full_path_in_wandb: if you want to load the model from a full path
    :param project_name:
    :param run_id:
    :param model_number:
    :param api:
    :return:
    """
    if api is None:
        api = wandb.Api()

    # case there is no path for model file
    chosen_model = {}
    run = api.run(f"{user_name}/{project_name}/{run_id}")
    if return_only_config:
        return get_hyper_param_dict_from_config(run.config)

    if model_full_path_in_wandb is None:
        files = run.files()
        models_info = []
        for file in tqdm(files, disable=not log):
            file_name = file.name
            if file_name.endswith('zip'):
                name_parts = file_name.split(".")[0].split("/")[-1].split("_")
                models_info.append({"name": file_name, "name_parts": name_parts})

        models_info.sort(
            key=lambda x: int(x["name_parts"][-1]) if model_number == -1 else int(x["name_parts"][-1]) == model_number,
            reverse=True)
        chosen_model = models_info[0]
    else:
        chosen_model = {"name": model_full_path_in_wandb,
                        "name_parts": model_full_path_in_wandb.split(".")[0].split("/")[-1].split("_")}

    # restore the model
    alg_name = chosen_model["name_parts"][1]
    model_wandb_path = chosen_model["name"]
    if log:
        print(f"model_wandb_path: {model_wandb_path}")
    cwd = os.getcwd()
    model_path = os.path.join(cwd, chosen_model["name"].split("/")[-1])
    f = wandb.restore(model_wandb_path, run_path=f"{user_name}/{project_name}/{run_id}", root=cwd)
    f.close()

    # create the environment
    hyper_param_dict = get_hyper_param_dict_from_config(run.config)
    env = gym.make(env_name, hyper_parameters_dict=hyper_param_dict)
    env.reset()

    model = alg_dict[alg_name].load(model_path, env=env)

    summary_timestamp = int(model_wandb_path.split(".")[0].split("_")[-1])
    run_summary = get_summary_by_run_and_timestamp(run, summary_timestamp)

    os.remove(model_path)

    return {"model": model, "hyper_param_dict": hyper_param_dict, "env": env, "run_summary": run_summary}


def get_summary_by_run_and_timestamp(run, summary_timestamp, timestamps_key="global_step"):
    """
    get the summary of run in timestamp from the df in wandb
    :param run:
    :param summary_timestamp:
    :param timestamps_key: the key for our timestamp
    :return:
    """
    history = run.history().sort_values("_step")
    summary = dict()
    # go over all the history df to find the data in the relevant time
    for row in history.iterrows():
        row_dict = row[1].to_dict()
        if row_dict[timestamps_key] >= summary_timestamp:
            break
        for key in row_dict:
            if str(row_dict[key]) != "nan":
                summary[key] = row_dict[key]

    return summary


def get_df_and_model_dict_by_run_id(project_name, run_id, model_number=-1, model_full_path_in_wandb=None, api=None,
                                    episodes=5, log=False, update_hyperparameters=dict(), user_name="yoavflato"):
    """
    get the model from wandb run id and run it on the environment
    :param user_name:
    :param update_hyperparameters: the hyperparameters to update in dict
    :param log: whether to log or not
    :param episodes:
    :param project_name:
    :param run_id:
    :param model_number:
    :param model_full_path_in_wandb:
    :param api:
    :return:
    """
    model_dict = get_model_by_run_id(project_name=project_name, run_id=run_id, model_number=model_number,
                                     model_full_path_in_wandb=model_full_path_in_wandb, api=api, log=log,
                                     user_name=user_name)
    env = model_dict["env"]
    model = model_dict["model"]
    training_utils.update(env.hyper_parameters_dict, update_hyperparameters)
    df = training_utils.get_df_for_analysis_by_model_env(model, env, episodes=episodes, log=log)
    return {"df": df, "model_dict": model_dict}


def get_hyper_param_dict_from_config(config):
    """
    get the hyper parameters dict from the config dict
    :param config: from wandb
    :return: config dict with real functions
    """
    for state in config["states"].keys():
        if "value" in config["states"][state].keys() or "value" in hyper_parameters_dict["states"].keys():
            config["states"][state]["function"] = hyper_parameters_dict["states"]["value"][state]["function"]
        else:
            config["states"][state]["function"] = hyper_parameters_dict["states"][state]["function"]

    if "rewards" in config.keys():
        for reward in config["rewards"].keys():
            if "value" in hyper_parameters_dict["rewards"]:
                config["rewards"][reward] = hyper_parameters_dict["rewards"]["value"][reward]
            else:
                config["rewards"][reward] = hyper_parameters_dict["rewards"][reward]

    # back compatibility
    if "outer_wind_functions" in config.keys():
        config["horizontal_wind_functions"] = config["outer_wind_functions"]

    if "outer_wind_settings" in config.keys():
        config["horizontal_wind_settings"] = config["outer_wind_settings"]

    if "outer_wind" in config.keys():
        config["horizontal_wind"] = config["outer_wind"]

    # update wind functions
    if "horizontal_wind_functions" in config.keys():
        for wind_function in config["horizontal_wind_functions"].keys():
            if "value" in hyper_parameters_dict["horizontal_wind_functions"]:
                config["horizontal_wind_functions"][wind_function] = hyper_parameters_dict["horizontal_wind_functions"][
                    "value"][wind_function]
            else:
                config["horizontal_wind_functions"][wind_function] = hyper_parameters_dict["horizontal_wind_functions"][
                    wind_function]

    return config
