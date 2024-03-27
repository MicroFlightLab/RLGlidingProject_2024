import shutil
import numpy as np
import plotly.express as px
import wandb
import gym
from tqdm import tqdm
import os
import pandas as pd
from Data import data_utils
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import plotly.io as pio
from RL.hyperparameters.hyperparameters import *
import collections.abc

from utils import graphical_utils

pio.renderers.default = "browser"

alg_dict = {
    "A2C": A2C,
    "SAC": SAC,
    "DDPG": DDPG,
    "PPO": PPO
}


def train_env(alg_name, params=dict(policy="MlpPolicy")):
    """
    function for training the glider model in environment

    In each training need to save:
    learning algorithm, learning parameters, environment parameters(wind, thermal_size), states/action (bounds for each)


    :param alg_name:
    :param params:
    :return:
    """
    algorithm = alg_dict[alg_name]
    params_str = "".join(f"{key}_{params[key]}" for key in params)
    models_dir = f"models/{alg_name}_{params_str}"
    logdir = "logs"
    latest_version_num = -1  # if not have version before

    env = gym.make('gym_glider:glider-v5', hyper_parameters_dict=hyper_parameters_dict)
    env.reset()

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        model = algorithm(env=env, verbose=1, tensorboard_log=logdir, **params)
    else:
        # get the latest model version
        latest_version_num = max([int(f.split(".")[0]) for f in os.listdir(models_dir)])
        model_path = os.path.join(models_dir, f"{latest_version_num}.zip")
        model = algorithm.load(model_path, env=env, verbose=1)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    TIMESTEPS = 10000
    iters = 33
    for i in range(latest_version_num // TIMESTEPS + 1, iters):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=alg_name)
        model.save(f"{models_dir}/{TIMESTEPS * i}")


def get_model(alg_name, dir_name, name, hyper_param_dict=None):
    # for customize hyper_parameters dict
    if hyper_param_dict is None:
        hyper_param_dict = hyper_parameters_dict

    models_dir = f"models/{dir_name}"
    env = gym.make('gym_glider:glider-v5', hyper_parameters_dict=hyper_param_dict)
    env.reset()

    model_path = os.path.join(models_dir, f"{name}.zip")
    model = alg_dict[alg_name].load(model_path, env=env)
    return model


def draw_env(alg_name, dir_name, name, animation=False):
    models_dir = f"models/{dir_name}"
    env = gym.make('gym_glider:glider-v5', hyper_parameters_dict=hyper_parameters_dict)
    print(env.observation_space.shape)
    env.reset()

    model_path = f"{models_dir}/{name}.zip"
    print(model_path)
    model = alg_dict[alg_name].load(model_path, env=env)

    episodes = 1

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
        print("finished route now ploting")
        env.render(animation=animation)


def get_df_for_analysis_by_model_env(model, env, log=True, episodes=15):
    """
    function for analyzing the environment that returns the df of the info
    running the model for episodes times and get the results

    there is a function called get_cols_by_category_of_analysis_df which returns explanation for each column -
    good for analysis and take original states

    :param log: whether to log the process
    :param model:
    :param env:
    :param episodes:
    :return:
    """
    env.reset()
    env_run_info_list = []  # contain the info in list of lists that will be converted to DB
    noises_dicts = []  # contain the noise values for each episode
    timeback = env.time_back

    # add the info from the environment - so need the keys
    info_keys = list(env.get_info().keys())  # the keys for

    for ep in tqdm(range(episodes), disable=not log):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)

            # adding the observation and action for the statistics
            env_run_info = obs.tolist()
            # add the tensor observation
            env_run_info += [model.policy.obs_to_tensor(obs)[0][0]]
            # add the real values of each state (obs)
            env_run_info += env.state_to_real_values(obs)
            env_run_info += action.tolist()
            # add real value of action
            env_run_info += env.action_to_info(action)

            # add all info parameters from environment to the df
            params_info = env.get_info()
            params_values = []
            for key in info_keys:
                for param_value in params_info[key]:
                    params_values.append(param_value)
            env_run_info += params_values

            # add the number of the episode
            env_run_info.append(ep)

            env_run_info_list.append(env_run_info)

            obs, rewards, done, info = env.step(action)

        noises_dicts.append(env.noises_dict.copy())

    df = pd.DataFrame(env_run_info_list)

    obs_to_param = env.obs_to_param.copy()  # dictionary that converts index to explanation of observation
    # add column of obs_tensor
    obs_to_param[len(obs_to_param)] = "obs_tensor"
    # add the real values of each state (obs)
    for k in range(len(env.obs_to_param.keys())):
        obs_to_param[len(obs_to_param)] = f"{env.obs_to_param[k]}_real"

    for k in range(len(env.action_to_param.keys())):
        obs_to_param[len(obs_to_param)] = f"{env.action_to_param[k]}_action"

    # add real value of action
    for k in range(len(env.action_to_param.keys())):
        obs_to_param[len(obs_to_param)] = f"{env.action_to_param[k]}_action_real"

    # add the names from the information of get_info() function
    params_info = env.get_info()
    total_timeback = len(params_info[info_keys[0]])
    for key in info_keys:
        for timeback in range(total_timeback):
            obs_to_param[len(obs_to_param)] = f"info_{key}_timeback{timeback}"

    # add the episode/route number
    obs_to_param[len(obs_to_param)] = "route_num"

    df.rename(columns=obs_to_param, inplace=True)

    df = data_utils.add_distance_from_center_for_glider(df)
    df = data_utils.add_thermal_classification_for_glider(df, old_version=False)

    # add noise to df
    noise_cols_per_state = dict()
    for i, noise_dict in enumerate(noises_dicts):
        for state in noise_dict.keys():
            if state not in noise_cols_per_state.keys():
                noise_cols_per_state[state] = []
            noise_lst_to_add = noise_dict[state][timeback:len(df[df["route_num"] == i]) + timeback]
            noise_cols_per_state[state] += list(noise_lst_to_add)

    for noise_col_name in noise_cols_per_state.keys():
        df[f"{noise_col_name}_noise"] = noise_cols_per_state[noise_col_name]

    return df


def get_cols_by_category_of_analysis_df(run_data_df, filter_by_timeback=[]):
    """
    categorize the run data df from get_df_for_analysis_by_model_env
    :param filter_by_timeback:
    :param run_data_df:
    :return:
    """
    state_cols = []
    observations = []
    action_cols = []

    for col in run_data_df.columns:
        col_list = col.split("_")
        if col_list[-1][:-1] == "timeback" and col_list[0] != "info":
            observations.append("_".join(col_list[:-1]))

    for col in run_data_df.columns:
        col_list = col.split("_")
        if col_list[-1] == "real" and (filter_by_timeback == [] or col_list[-2][-1] in filter_by_timeback) and \
                col_list[-2] != "action":
            state_cols.append(col)

        if col_list[-1] == "real" and col_list[-2] == "action":
            action_cols.append(col)

    categorized_cols = {
        "state_cols": state_cols,
        "action_cols": action_cols,
        "observations": observations
    }

    return categorized_cols


def get_df_for_analysis(alg_name, dir_name, name, episodes=40, custom_hyper_dict=None):
    """
    function for analyzing the environment that returns the df
    :param custom_hyper_dict: for the case of wanted to change the hyper parameters
    :param episodes: number of episodes to run for getting the data
    :param alg_name:
    :param dir_name:
    :param name: name of the model
    :return:
    """
    # run a lot of simulations
    # graph of action as function of parameter
    if custom_hyper_dict is None:
        print("using default hyper parameters")
        chosen_hyper_param_dict = hyper_parameters_dict
    else:
        print("using custom hyper parameters")
        chosen_hyper_param_dict = custom_hyper_dict

    env = gym.make('gym_glider:glider-v5', hyper_parameters_dict=chosen_hyper_param_dict)

    model_path = fr"{dir_name}/{name}.zip"
    model = alg_dict[alg_name].load(model_path, env=env)

    df = get_df_for_analysis_by_model_env(model, env, episodes=episodes)
    return df


def get_model_path_from_wandb_id(wandb_path, wandb_file_name):
    """
    get the path of the model from the wandb path
    :param wandb_file_name:
    :param wandb_path:
    :return:
    """
    f = wandb.restore(wandb_file_name, run_path=wandb_path)
    f.close()
    return f.name


def make_env(config):
    env = gym.make(config["env_name"], hyper_parameters_dict=config)
    env = Monitor(env)
    return env


def average_vz_by_direction_in_thermals(params):
    """
    calculate the average vz by the angle from the wind
    :param params:
    :return:
    """
    info_df = params["info_df"]
    thermals_info_df = info_df[info_df["is_thermal"] == 1]
    if "info_vz_timeback0" in thermals_info_df.columns:
        thermals_info_df["vz"] = thermals_info_df["info_vz_timeback0"]
    else:
        thermals_info_df['vz'] = thermals_info_df.apply(
            lambda row: row.info_velocity_timeback0 * np.sin(np.deg2rad(row.info_glide_angle_timeback0)),
            axis=1)
    hyper_param_dict = params["hyper_params_dict"]
    return average_param_by_direction(thermals_info_df, hyper_param_dict, "vz")


def average_vz_by_wind_size(params):
    """
    creates graph of the average vz by the wind size
    :param params:
    :return:
    """
    info_df = params["info_df"]
    hyper_param_dict = params["hyper_params_dict"]
    vz_by_wind = info_df.groupby("info_wind_speed_timeback0")["info_vz_timeback0"].mean()
    count_by_wind = info_df.groupby("info_wind_speed_timeback0")["info_vz_timeback0"].count()
    var_by_wind = pd.DataFrame({"vz": vz_by_wind, "len": count_by_wind, "wind": vz_by_wind.index})
    var_by_wind["len"] = var_by_wind["len"]/200
    num_samples_in_bin = 1 + len(vz_by_wind) // 7

    figs = {
        "vz": average_param_by_param(var_by_wind, "wind", "vz", num_samples_in_bin=num_samples_in_bin),
        "len/200": average_param_by_param(var_by_wind, "wind", "len", num_samples_in_bin=num_samples_in_bin)
    }

    fig = graphical_utils.combine_graphs(figs)

    return fig


def average_velocity_by_direction(params):
    """
    calculate the average velocity by the angle from the wind
    :param params:
    :return:
    """
    info_df = params["info_df"]
    hyper_param_dict = params["hyper_params_dict"]
    return average_param_by_direction(info_df, hyper_param_dict, "info_velocity_timeback0")


def median_distance_by_direction(params):
    """
    calculate the average distance by the angle from the wind
    :param params:
    :return:
    """
    info_df = params["info_df"]
    hyper_param_dict = params["hyper_params_dict"]
    return average_param_by_direction(info_df, hyper_param_dict, "real_distance", y_func_name="median")


def average_param_by_direction(info_df, hyper_param_dict, param, num_samples_in_bin=20, y_func_name="mean",
                               error_y_func_name="sem"):
    """
    generic function for parameter as function of angle from wind
    :param error_y_func_name:
    :param y_func_name: the function of y value - median/mean
    :param num_samples_in_bin: the average number of samples in each bin
    :param info_df:
    :param hyper_param_dict:
    :return:
    """
    if "info_wind_angle_timeback0" in info_df.columns:
        info_df["wind_angle"] = info_df["info_wind_angle_timeback0"]

    if "wind_angle" not in info_df.columns:
        outer_wind = hyper_param_dict["horizontal_wind"]
        wind_angle = np.rad2deg(np.arctan2(outer_wind[1], outer_wind[0]))
        info_df["wind_angle"] = wind_angle
    info_df["relative_angle"] = (2 * (info_df["info_bank_angle_timeback0"] >= 0) - 1) * (
            ((info_df["info_side_angle_timeback0"] - info_df["wind_angle"]) % 360) - 180)
    fig = average_param_by_param(info_df, "relative_angle", param, num_samples_in_bin=num_samples_in_bin,
                                 y_func_name=y_func_name, error_y_func_name=error_y_func_name)
    return fig


def average_param_by_param(info_df, x_param, y_param, num_samples_in_bin=20, y_func_name="mean",
                           error_y_func_name="sem"):
    """
    generic function for average parameter as function of another parameter with plotly
    :param error_y_func_name:
    :param y_func_name: the function of y value - median/mean
    :param info_df:
    :param x_param:
    :param y_param:
    :param num_samples_in_bin:
    :return:
    """
    # case of no data - might happen at the beginning of the run - runner cannot thermal
    if len(info_df) < num_samples_in_bin:
        return px.line()

    num_bins = min(72, int(len(info_df) / num_samples_in_bin))
    pd_bins = pd.cut(info_df[x_param], bins=num_bins)
    y_param_by_x_param = info_df.groupby(pd_bins)[y_param].agg(['count', 'mean', 'sem', 'median'])
    y_param_by_x_param[f"{x_param}_mid"] = [interval.mid for interval in y_param_by_x_param.index]
    fig = px.line(y_param_by_x_param, x=f"{x_param}_mid", y=y_func_name, error_y=error_y_func_name,
                  hover_data=["count"],
                  labels={y_func_name: y_param, "angle_mid": x_param})
    return fig


def average_bank_angle_in_thermals(params):
    """
    the mean of the absolute bank angles
    :param params:
    :return:
    """
    info_df = params["info_df"]
    thermals_info_df = info_df[info_df["is_thermal"] == 1]
    # case of no data - might happen at the beginning of the run - runner cannot thermal
    if len(thermals_info_df) == 0:
        return 0
    return np.mean(np.abs(thermals_info_df["info_bank_angle_timeback0"]))


def average_bank_angle(params):
    """
    the mean of the absolute bank angles
    :param params:
    :return:
    """
    info_df = params["info_df"]
    return np.mean(np.abs(info_df["info_bank_angle_timeback0"]))


def average_attack_angle(params):
    """
    the mean of the absolute attack angles
    :param params:
    :return:
    """
    info_df = params["info_df"]
    return np.mean(np.abs(info_df["info_attack_angle_timeback0"]))


def average_vz(params):
    """
    the mean of vz in thermal
    :param params:
    :return:
    """
    info_df = params["info_df"]
    return np.mean(info_df["info_vz_timeback0"])


def average_vz_in_thermals(params):
    """
    the mean of vz in thermal
    :param params:
    :return:
    """
    info_df = params["info_df"]
    thermals_params = dict()
    thermals_params["info_df"] = info_df[info_df["is_thermal"] == 1]
    # case of no data - might happen at the beginning of the run - runner cannot thermal
    if len(thermals_params["info_df"]) == 0:
        return 0
    return average_vz(thermals_params)


def std_bank_angle_in_thermals(params):
    """
    the variance of the absolute bank angles
    :param params:
    :return:
    """
    info_df = params["info_df"]
    thermals_info_df = info_df[info_df["is_thermal"] == 1]
    # case of no data - might happen at the beginning of the run - runner cannot thermal
    if len(thermals_info_df) == 0:
        return 0
    return np.std(np.abs(thermals_info_df["info_bank_angle_timeback0"]))


def median_distance_from_center(params):
    """
    the distance from the center of the thermal
    :param params:
    :return:
    """
    info_df = params["info_df"]
    return np.median(info_df["real_distance"])


def median_distance_from_center_in_thermals(params):
    """
    the distance from the center of the thermal in thermals
    :param params:
    :return:
    """
    info_df = params["info_df"]
    thermals_params = dict()
    thermals_params["info_df"] = info_df[info_df["is_thermal"] == 1]
    # case of no data - might happen at the beginning of the run - runner cannot thermal
    if len(thermals_params["info_df"]) == 0:
        return 0
    return median_distance_from_center(thermals_params)


def bank_by_velocity(params):
    """
    calculate graph of bank angle by velocity speed
    :param params:
    :return:
    """
    info_df = params["info_df"]
    info_df["abs_bank_angle"] = np.abs(info_df["info_bank_angle_timeback0"])
    pd_bins = pd.cut(info_df['info_velocity_timeback0'], bins=60)
    bank_by_velocity_df = info_df.groupby(pd_bins)['abs_bank_angle'].agg(['count', 'mean', 'std'])
    bank_by_velocity_df["velocity_mid"] = [interval.mid for interval in bank_by_velocity_df.index]
    fig = px.line(bank_by_velocity_df, x="velocity_mid", y="mean", error_y="std",
                  labels={"mean": "bank_angle", "velocity_mid": "Veclocity"}, hover_data=["count"])
    return fig


def direction_change(params):
    """
    calculate the average direction change in second
    that means the probability to change direction in each timestamp
    :param params:
    :return:
    """
    info_df = params["info_df"]
    hyper_params_dict = params["hyper_params_dict"]
    # there is -1 because the implementation of training_utils.get_df_for_analysis_by_model_env
    # the last point does not count
    flight_duration_data = hyper_params_dict["flight_duration"] - 1
    average_direction_change = len(
        [(i, val) for i, val in enumerate((info_df["info_bank_angle_timeback0"] >= 0).diff()) if
         val and (i % flight_duration_data) > 1]) / len(info_df)
    return average_direction_change


def average_thermal_in_and_out(params):
    """
    calculate the number of times the bird entered and exited the thermal
    :param params:
    :return:
    """
    info_df = params["info_df"]
    return info_df.groupby("route_num")["is_thermal"].apply(lambda x: (np.abs(x.diff()) == 1).sum()).mean()


def average_thermal_time(params):
    """
    calculate the average time spent in the thermal
    :param params:
    :return:
    """
    info_df = params["info_df"]
    return info_df.groupby("route_num")["is_thermal"].apply(lambda x: (x == 1).sum()).mean()


def param_in_polar_by_angle(info_df, angle_param="angle_from_wind_timeback0_real",
                            param_to_show="angle_from_wind_timeback0_real", num_bins=20,
                            color="kmeans_neurons_cluster", func_name="count", is_bar=False, title=None):
    """
    return polar plot of parameter by angle
    :param title:
    :param info_df:
    :param num_bins:
    :param angle_param:
    :param color:
    :param param_to_show:
    :param func_name: the function to check of the parameter in the angles
    :param is_bar:
    :return:
    """
    pd_bins = pd.cut(info_df[angle_param], bins=num_bins)

    # determine the angle column and the color if exist
    groupby_cols = [pd_bins]
    if color is not None:
        groupby_cols = [pd_bins, color]

    # divide to intervals
    func_by_intervals_and_color = info_df.groupby(groupby_cols)[param_to_show].agg([func_name])
    if color is not None:
        func_by_intervals_and_color["angles"] = [ind[0].mid for ind in func_by_intervals_and_color.index]
        func_by_intervals_and_color[color] = [ind[1] for ind in func_by_intervals_and_color.index]
    else:
        func_by_intervals_and_color["angles"] = [ind.mid for ind in func_by_intervals_and_color.index]

    # plot
    if is_bar:
        fig = px.bar_polar(func_by_intervals_and_color, theta="angles", r=func_name, color=color)
        max_val = max(func_by_intervals_and_color.groupby("angles")["count"].sum())
        fig = fig.update_polars(radialaxis_range=list([-max_val, 1.5 * max_val]))
    else:
        if color is not None:
            color_name = color
        else:
            color_name = "type"
            func_by_intervals_and_color[color_name] = "graph"
        # add zero line
        angles = np.linspace(-180, 180, 100)
        zero_line = pd.DataFrame({"angles": angles, color_name: ["zero line"] * len(angles),
                                  func_name: [0] * len(angles)}).sort_values(by=["angles"])
        func_by_intervals_and_color = pd.concat([func_by_intervals_and_color, zero_line])

        # plot line
        fig = px.line_polar(func_by_intervals_and_color, theta="angles", r=func_name, color=color_name)
    fig = fig.update_layout(polar_angularaxis=dict(tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                                   ticktext=['$0^\\circ$', '$45^\\circ$', '$90^\\circ$', '$135^\\circ$',
                                                             '$-180^\\circ$', '$-135^\\circ$', '$-90^\\circ$',
                                                             '$-45^\\circ$']))

    # update title
    title_text = param_to_show
    if title is not None:
        title_text = title
    fig = fig.update_layout(title_text=title_text)
    return fig


def delete_folder(folder):
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            print("finished and deleted files")
        except Exception as e:
            print(f'Failed to delete {folder}. Reason: {e}')
    else:
        print("path doesn't exist")


def update(d, u):
    """
    updtae nested dictionaries
    :param d:
    :param u:
    :param u:
    :return:
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping) and isinstance(d.get(k, {}), collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# fake model in order to test the df
class FakeModel:
    def __init__(self, env):
        self.env = env
        self.bank_angle = 0
        self.num_bank_angles = 5

    def predict(self, obs):
        print(obs)
        if self.bank_angle < self.num_bank_angles:
            self.bank_angle += 1
            return np.array([1]), None
        # return np.array([0]), None
        return self.env.action_space.sample(), None


if __name__ == "__main__":
    # train_env("A2C")
    # train_env("SAC")
    # train_env("DDPG")
    # train_env("PPO")
    # draw_env("DDPG", "DDPG_policy_MlpPolicy", 240000, animation=True)
    # df = get_df_for_analysis("DDPG", "DDPG_policy_MlpPolicy", 240000, 10)
    # draw_env("DDPG", "check", "model_DDPG_400000", animation=False)  # model_DDPG_400000
    # print(df.head())
    # m = get_model("DDPG", "DDPG_policy_MlpPolicy", 240000)
    # DDPG().policy.predict()
    # print(type(m.policy))

    dummy_test = True
    if dummy_test:
        params = hyper_parameters_dict
        # change the outer wind to be 0
        params["horizontal_wind"][0] = 0
        params["w_star"] = 0
        env = gym.make('gym_glider:glider-v5', hyper_parameters_dict=params)
        model = FakeModel(env)
        df = get_df_for_analysis_by_model_env(model, env, episodes=1)
        df.to_pickle("df_test_no_wind.pkl")

