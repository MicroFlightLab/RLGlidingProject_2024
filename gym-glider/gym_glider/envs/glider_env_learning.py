import time
import gym
import numpy as np
from Simulation import glider_simulation
from RL.hyperparameters.hyperparameters import hyper_parameters_dict as hyper_parameters_dict_template


# environment for v2 of glider simulator - equations from paper
# for learning with hyper-parameters and sweeps
class GliderEnvLearning(gym.Env):
    def __init__(self, hyper_parameters_dict):
        self.hyper_parameters_dict = hyper_parameters_dict
        self.time_back = self.hyper_parameters_dict["time_back"]  # how much seconds each state is looking back
        # the reward/punishment for out movements
        self.out_of_bounds_punishment = 1000
        self.movement_error_punishment_per_sec = 1
        if "out_of_bounds_punishment" in self.hyper_parameters_dict.keys():  # backward compatibility
            self.out_of_bounds_punishment = self.hyper_parameters_dict["out_of_bounds_punishment"]
        if "movement_error_punishment_per_sec" in self.hyper_parameters_dict.keys():
            self.movement_error_punishment_per_sec = self.hyper_parameters_dict["movement_error_punishment_per_sec"]

        # create obs_to_param variable for explaining the environment
        self.action_to_param = dict()
        self.obs_to_param = dict()  # observation_number to param
        self.create_explaining_variables()  # create the conversion dictionaries for actions and states

        # create noises dict
        self.noises_dict = dict()  # {state: [noises for each step]}

        action_high = np.float32(
            np.array([1 for i in range(len(self.action_to_param))]))  # 1 is maximum because values are normalized
        self.action_space = gym.spaces.Box(-action_high, action_high)

        observation_high = np.float32(
            np.array([1 for i in
                      range(len(self.obs_to_param))]))  # 1 is maximum because values are normalized
        self.observation_space = gym.spaces.Box(-observation_high, observation_high)

        self.rl_glider_simulation = None
        self.reset()

    """
        helpers
    """

    def info_to_state(self, info):
        """
        convert current information to state
        :return: the proper state
        """
        # in the return of info var_time1 is before var_time0 (earlier times appears after the current time values)
        states = []
        states_dict = self.hyper_parameters_dict["states"]

        # need to iterate in the order of the obs_to_param dict
        if "value" in hyper_parameters_dict_template["states"].keys():
            # case of sweep - dictionaries are changed
            param_names_by_serial_number = sorted([param for param in states_dict.keys()],
                                                  key=lambda x: hyper_parameters_dict_template["states"]["value"][x][
                                                      "state_serial_number"])
        else:
            param_names_by_serial_number = sorted([param for param in states_dict.keys()],
                                                  key=lambda x: hyper_parameters_dict_template["states"][x][
                                                      "state_serial_number"])

        for param in param_names_by_serial_number:
            param_dict = states_dict[param]
            # if param in use add to state
            if param_dict["use"]:
                # add the normalized params
                param_range = (param_dict["max"] - param_dict["min"]) / 2
                param_mean = (param_dict["max"] + param_dict["min"]) / 2
                states += [(val - param_mean) / param_range for val in param_dict["function"](info)]

        return np.array(states)

    def get_info(self):
        """
        get all of the information from the simulation
        :return:
        """
        # in the return of info var_timeback0 is before var_timeback1 (earlier times appears after the current time values)
        # note: there is legacy info - thermal center - this info os the center timeback seconds ago and not as expected
        # this was fixed in fixed_thermal_center
        info = self.rl_glider_simulation.get_curr_state_for_learning(self.time_back)
        return info

    def state_to_real_values(self, state):
        """
        convert the state to real values
        :param state:
        :return:
        """
        states_dict = self.hyper_parameters_dict["states"]
        real_values = []
        for k in range(len(self.obs_to_param.keys())):
            param = self.obs_to_param[k]
            param_name = "_".join(param.split("_")[:-1])
            param_dict = states_dict[param_name]
            param_range = (param_dict["max"] - param_dict["min"]) / 2
            param_mean = (param_dict["max"] + param_dict["min"]) / 2
            real_values.append(param_mean + param_range * state[k])
        return real_values

    def real_values_to_state(self, real_values):
        """
        convert the real values to the state, by the range of variables
        :param real_values:
        :return:
        """
        states_dict = self.hyper_parameters_dict["states"]
        state = []
        for k in range(len(self.obs_to_param.keys())):
            param = self.obs_to_param[k]
            param_name = "_".join(param.split("_")[:-1])
            param_dict = states_dict[param_name]
            param_range = (param_dict["max"] - param_dict["min"]) / 2
            param_mean = (param_dict["max"] + param_dict["min"]) / 2
            state.append((real_values[k] - param_mean) / param_range)
        return np.array(state)

    def action_to_info(self, action):
        """
        convert the action to real values
        not considering the max value of each action
        :param action:
        :return:
        """
        real_actions = []
        actions_dict = self.hyper_parameters_dict["actions"]
        for i, param in enumerate(action):
            param_name = self.action_to_param[i]
            action_param = actions_dict[param_name]["max_action"] * param
            real_actions.append(action_param)
        return real_actions

    def set_params(self, action):
        """
        set the angles in the simulator from the action, setthe wingspan from the action
        need to add code for each action
        :param action:
        :return:
        """
        actions_dict = self.hyper_parameters_dict["actions"]

        # case of set param directly via action
        if ("set_actions_directly" in self.hyper_parameters_dict.keys()) and self.hyper_parameters_dict[
            "set_actions_directly"]:
            for i, param in enumerate(action):
                param_name = self.action_to_param[i]
                max_value = actions_dict[param_name]["max_value"]
                if "min_value" not in actions_dict[param_name].keys():
                    new_value = max_value * param
                else:
                    min_value = actions_dict[param_name]["min_value"]
                    # new value = average + (range/2) * param
                    new_value = ((max_value + min_value) / 2) + (param * (max_value - min_value) / 2)
                self.rl_glider_simulation.set_param_by_name(param_name, new_value)
        else:
            # case of set the param via action and the last param
            for i, param in enumerate(action):
                param_name = self.action_to_param[i]

                # get the last param
                curr_value = self.rl_glider_simulation.get_param_by_name(param_name)

                # set the new param
                new_value = curr_value + actions_dict[param_name]["max_action"] * param
                if new_value > actions_dict[param_name]["max_value"]:
                    new_value = actions_dict[param_name]["max_value"]
                # case of angles that doesnt have min value in hyper-parameters dict
                if "min_value" in actions_dict[param_name].keys():
                    if new_value < actions_dict[param_name]["min_value"]:
                        new_value = actions_dict[param_name]["min_value"]
                else:
                    if new_value < -actions_dict[param_name]["max_value"]:
                        new_value = -actions_dict[param_name]["max_value"]

                # set the value
                self.rl_glider_simulation.set_param_by_name(param_name, new_value)

    def create_explaining_variables(self):
        """
        create variables which will help understand the environment
        :return:
        """
        # states
        params_names = []
        states_dict = self.hyper_parameters_dict["states"]
        for param in states_dict.keys():
            param_dict = states_dict[param]
            # if param in use add to state
            if param_dict["use"]:
                params_names.append(param)

        # for preserving the order of the params in different dictionaries
        if "value" in hyper_parameters_dict_template["states"].keys():
            # case of sweep - dictionaries are changed
            # important note - might be a problem of sweeping over actions/states
            params_names = sorted(params_names,
                                  key=lambda x: hyper_parameters_dict_template["states"]["value"][x][
                                      "state_serial_number"])
        else:
            params_names = sorted(params_names,
                                  key=lambda x: hyper_parameters_dict_template["states"][x]["state_serial_number"])

        counter = 0
        for param in params_names:
            for i in range(self.time_back):
                self.obs_to_param[counter] = f"{param}_timeback{i}"
                counter += 1

        # actions
        actions_names = []
        actions_dict = self.hyper_parameters_dict["actions"]
        for param in actions_dict.keys():
            param_dict = actions_dict[param]
            # if param in use add to state
            if param_dict["use"]:
                actions_names.append(param)

        if "value" in hyper_parameters_dict_template["actions"].keys():
            # case of sweep - dictionaries are changed
            # important note - might be a problem of sweeping over actions/states
            actions_names = sorted(actions_names,
                                   key=lambda x: hyper_parameters_dict_template["actions"]["value"][x][
                                       "action_serial_number"])
        else:
            actions_names = sorted(actions_names,
                                   key=lambda x: hyper_parameters_dict_template["actions"][x]["action_serial_number"])

        for i, param in enumerate(actions_names):
            self.action_to_param[i] = param

    def create_noises_dict(self):
        """
        creating the dict for {state: [noise for each time]} - for every run utilize the different noise
        :return:
        """
        states_names = list(set("_".join(obs_name.split("_")[:-1]) for obs_name in self.obs_to_param.values()))
        # Note: there is problem with time 0 - not sure why - legacy code - that's why don't need to add 1
        total_steps = int(
            np.round(self.rl_glider_simulation.num_of_seconds / self.rl_glider_simulation.step_size_seconds))
        for state_name in states_names:
            if "noise" not in self.hyper_parameters_dict["states"][state_name].keys():
                # backward compatibility
                self.noises_dict[state_name] = np.zeros(total_steps)
            else:
                self.noises_dict[state_name] = np.random.normal(0,
                                                                self.hyper_parameters_dict["states"][state_name][
                                                                    "noise"], total_steps)

    def add_noise_to_state(self, state):
        """
        add noise to the state from the hyper-parameters
        :return:
        """
        state_real_values = self.state_to_real_values(state)
        curr_time = self.rl_glider_simulation.curr_time
        step_size = self.rl_glider_simulation.step_size_seconds

        states_dict = self.hyper_parameters_dict["states"]
        for k in range(len(self.obs_to_param.keys())):
            param = self.obs_to_param[k]
            param_name = "_".join(param.split("_")[:-1])
            param_timeback = int(param.split("_")[-1].split("timeback")[-1])
            param_dict = states_dict[param_name]
            if "noise" in param_dict.keys():
                index_for_noise = int(
                    np.round(curr_time / step_size)) - param_timeback - 1  # -1 because starts from step 1
                state_real_values[k] += self.noises_dict[param_name][index_for_noise]

        return self.real_values_to_state(state_real_values)

    def change_noise_in_env(self, states_noise_dict=dict(), zero=False):
        """
        change the noise in the environment
        :param zero: True iff zero all the noise that not in the states_noise_dict
        :param states_noise_dict: dictionary of the noise for each state that want to change {state_name:noise}
        :return:
        """
        states = self.hyper_parameters_dict["states"]
        for state in states:
            if state in states_noise_dict.keys():
                states[state]["noise"] = states_noise_dict[state]
            elif zero:
                states[state]["noise"] = 0

    """
        environment functions
    """

    def step(self, action):
        # doing the action
        self.set_params(action)

        # running the simulation
        ret_dict = self.rl_glider_simulation.step()
        done = ret_dict["out_of_bounds"] or ret_dict["out_of_time"] or ret_dict["movement_error"]

        # for saving the info
        info = self.get_info()

        # get the next state
        state = self.add_noise_to_state(self.info_to_state(info))  # add noise

        # get the reward
        if ret_dict["movement_error"]:
            # we want the glider to prefer going upwards and get movement error than only diving and not learning
            reward = -((self.rl_glider_simulation.num_of_seconds - self.rl_glider_simulation.curr_time) /
                       self.hyper_parameters_dict["step_size_seconds"]) * self.movement_error_punishment_per_sec
        elif ret_dict["out_of_bounds"]:
            reward = -self.out_of_bounds_punishment
        else:
            # backward compatibility
            if "chosen_reward" in self.hyper_parameters_dict.keys():
                chosen_reward = self.hyper_parameters_dict["chosen_reward"]
                reward = self.hyper_parameters_dict["rewards"][chosen_reward](info)
            else:
                reward = self.rl_glider_simulation.get_z_velocity()

        return state, reward, done, info

    def reset(self):
        # start position
        start_position = np.random.uniform(low=-50, high=50, size=(2,))
        side_angle = np.random.uniform(low=-100, high=100)

        # create initial params dict
        initial_params_dict = {"x": start_position[0], "y": start_position[1], "side_angle": side_angle}
        params = ["w_star", "thermal_height", "horizontal_wind", "velocity", "mode", "noise_wind", "time_for_noise_change",
                  "thermals_settings_list", "m", "wingspan", "aspect_ratio"]
        for param in params:
            if param in self.hyper_parameters_dict.keys():
                initial_params_dict[param] = self.hyper_parameters_dict[param]

        # option for changing wind if want to use - to make more robust agent
        if "horizontal_wind_settings" in self.hyper_parameters_dict.keys() and "horizontal_wind_functions" in self.hyper_parameters_dict.keys():
            if self.hyper_parameters_dict["horizontal_wind_settings"] is not None:
                func_name = self.hyper_parameters_dict["horizontal_wind_settings"]["func_name"]
                params = self.hyper_parameters_dict["horizontal_wind_settings"]["params"]
                horizontal_wind_func = self.hyper_parameters_dict["horizontal_wind_functions"][func_name]
                initial_params_dict["horizontal_wind"] = horizontal_wind_func(params)

        # create the simulation environment
        self.rl_glider_simulation = glider_simulation.GliderFlight(
            num_of_seconds=self.hyper_parameters_dict["flight_duration"],
            step_size_seconds=self.hyper_parameters_dict["step_size_seconds"],
            initial_params_dict=initial_params_dict)

        # create the noises' dict for the new run
        self.create_noises_dict()

        # reset the simulation
        for i in range(self.time_back):
            # Note: need self.time_back because every step creates another second so in the end there will be
            # self.timeback and time 0 which is problematic
            ret_dict = self.rl_glider_simulation.step()
            done = ret_dict["out_of_bounds"] or ret_dict["out_of_time"] or ret_dict["movement_error"]
            if done:
                print(f"error in reset, trying again {ret_dict}")
                self.reset()

        state = self.add_noise_to_state(self.info_to_state(self.get_info()))  # add noise

        return state

    def render(self, mode="human", free_txt="", animation=False, return_fig=False, return_both=False):
        fig = self.rl_glider_simulation.draw_simulation(free_txt=free_txt, animation=animation, return_fig=return_fig,
                                                     return_both=return_both)
        if return_fig:
            return fig
