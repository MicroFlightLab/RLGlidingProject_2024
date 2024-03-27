import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp
from Simulation import wind_simulation
import time
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from Data import data_utils
from utils import graphical_utils

pio.renderers.default = "browser"


class GliderFlight:
    """
    Glider Simulation class for the RL algorithms
    different from version 1 - using RK integration with alpha,beta control
    """

    def __init__(self, num_of_seconds=180, step_size_seconds=5, save_information=True, initial_params_dict={},
                 dt=0.01):
        """
        initialization of simulation params, constants and data storage
        :param num_of_seconds:
        :param step_size_seconds:
        :param save_information:
        :param initial_params_dict: initial parameters list, all angle parameters need to be in degrees
        """
        # physics constants
        self.m = 5  # 2.3  # in kg : most of the times between 0.2-5.5 (from NASA paper)
        self.wingspan = 2.5  # in meter : most of the times between 1.5-3.5 (from NASA paper)
        self.aspect_ratio = 16  # most of the times between 6-20 (from NASA paper)
        self.g = 9.81
        self.rho = 1.225
        if "m" in initial_params_dict:
            self.m = initial_params_dict["m"]
        if "wingspan" in initial_params_dict:
            self.wingspan = initial_params_dict["wingspan"]
        if "aspect_ratio" in initial_params_dict:
            self.aspect_ratio = initial_params_dict["aspect_ratio"]

        # aerodynamic constants
        # from: https://ntrs.nasa.gov/api/citations/20040031358/downloads/20040031358.pdf
        # mass; mass in [0.23kg; 5.44kg]
        # wingspan; wing span in [1.52m; 3.55m]
        # aspect_ratio; aspect ratio in [6; 20]
        # AR_V; aspect ratio of vertical tail
        # lt; fuselage moment arm length
        # V_H; horizontal tail volume ratio
        # V_V; vertical tail volume ratio
        # c_; mean aerodynamic cord
        # S; wing surface area
        # S_F; fuselage area
        # S_T; horizontal tail surface
        # S_V; vertical tail surface
        # e; Oswald efficiency number
        # Re; Reynolds number
        # a0; lift curve slope
        # alpha0; zero point
        # C_d_0, C_d_L; wing profile drag coefficients
        # C_L_min, C_L_alpha, C_C_beta; minimum lift
        # C_D_F; fuselage drag coefficient
        # C_D_T; tail drag coefficient
        # C_D_E; miscellaneous "extra drag" coefficient
        # C_D_0; constant part of the drag coefficient with wind
        self.AR_V = 0.5 * self.aspect_ratio
        self.lt = 0.28 * self.wingspan
        self.V_H = 0.4
        self.V_V = 0.02
        self.c_ = 1.03 * self.wingspan / self.aspect_ratio
        self.S = self.wingspan * self.wingspan / self.aspect_ratio
        # best function to fit to the NASA paper data (here we calculate in meter and the paper uses inch)
        self.S_F = 0.01553571429 * np.power(self.wingspan, 2) + 0.01950357142 * self.wingspan - 0.01030412685
        self.S_T = self.V_H * self.c_ * self.S / self.lt
        self.S_V = self.V_V * self.wingspan * self.S / self.lt
        self.e = 0.95
        self.Re = 150000
        self.a0 = 0.1 * (180 / np.pi)
        self.alpha0 = -2.5 * (np.pi / 180)
        self.C_d_0 = 0.01
        self.C_d_L = 0.05
        self.C_L_min = 0.4
        self.C_L_alpha = self.a0 / (1 + self.a0 / (np.pi * self.e * self.aspect_ratio))
        self.C_C_beta = (self.a0 / (1 + self.a0 / (np.pi * self.e * self.AR_V))) * (self.S_V / self.S)
        self.C_D_F = 0.008
        self.C_D_T = 0.01
        self.C_D_E = 0.002
        self.C_D_0 = (self.C_D_F * self.S_F / self.S) + (self.C_D_T * (
                self.S_T + self.S_V) / self.S) + self.C_D_E + self.C_d_0

        # boundaries
        self.x_bounds = [-5000, 5000]
        self.y_bounds = [-5000, 5000]
        self.z_bounds = [0, 1000]

        # thermal enviroment
        thermal_params_dict = dict(x_bounds=self.x_bounds, y_bounds=self.y_bounds, z_bounds=self.z_bounds,
                                   total_time=num_of_seconds)
        if "w_star" in initial_params_dict:
            thermal_params_dict["w_star"] = initial_params_dict["w_star"]
        if "thermal_height" in initial_params_dict:
            thermal_params_dict["thermal_height"] = initial_params_dict["thermal_height"]
        if "horizontal_wind" in initial_params_dict:
            thermal_params_dict["horizontal_wind"] = initial_params_dict["horizontal_wind"]
        if "noise_wind" in initial_params_dict:
            thermal_params_dict["noise_wind"] = initial_params_dict["noise_wind"]
        if "mode" in initial_params_dict:
            thermal_params_dict["mode"] = initial_params_dict["mode"]
        if "time_for_noise_change" in initial_params_dict:
            thermal_params_dict["time_for_noise_change"] = initial_params_dict["time_for_noise_change"]
        if "thermals_settings_list" in initial_params_dict:
            thermal_params_dict["thermals_settings_list"] = initial_params_dict["thermals_settings_list"]
        self.thermal = wind_simulation.ThermalArea(**thermal_params_dict)

        # control variables
        self.sideslip_angle = 0 if "sideslip_angle" not in initial_params_dict.keys() else \
            np.deg2rad(initial_params_dict["sideslip_angle"])
        self.attack_angle = np.deg2rad(10) if "attack_angle" not in initial_params_dict.keys() else \
            np.deg2rad(initial_params_dict["attack_angle"])
        self.bank_angle = 0 if "bank_angle" not in initial_params_dict.keys() else \
            np.deg2rad(initial_params_dict["bank_angle"])

        # variables for simulation
        self.z = 500 if "z" not in initial_params_dict.keys() else \
            initial_params_dict["z"]
        self.x = 0 if "x" not in initial_params_dict.keys() else \
            initial_params_dict["x"]
        self.y = 0 if "y" not in initial_params_dict.keys() else \
            initial_params_dict["y"]
        self.v = 5 if "velocity" not in initial_params_dict.keys() else \
            initial_params_dict["velocity"]
        self.glide_angle = np.deg2rad(5) if "glide_angle" not in initial_params_dict.keys() else \
            np.deg2rad(initial_params_dict["glide_angle"])
        self.side_angle = 0 if "side_angle" not in initial_params_dict.keys() else \
            np.deg2rad(initial_params_dict["side_angle"])

        # params for simulation
        self.dt = dt
        self.step_size_seconds = step_size_seconds
        self.curr_time = 0
        self.num_of_seconds = num_of_seconds

        # arrays of information
        # [x, y, z, velocity, glide_angle, side_angle, bank_angle, attack_angle, sideslip_angle]
        self.param_to_index = {
            "x": 0,
            "y": 1,
            "z": 2,
            "velocity": 3,
            "glide_angle": 4,
            "side_angle": 5,
            "bank_angle": 6,
            "attack_angle": 7,
            "sideslip_angle": 8,
            "l": 9,
            "d": 10,
            "c": 11,
            "velocity_w": 12,
            "glide_angle_w": 13,
            "side_angle_w": 14,
            "bank_angle_w": 15,
            "attack_angle_w": 16,
            "sideslip_angle_w": 17,
            "l_w": 18,
            "d_w": 19,
            "c_w": 20,
            "wingspan": 21
        }

        # i found that saving inforamtion did not change runtime but for now it still here - maybe need to delete
        self.save_information = save_information  # if false do not save any information
        self.save_all_information = True  # decides whether to save only x,y,z,velocity or all information
        self.timestamps = np.arange(0, num_of_seconds, self.dt)
        if self.save_information:
            num_params_route_data = len(self.param_to_index) if self.save_all_information else 4
            self.route_data = np.zeros(shape=(self.timestamps.size, num_params_route_data))

    """
        calculations
    """

    def calculate_drag(self, v, attack_angle, sideslip_angle):
        return (self.rho * self.S * self.calculate_Cd(attack_angle, sideslip_angle) * v ** 2) / 2

    def calculate_lift(self, v, attack_angle, sideslip_angle):
        return (self.rho * self.S * self.calculate_Cl(attack_angle, sideslip_angle) * v ** 2) / 2

    def calculate_side(self, v, attack_angle, sideslip_angle):
        return (self.rho * self.S * self.calculate_Cc(attack_angle, sideslip_angle) * v ** 2) / 2

    # calculations from nasa paper
    # help from: https://github.com/SuReLI/L2F-sim/blob/master/src/aircraft/beeler_glider/beeler_glider.hpp
    def calculate_Cl(self, alhpa, beta):
        return self.C_L_alpha * (alhpa - self.alpha0)

    def calculate_Cd(self, alpha, beta):
        Cl = self.calculate_Cl(alpha, beta)
        Cc = self.calculate_Cc(alpha, beta)
        return self.C_D_0 + self.C_d_L * np.power(Cl - self.C_L_min, 2) + (
                np.power(Cl, 2) + np.power(Cc, 2) * (self.S / self.S_V)) / (np.pi * self.e * self.aspect_ratio)

    def calculate_Cc(self, alpha, beta):
        return self.C_C_beta * beta

    """
        support functions
    """

    def is_out_of_bounds(self, curr_pos):
        """
        check out of bounds in location
        :param curr_pos:
        :return: True iff glider out of bounds
        """
        if curr_pos[0] < self.x_bounds[0] or curr_pos[0] > self.x_bounds[1]:
            return True
        if curr_pos[1] < self.y_bounds[0] or curr_pos[1] > self.y_bounds[1]:
            return True
        if curr_pos[2] < self.z_bounds[0] or curr_pos[2] > self.z_bounds[1]:
            return True
        return False

    def arccos_bound(self, val):
        """
        in order to deal with computational errors
        :param val:
        :return:
        """
        return np.arccos(self.bound_one(val))

    def arcsin_bound(self, val):
        """
        in order to deal with computational errors
        :param val:
        :return:
        """
        return np.arcsin(self.bound_one(val))

    def bound_one(self, val):
        """
        bound the values between 1 and -1 for arccos and arcsin
        :param val:
        :return:
        """
        if val > 1:
            return 1
        if val < -1:
            return -1
        return val

    def time_to_index(self, t):
        return int(t / self.dt)

    """
        set functions
    """

    def set_bank_angle(self, new_bank_angle):
        """
        set the bank angle
        :param new_bank_angle: angle in degrees
        :return:
        """
        self.bank_angle = np.deg2rad(new_bank_angle)

    def set_param_by_name(self, param_name, param_value):
        """
        set the param to value
        :param param_value: the new value of param (angle in degrees)
        :param param_name: the name of the angle
        :return:
        """
        if param_name == "bank_angle":
            self.bank_angle = np.deg2rad(param_value)
        elif param_name == "sideslip_angle":
            self.sideslip_angle = np.deg2rad(param_value)
        elif param_name == "attack_angle":
            self.attack_angle = np.deg2rad(param_value)
        elif param_name == "velocity":
            self.v = param_value
        elif param_name == "x":
            self.x = param_value
        elif param_name == "y":
            self.y = param_value
        elif param_name == "z":
            self.z = param_value
        elif param_name == "wingspan":
            self.set_wing_parameters(param_value)

    def set_angles(self, new_bank=None, new_attack=None, new_sideslip=None):
        """
        set the [bank, attack, sideslip] from new_bank, new_attack, new_sideslip
        :param new_bank:
        :param new_attack:
        :param new_sideslip:
        :return:
        """
        if new_bank is not None:
            self.bank_angle = np.deg2rad(new_bank)
        if new_attack is not None:
            self.attack_angle = np.deg2rad(new_attack)
        if new_sideslip is not None:
            self.sideslip_angle = np.deg2rad(new_sideslip)

    def set_wing_parameters(self, wingspan, aspect_ratio=None):
        """
        set the wing parameters, for wing loading
        :param wingspan:
        :param aspect_ratio: if None, keep the same width and create the aspect ratio
        :return:
        """
        curr_wingspan = self.wingspan
        curr_aspect_ratio = self.aspect_ratio

        if aspect_ratio is None:
            new_aspect_ratio = wingspan * curr_aspect_ratio / curr_wingspan  # keep the same width
            self.aspect_ratio = new_aspect_ratio
        else:
            self.aspect_ratio = aspect_ratio

        self.wingspan = wingspan
        self.update_aerodynamic_constants()

    def update_aerodynamic_constants(self):
        """
        update the aerodynamic constants by wingspan and aspect ratio
        :return:
        """
        self.AR_V = 0.5 * self.aspect_ratio
        self.lt = 0.28 * self.wingspan
        self.V_H = 0.4
        self.V_V = 0.02
        self.c_ = 1.03 * self.wingspan / self.aspect_ratio
        self.S = self.wingspan * self.wingspan / self.aspect_ratio
        # best function to fit to the NASA paper data (here we calculate in meter and the paper uses inch)
        self.S_F = 0.01553571429 * np.power(self.wingspan, 2) + 0.01950357142 * self.wingspan - 0.01030412685
        self.S_T = self.V_H * self.c_ * self.S / self.lt
        self.S_V = self.V_V * self.wingspan * self.S / self.lt
        self.e = 0.95
        self.Re = 150000
        self.a0 = 0.1 * (180 / np.pi)
        self.alpha0 = -2.5 * (np.pi / 180)
        self.C_d_0 = 0.01
        self.C_d_L = 0.05
        self.C_L_min = 0.4
        self.C_L_alpha = self.a0 / (1 + self.a0 / (np.pi * self.e * self.aspect_ratio))
        self.C_C_beta = (self.a0 / (1 + self.a0 / (np.pi * self.e * self.AR_V))) * (self.S_V / self.S)
        self.C_D_F = 0.008
        self.C_D_T = 0.01
        self.C_D_E = 0.002
        self.C_D_0 = (self.C_D_F * self.S_F / self.S) + (self.C_D_T * (
                self.S_T + self.S_V) / self.S) + self.C_D_E + self.C_d_0

    """
        get function - for RL
    """

    def get_curr_state_for_learning(self, time_back):
        """
        :param time_back: the number of time in the past that each state have
        :return: dictionary of current state for all the time back, angles in degrees
        """
        initial_params = self.param_to_index.keys()
        curr_state_dict = {param: [] for param in initial_params}
        for param in initial_params:
            if "angle" in param:
                curr_state_dict[param] = np.array(
                    [np.rad2deg(self.get_param_in_time(param, self.curr_time - i)) for i in
                     range(time_back)])
            else:
                curr_state_dict[param] = np.array(
                    [self.get_param_in_time(param, self.curr_time - i) for i in range(time_back)])

        # get time series of params
        wind_diffs = np.array([self.get_wind_diff_in_wings(self.curr_time - i) for i in range(time_back)])
        curr_state_dict["wind_diffs"] = wind_diffs

        # add wind direction
        horizontal_wind = self.thermal.horizontal_wind
        wind_angle = np.array([np.rad2deg(np.arctan2(horizontal_wind[1], horizontal_wind[0])) for i in range(time_back)])
        curr_state_dict["wind_angle"] = wind_angle
        curr_state_dict["wind_speed"] = np.array([np.linalg.norm(horizontal_wind) for i in range(time_back)])

        # list of len time_back of the centers
        for num in range(self.thermal.num_thermals):
            # this is legacy code for work with versions before
            thermal_center = np.array(
                [self.thermal.get_thermals_centers(curr_state_dict["z"][i], self.curr_time - time_back)[num] for i in
                 range(time_back)])
            curr_state_dict[f"thermal_center_{num}"] = thermal_center

            # this is the fixed center
            thermal_center_fixed = np.array(
                [self.thermal.get_thermals_centers(curr_state_dict["z"][i], self.curr_time - i)[num] for i in
                 range(time_back)])
            curr_state_dict[f"fixed_thermal_center_{num}"] = thermal_center_fixed

        # add time
        curr_state_dict["time"] = np.array([self.curr_time - i for i in range(time_back)])

        # add vz from velocity and glide angle
        curr_state_dict["vz"] = curr_state_dict["velocity"] * np.sin(np.deg2rad(curr_state_dict["glide_angle"]))

        curr_state_dict["angle_from_wind"] = (2 * (curr_state_dict["bank_angle"] >= 0) - 1) * (
                ((curr_state_dict["side_angle"] - curr_state_dict["wind_angle"]) % 360) - 180)

        return curr_state_dict

    def get_curr_state(self, t=-1):
        """
        :param t: the time want to get the state, t==-1 if want the current time
        :return: dictionary of current state
        """
        if t == -1:
            t = self.curr_time

        curr_state_dict = {
            "z": self.get_param_in_time("z", t),
            "y": self.get_param_in_time("y", t),
            "x": self.get_param_in_time("x", t),
            "velocity": self.get_param_in_time("velocity", t),
            "glide_angle": self.get_param_in_time("glide_angle", t),
            "side_angle": self.get_param_in_time("side_angle", t),
            "bank_angle": self.get_param_in_time("bank_angle", t),
            "attack_angle": self.get_param_in_time("attack_angle", t),
            "sideslip_angle": self.get_param_in_time("sideslip_angle", t),
            "curr_time": t
        }
        return curr_state_dict

    def get_wind(self, curr_pos, t):
        """
        get wind in location and in time
        :param curr_pos:
        :param t:
        :return: vector of wind direction
        """
        return self.thermal.get_wind_vel(curr_pos, t)

    def get_wind_diff_in_wings(self, t=-1):
        """
        calculate the wind differnce between right and left wings
        :param t: not -1 iff want to calculate the wind_diff from past time t
        :return: differnce in wind between right\left wings only at z axis
        """
        # TODO think about correctness
        if t == -1:
            t = self.curr_time

        curr_state = self.get_curr_state(t)

        # calculate relative vector of right wing from center
        rot_from_i_to_v = Rotation.from_euler('zyx', [curr_state["side_angle"], curr_state["glide_angle"],
                                                      curr_state["bank_angle"]])
        rot_from_v_to_b = Rotation.from_euler('yz', [curr_state["attack_angle"], curr_state["sideslip_angle"]])
        right_wing_rel_pos = rot_from_v_to_b.apply(rot_from_i_to_v.apply(np.array([0, -self.wingspan / 2, 0])))

        # calculate left and right eing positions
        curr_pos = np.array([curr_state["x"], curr_state["y"], curr_state["z"]])
        right_wing_pos = curr_pos + right_wing_rel_pos
        left_wing_pos = curr_pos - right_wing_rel_pos  # left wing is in opposite side of right wing

        # calculate diff
        left_wing_vel = self.get_wind(left_wing_pos, curr_state["curr_time"])
        right_wing_vel = self.get_wind(right_wing_pos, curr_state["curr_time"])

        return (right_wing_vel - left_wing_vel)[2]

    def get_z_height(self):
        """
        for reward function
        :return: cuurent z height
        """
        return self.z

    def get_z_velocity(self):
        """
        :return: velocity in z axis
        """
        return self.v * np.sin(self.glide_angle)

    def get_bank_angle(self):
        """
        :return: the current bank angle in degrees
        """
        return np.rad2deg(self.bank_angle)

    def get_aspect_ratio(self):
        """
        :return: the current aspect ratio
        """
        return self.aspect_ratio

    def get_wingspan(self):
        """
        :return: the current wingspan
        """
        return self.wingspan

    def get_param_by_name(self, param_name):
        """
        :param param_name: the name of the param
        :return: the current param by name
        """
        param_value = self.get_param_in_time(param_name, self.curr_time)
        if "angle" in param_name:
            return np.rad2deg(param_value)
        return param_value

    def get_param_in_time(self, param, t):
        """
        get_param_in_time("side_angle", 7) -> get the side angle 7 seconds after the
        simulated started
        :param param: the parameter want to get
        :param t: the time want to get
        :return: the parameter in the specific time
        """
        # the -1 is because doing calculation of [0,5] the last point calculated is 5-dt
        time_index = self.time_to_index(t) - 1
        if t == 0:  # the case of time zero (no need to take -1, need 0)
            time_index += 1

        return self.route_data[time_index, self.param_to_index[f"{param}"]]

        """
       simulation of glider with wind
    """

    def simple_glider_equations_without_wind(self, t, state):
        """
        equations for glider ode without wind
        :param t:
        :param state:
        :return: derivative value
        """
        x, y, z, v, glide_angle, side_angle = state
        l = self.calculate_lift(v, self.attack_angle, self.sideslip_angle)
        d = self.calculate_drag(v, self.attack_angle, self.sideslip_angle)
        c = self.calculate_side(v, self.attack_angle, self.sideslip_angle)
        dz = v * np.sin(glide_angle)
        dx = v * np.cos(side_angle) * np.cos(glide_angle)
        dy = v * np.sin(side_angle) * np.cos(glide_angle)
        dv = -d / self.m - self.g * np.sin(glide_angle)
        d_glide_angle = (l * np.cos(self.bank_angle) + c * np.sin(self.bank_angle)) / (self.m * v) - self.g * np.cos(
            glide_angle) / v
        d_side_angle = (l * np.sin(self.bank_angle) - c * np.cos(self.bank_angle)) / (self.m * v * np.cos(glide_angle))
        return [dx, dy, dz, dv, d_glide_angle, d_side_angle]

    def glider_equations_with_wind(self, t, state):
        """
        equations for glider ode with wind
        Important note - my orientation for angles is different than the paper
        :param t:
        :param state:
        :return:
        """
        # i - inertial world reference frame
        # v - velocity world reference frame
        # w - wind world reference frame
        # b - body world reference frame
        # m - inertial world after glide angle(in wind frame) and side angle(in wind frame) rotations refrence frame
        # vtag - velocity world reference frame without bank angle
        x, y, z, v, glide_angle, side_angle = state

        # find side and glide angle in wind frame
        rot_from_i_to_v_positive_glide = Rotation.from_euler('xyz', [self.bank_angle, -glide_angle, side_angle])
        v_vector_in_i = rot_from_i_to_v_positive_glide.apply(np.array([v, 0, 0]))
        w_vector_in_i = self.get_wind(np.array([x, y, z]), t)
        relative_velocity = v_vector_in_i - w_vector_in_i  # v - w
        relative_velocity_size = np.linalg.norm(relative_velocity)
        relative_velocity_direction = relative_velocity / relative_velocity_size
        glide_angle_in_w = self.arcsin_bound(relative_velocity_direction[2])
        side_angle_in_w = np.sign(relative_velocity_direction[1] / np.cos(glide_angle_in_w)) * self.arccos_bound(
            relative_velocity_direction[0] / np.cos(glide_angle_in_w))

        # find attack and sideslip angle in wind frame
        # from b to i, from i to m
        rot_from_i_to_v = Rotation.from_euler('xyz', [self.bank_angle, glide_angle, side_angle])
        rot_from_v_to_b = Rotation.from_euler('zy', [-self.sideslip_angle, self.attack_angle])
        rot_from_b_to_m = Rotation.from_euler('zy', [-side_angle_in_w, -glide_angle_in_w])
        third_col_of_m = rot_from_b_to_m.apply(rot_from_i_to_v.apply(rot_from_v_to_b.apply(np.array([0, 0, 1]))))
        second_col_of_m = rot_from_b_to_m.apply(rot_from_i_to_v.apply(rot_from_v_to_b.apply(np.array([0, 1, 0]))))
        first_col_of_m = rot_from_b_to_m.apply(rot_from_i_to_v.apply(rot_from_v_to_b.apply(np.array([1, 0, 0]))))
        attack_angle_in_w = self.arcsin_bound(third_col_of_m[0])
        bank_angle_in_w = np.sign(-third_col_of_m[1] / np.cos(attack_angle_in_w)) * self.arccos_bound(
            third_col_of_m[2] / np.cos(attack_angle_in_w))
        sideslip_angle_in_w = np.sign(second_col_of_m[0] / np.cos(attack_angle_in_w)) * self.arccos_bound(
            first_col_of_m[0] / np.cos(attack_angle_in_w))

        # calculate lift, drag and side
        l_w = self.calculate_lift(relative_velocity_size, attack_angle_in_w, sideslip_angle_in_w)
        d_w = self.calculate_drag(relative_velocity_size, attack_angle_in_w, sideslip_angle_in_w)
        c_w = self.calculate_side(relative_velocity_size, attack_angle_in_w, sideslip_angle_in_w)

        # calculate derivative
        dz = v * np.sin(glide_angle)
        dx = v * np.cos(side_angle) * np.cos(glide_angle)
        dy = v * np.sin(side_angle) * np.cos(glide_angle)

        rot_from_v_to_i = rot_from_i_to_v.inv()
        rot_from_i_to_w = Rotation.from_euler('xyz', [bank_angle_in_w, glide_angle_in_w, side_angle_in_w])
        forces_in_v = rot_from_v_to_i.apply(rot_from_i_to_w.apply(np.array([-d_w, -c_w, -l_w])))
        d_v, c_v, l_v = -forces_in_v

        dv = - d_v / self.m - self.g * np.sin(glide_angle)
        d_glide_angle = (c_v * np.sin(self.bank_angle) + l_v * np.cos(self.bank_angle)) / (
                self.m * v) - self.g * np.cos(glide_angle) / v
        d_side_angle = (l_v * np.sin(self.bank_angle) - c_v * np.cos(self.bank_angle)) / (
                self.m * v)

        # save l,d,c
        if self.save_information and self.save_all_information:
            self.route_data[self.time_to_index(t), self.param_to_index["l"]] = l_v
            self.route_data[self.time_to_index(t), self.param_to_index["d"]] = d_v
            self.route_data[self.time_to_index(t), self.param_to_index["c"]] = c_v
            self.route_data[self.time_to_index(t), self.param_to_index["l_w"]] = l_w
            self.route_data[self.time_to_index(t), self.param_to_index["d_w"]] = d_w
            self.route_data[self.time_to_index(t), self.param_to_index["c_w"]] = c_w
            self.route_data[self.time_to_index(t), self.param_to_index["glide_angle_w"]] = glide_angle_in_w
            self.route_data[self.time_to_index(t), self.param_to_index["side_angle_w"]] = side_angle_in_w
            self.route_data[self.time_to_index(t), self.param_to_index["attack_angle_w"]] = attack_angle_in_w
            self.route_data[self.time_to_index(t), self.param_to_index["bank_angle_w"]] = bank_angle_in_w
            self.route_data[self.time_to_index(t), self.param_to_index["sideslip_angle_w"]] = sideslip_angle_in_w
            self.route_data[self.time_to_index(t), self.param_to_index["velocity_w"]] = relative_velocity_size

        return [dx, dy, dz, dv, d_glide_angle, d_side_angle]

    def step(self, with_wind=True):
        """
        step in RL algorithm
        :return: True iff done (out_of_bounds or time)
        """
        # out of bounds - means that the glider is not in interesting place (bounds declared at the beginning)
        # out of time - means that the time is finished
        # movement error - means that there is an error in movement calculations mostly because of low speed and high wind
        ret_value = {"out_of_bounds": False, "out_of_time": False, "movement_error": False}
        # check - out of time bound
        end_time = self.curr_time + self.step_size_seconds
        if end_time >= self.num_of_seconds:
            end_time = self.num_of_seconds
            ret_value["out_of_time"] = True
            # the case of already finished
            if self.curr_time >= self.num_of_seconds:
                return ret_value
        # check position bound
        if self.is_out_of_bounds(np.array([self.x, self.y, self.z])):
            ret_value["out_of_bounds"] = True
            return ret_value
        # case of low velocities, not good state, calculations won't work
        if self.v < np.linalg.norm(self.thermal.horizontal_wind):
            ret_value["movement_error"] = True
            return ret_value

        # run the ODE solver
        init_values = [self.x, self.y, self.z, self.v, self.glide_angle, self.side_angle]
        t_eval = self.timestamps[self.time_to_index(self.curr_time):self.time_to_index(end_time)]
        t_span = (self.curr_time, end_time - self.dt / 2)  # - self.dt/2 in order to prevent illegal time calculation

        if with_wind:
            result = solve_ivp(self.glider_equations_with_wind, t_span, init_values, method="RK45",
                               t_eval=t_eval,
                               first_step=self.dt)
            # if not result.success:
            #     result = solve_ivp(self.glider_equations_with_wind_from_l2f, t_span, init_values, method="RK45",
            #                        t_eval=t_eval)
        else:
            result = solve_ivp(self.simple_glider_equations_without_wind, t_span, init_values, method="RK45",
                               t_eval=t_eval)

        # add data for future calculations and graphs
        if self.save_information:
            try:
                if self.save_all_information:
                    # case of saving all information
                    self.route_data[self.time_to_index(self.curr_time):self.time_to_index(end_time),
                    0:6] = np.transpose(
                        result.y)

                    self.route_data[self.time_to_index(self.curr_time):self.time_to_index(end_time), 6:9] = [
                        self.bank_angle,
                        self.attack_angle,
                        self.sideslip_angle]

                    self.route_data[self.time_to_index(self.curr_time):self.time_to_index(end_time),
                    self.param_to_index["wingspan"]] = [self.wingspan]
                else:
                    # save only x,y,z,velocity
                    self.route_data[self.time_to_index(self.curr_time):self.time_to_index(end_time),
                    0:4] = np.transpose(
                        result.y)[:, 0:4]
            except:
                print(f"error: {self.get_curr_state()}")
                ret_value["movement_error"] = True
                return ret_value

        self.curr_time = end_time
        self.x, self.y, self.z, self.v, self.glide_angle, self.side_angle = result.y[:, -1]

        return ret_value

    """
        draw the flight and some parameters graph
    """

    def draw_simulation(self, free_txt="", animation=False, return_fig=False, return_both=False,
                        num_samples_to_ignore=8, num_samples_to_smooth=3):
        """
        draw the simulation in 3d with colors by velocity
        :param num_samples_to_smooth: constant for thermal detection by vz > 0
        :param num_samples_to_ignore: constant for thermal detection by vz > 0
        :param return_both: True iff return both fig and animation
        :param return_fig: true iff return the figure
        :param animation: if wants to show animation
        :param free_txt:
        :return:
        """
        vels = self.route_data[:self.time_to_index(self.curr_time), self.param_to_index["velocity"]]
        x = self.route_data[:self.time_to_index(self.curr_time), self.param_to_index["x"]]
        y = self.route_data[:self.time_to_index(self.curr_time), self.param_to_index["y"]]
        z = self.route_data[:self.time_to_index(self.curr_time), self.param_to_index["z"]]
        # creating dataframe for animation
        index_to_param = {v: k for k, v in self.param_to_index.items()}
        columns_for_df = [index_to_param[i] for i in range(len(index_to_param))]
        df = pd.DataFrame(self.route_data[:self.time_to_index(self.curr_time), :], columns=columns_for_df)
        df["t"] = df.index * self.dt
        smaller_df = df[df.index % int(np.round(1 / self.dt)) == 0]  # no need to plot all the plots in every self.dt
        pd.options.mode.chained_assignment = None
        smaller_df['vz'] = df.apply(lambda row: row.velocity * np.sin(row.glide_angle), axis=1)
        smaller_df["bank_angle_deg"] = np.rad2deg(smaller_df["bank_angle"])
        smaller_df["glide_angle_deg"] = np.rad2deg(smaller_df["glide_angle"])
        smaller_df["attack_angle_deg"] = np.rad2deg(smaller_df["attack_angle"])
        smaller_df["sideslip_angle_deg"] = np.rad2deg(smaller_df["sideslip_angle"])
        smaller_df["side_angle_deg"] = np.rad2deg(smaller_df["side_angle"])

        # add the angles in the wind frame
        smaller_df["bank_angle_w_deg"] = np.rad2deg(smaller_df["bank_angle_w"])
        smaller_df["glide_angle_w_deg"] = np.rad2deg(smaller_df["glide_angle_w"])
        smaller_df["attack_angle_w_deg"] = np.rad2deg(smaller_df["attack_angle_w"])
        smaller_df["sideslip_angle_w_deg"] = np.rad2deg(smaller_df["sideslip_angle_w"])
        smaller_df["side_angle_w_deg"] = np.rad2deg(smaller_df["side_angle_w"])

        # add thermal classification
        smaller_df = data_utils.add_thermal_classification_for_glider(smaller_df, vz_column="vz")

        figs = dict()
        if animation or return_both:
            fig = self.draw_animation(smaller_df, free_txt)
            if return_both:
                figs["animation"] = fig
            elif return_fig:
                return fig
            else:
                fig.show()

        if not animation or return_both:
            # scatter3D
            use_plotly = True
            if use_plotly:
                title_txt = "Glider Flight for {0} seconds {1}".format(int(self.curr_time), free_txt)
                fig = px.scatter_3d(smaller_df, x="x", y="y", z="z", color="vz", title=title_txt,
                                    hover_data=["t", "velocity", "bank_angle_deg", "attack_angle_deg",
                                                "sideslip_angle_deg",
                                                "glide_angle_deg", "side_angle_deg", "x", "y", "z", "wingspan",
                                                "is_thermal"],
                                    color_continuous_scale=px.colors.sequential.YlOrRd)

                # plot the thermals centers
                plot_thermal = True  # not plotting the thermal because it can be moving
                if plot_thermal:
                    thermals_plots = self.get_thermals_plots(smaller_df)
                    for thermal_plot in thermals_plots:
                        fig.add_trace(thermal_plot)
                fig.update_layout(scene=self.get_ranges_of_df_for_plot(smaller_df))
                fig.update_layout(legend_orientation="h")  # providing legend and colorbar from overlap

                if return_both:
                    figs["figure"] = fig
                elif return_fig:
                    return fig
                else:
                    fig.show()
            else:
                ax = plt.axes(projection='3d')
                cm = plt.cm.get_cmap('YlOrRd')
                sc = ax.scatter(x, y, z, c=vels, cmap=cm)
                plt.colorbar(sc)
                plt.title("Glider Flight for {0} seconds {1}".format(int(self.curr_time), free_txt))

                if return_fig:
                    return plt
                plt.show()

        if return_both:
            # draw a graph of the control parameters during the flight
            fig = graphical_utils.draw_multiple_graphs(smaller_df, x="t",
                                                       y_list=["bank_angle_w_deg", "attack_angle_w_deg",
                                                               "sideslip_angle_w_deg",
                                                               "bank_angle_deg", "attack_angle_deg",
                                                               "sideslip_angle_deg",
                                                               "wingspan"])

            figs["controls"] = fig

            return figs

    def get_thermals_plots(self, df, t=-1):
        """
        function for getting plots of line that represent the center of thermal
        :param t: parameter t for case of moving wind, if t==-1 take thermal center for each z
        :param df:
        :param self:
        :return: plot of all the center of the thermals
        """
        max_wind_z = self.thermal.get_max_wind_z()
        axis_ranges = self.get_ranges_of_df_for_plot(df, return_ranges=True)

        if t == -1:
            thermal_z = df["z"].to_numpy()
            thermal_t = df["t"].to_numpy()
        else:
            thermal_z = np.linspace(axis_ranges["z"][0], axis_ranges["z"][1], 100)
            thermal_t = np.repeat(t, len(thermal_z))

        thermals_centers = np.array(
            [self.thermal.get_thermals_centers(z, thermal_t[i]) for i, z in enumerate(thermal_z)])
        thermals_x = thermals_centers[:, :, 0].T
        thermals_y = thermals_centers[:, :, 1].T
        thermals_plots = []

        for i in range(len(thermals_x)):
            thermals_velocity = []  # add text of the velocity of thermal
            for j, z in enumerate(thermal_z):
                current_loc = np.array([thermals_x[i][j], thermals_y[i][j], thermal_z[j]])
                thermals_velocity.append(self.thermal.get_wind_vel(current_loc, thermal_t[j]))
            thermals_plots.append(go.Scatter3d(
                x=thermals_x[i], y=thermals_y[i], z=thermal_z,
                text=[
                    f"wx: {vel[0]}<br>wy: {vel[1]}<br>wz: {vel[2]}<br>t: {thermal_t[t]}"
                    for t, vel in enumerate(thermals_velocity)],
                hoverinfo="x+y+z+text",
                marker=dict(
                    color=[vel[2] for vel in thermals_velocity],
                    size=2,
                    cmax=max_wind_z,
                    cmin=0,
                    colorscale='Blues',
                    showscale=False,
                    colorbar=dict(title='wind_z')
                ),
                line=dict(
                    color=[vel[2] for vel in thermals_velocity],
                    width=2,
                    cmax=max_wind_z,
                    cmin=0,
                    colorscale='Blues',
                    showscale=False,
                    colorbar=dict(title='wind_z')
                ),
                name=f"Thermal {i} Center"
            ))

        return thermals_plots

    def draw_animation(self, df, free_txt=""):
        """
        function for drawing x,y,z animation from given df, animation
        :param df:
        :return:
        """
        dt = df["t"].iloc[1]
        num_seconds_per_frame = 0.1
        # the case cannot plot in num_seconds_per_frame
        if dt > num_seconds_per_frame:
            num_seconds_per_frame = dt

        thermals_plots = self.get_thermals_plots(df, t=-1)  # get the plot for the thermals centers
        num_thermals = len(thermals_plots)
        min_vz = df["vz"].min()
        max_vz = df["vz"].max()

        # Create figure
        fig = go.Figure([go.Scatter3d(x=df["x"],
                                      y=df["y"],
                                      z=df["z"],
                                      mode='markers',
                                      text=[
                                          f"bank_angle_deg: {df.loc[i]['bank_angle_deg']:.2f}<br>velocity: {df.loc[i]['velocity']:.2f}<br>t: {df.loc[i]['t']}"
                                          f"<br>attack_angle_deg: {df.loc[i]['attack_angle_deg']:.2f}<br>sideslip_angle_deg: {df.loc[i]['sideslip_angle_deg']:.2f}"
                                          f"<br>glide_angle_deg: {df.loc[i]['glide_angle_deg']:.2f}<br>side_angle_deg: {df.loc[i]['side_angle_deg']:.2f}<br>wingspan: {df.loc[i]['wingspan']:.2f}"
                                          for i in df.index],
                                      hoverinfo="x+y+z+text",
                                      marker=dict(
                                          size=6,
                                          symbol=["circle" if i == 1 else "square" for i in
                                                  df["is_thermal"]],
                                          color=df["vz"],
                                          # set color to an array/list of desired values
                                          colorscale='YlOrRd',  # choose a colorscale
                                          opacity=0.8,
                                          showscale=True,
                                          colorbar=dict(title='velocity_z'),
                                          cmin=min_vz,
                                          cmax=max_vz
                                      ),
                                      name="Vulture Route"
                                      )] + thermals_plots)
        fig.update_layout(legend_orientation="h")  # providing legend and colorbar from overlap

        # Frames
        frames = [go.Frame(data=[go.Scatter3d(x=df["x"][:k + 1],
                                              y=df["y"][:k + 1],
                                              z=df["z"][:k + 1],
                                              mode='markers',
                                              text=[
                                                  f"bank_angle: {df.loc[i]['bank_angle_deg']:.2f}<br>velocity: {df.loc[i]['velocity']:.2f}<br>t: {df.loc[i]['t']}"
                                                  f"<br>attack_angle: {df.loc[i]['attack_angle_deg']:.2f}<br>sideslip_angle: {df.loc[i]['sideslip_angle_deg']:.2f}"
                                                  f"<br>glide_angle: {df.loc[i]['glide_angle_deg']:.2f}<br>side_angle: {df.loc[i]['side_angle_deg']:.2f}<br>wingspan: {df.loc[i]['wingspan']:.2f}"
                                                  for i in df.index[:k + 1]],
                                              hoverinfo="x+y+z+text",
                                              marker=dict(
                                                  size=6,
                                                  color=df["vz"][:k + 1],
                                                  symbol=["circle" if i == 1 else "square" for i in
                                                          df["is_thermal"][:k + 1]],
                                                  # set color to an array/list of desired values
                                                  colorscale='YlOrRd',  # choose a colorscale
                                                  opacity=0.8,
                                                  showscale=True,
                                                  colorbar=dict(title='velocity_z'),
                                                  cmin=min_vz,
                                                  cmax=max_vz
                                              )
                                              )
                                 ] + self.get_thermals_plots(df, t=k * num_seconds_per_frame),
                           traces=[i for i in range(num_thermals + 1)],
                           name=f'glider flight - {k * num_seconds_per_frame} seconds'
                           ) for k in range(1, len(df["x"]), int(num_seconds_per_frame / dt))
                  ]

        fig.update(frames=frames)

        sliders = [
            {"pad": {"b": 10, "t": 60},
             "len": 0.9,
             "x": 0.1,
             "y": 0,

             "steps": [
                 {"args": [[f.name], graphical_utils.frame_args(10)],
                  "label": f"{k * num_seconds_per_frame : .3f}",
                  "method": "animate",
                  } for k, f in enumerate(fig.frames)
             ]
             }
        ]

        fig.update_layout(

            updatemenus=[{"buttons": [
                {
                    "args": [None, graphical_utils.frame_args(50)],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [[None], graphical_utils.frame_args(0)],
                    "label": "Pause",
                    "method": "animate",
                }],

                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
            ],
            sliders=sliders
        )

        fig.update_layout(scene=self.get_ranges_of_df_for_plot(df))

        fig.update_layout(
            title_text=f'Glider flight for {num_seconds_per_frame * len(fig.frames):.2f} seconds {free_txt}')
        fig.update_layout(sliders=sliders)
        return fig

    @staticmethod
    def get_ranges_of_df_for_plot(df, return_ranges=False):
        """
        get the df and return the scene dictionary for plot;y
        :param return_ranges: True iff want the ranges and not the scene
        :param df:
        :return:
        """
        max_distance = max([max(df["x"]) - min(df["x"]), max(df["y"]) - min(df["y"]), max(df["z"]) - min(df["z"])])
        middle_points = [(max(df["x"]) + min(df["x"])) / 2, (max(df["y"]) + min(df["y"])) / 2,
                         (max(df["z"]) + min(df["z"])) / 2]
        ranges = {
            "x": [middle_points[0] - max_distance / 2, middle_points[0] + max_distance / 2],
            "y": [middle_points[1] - max_distance / 2, middle_points[1] + max_distance / 2],
            "z": [middle_points[2] - max_distance / 2, middle_points[2] + max_distance / 2]
        }

        if return_ranges:
            return ranges

        scene = dict(xaxis=dict(range=ranges["x"], autorange=False),
                     yaxis=dict(range=ranges["y"], autorange=False),
                     zaxis=dict(range=ranges["z"], autorange=False),
                     aspectmode="cube"
                     )

        return scene

    def generic_draw(self, lst_of_params, calc_func=None, calc_func_name=""):
        """
        plot in same graphs every param in lst_of_params, or can run function with calc_func(params)
        for example if we want to calculate energy:
        calc_func = lambda x: g*x[0]+0.5x[1]**2, lst_of_params = ["z","velocity"], calc_func_name = "energy"
        :param lst_of_params: list of parametrs that we want to draw, such as "x", "sideslip_angle"
        :param calc_func: functions on params
        :param calc_func_name: name of the function graph
        :return:
        """
        curr_time_index = self.time_to_index(self.curr_time)
        if calc_func is not None:
            # case of function graph
            plt.title("{0} graph".format(calc_func_name))
            args = [self.route_data[:curr_time_index, self.param_to_index[param]] for param in lst_of_params]
            func_lst = calc_func(args)
            plt.plot(self.timestamps[:curr_time_index], func_lst)
        else:
            # case of regular graph
            plt.title("{0} graph".format(",".join(lst_of_params)))
            for param in lst_of_params:
                plt.plot(self.timestamps[:curr_time_index],
                         self.route_data[:curr_time_index, self.param_to_index[param]],
                         label=param)
            plt.legend()
        plt.show()



if __name__ == "__main__":
    # in order to get off scientific mode and show rotations plots:
    # https://intellij-support.jetbrains.com/hc/en-us/community/posts/115000742910-SciView-3d-plots-not-draggable-rotationable
    pass
