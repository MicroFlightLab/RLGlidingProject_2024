import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class ThermalArea:
    def __init__(self, x_bounds=[-500, 500], y_bounds=[-500, 500], z_bounds=[0, 1000],
                 thermal_height=2000, w_star=10, horizontal_wind=None, mode="moving", noise_wind=None, total_time=200,
                 time_for_noise_change=5, thermals_settings_list=[], decay_by_time_factor=lambda t: 1):
        """
        old version:
        class for representing an area between: x_bounds, y_bounds, z_bounds (bounds=[bound0,bound1])

        new version:
        old version still compatible but in the new version can use additional wind
        :param x_bounds:
        :param y_bounds:
        :param z_bounds:
        :param total_time: the total time of the thermal
        :param time_for_noise_change: time for noise change, if the area is very turbulent, the noise will change more often
        :param horizontal_wind: the horizontal wind velocity
        :param noise_wind: the power of the gusts / the noise in the wind (random gaussian wind for each direction)
        :param mode: "moving" "fixed" or "mountain" (mountain is for mountain thermal, fixed is for fixed center thermal)
        :param w_star: the strength of the thermal
        :param thermal_height: the height of the thermal
        :param thermals_settings_list: list of dictionaries of the form:
        {"mode": "moving", "center": [x,y,z], "w_star": w_star, "thermal_height": thermal_height, "decay_by_time_factor": decay_by_time_factor}
        :param decay_by_time_factor: function of time that returns the decay factor of the thermal
        """
        self.thermals_settings_list = thermals_settings_list
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.horizontal_wind = np.array(horizontal_wind) if horizontal_wind is not None else np.array([0, 0, 0])
        self.noise_wind = np.array(noise_wind) if noise_wind is not None else np.array([0, 0, 0])
        self.w_star = w_star
        self.thermal_height = thermal_height
        self.mode = mode
        self.decay_by_time_factor = decay_by_time_factor
        self.num_thermals = 0  # initialize the number of thermals

        # create noise for total_time/time_for_noise_change for all 3 axis by the noise wind
        self.time_for_noise_change = time_for_noise_change
        self.total_time = total_time
        self.noise_in_time = np.random.normal(0, self.noise_wind,
                                              (int(self.total_time / self.time_for_noise_change), 3))

        # creating environment
        self.thermals = []
        if len(self.thermals_settings_list) == 0:
            # set default thermal in case of no thermals
            self.thermals_settings_list = [
                {"mode": self.mode, "center": [np.mean(self.x_bounds), np.mean(self.y_bounds), np.mean(self.z_bounds)],
                 "w_star": self.w_star, "thermal_height": self.thermal_height,
                 "decay_by_time_factor": self.decay_by_time_factor}]

        for thermal_dict in self.thermals_settings_list:
            centers = thermal_dict["center"]
            mode = thermal_dict["mode"]
            w_star = thermal_dict["w_star"]
            thermal_height = thermal_dict["thermal_height"]
            decay_by_time_factor = thermal_dict["decay_by_time_factor"]
            self.add_thermal(centers, w_star=w_star, thermal_height=thermal_height, mode=mode,
                             decay_by_time_factor=decay_by_time_factor)
            self.num_thermals += 1

    def add_thermal(self, centers, w_star=None, thermal_height=None, mode=None, decay_by_time_factor=None):
        """
        add a thermal to the area
        :param centers:
        :param w_star:
        :param thermal_height:
        :param mode:
        :param decay_by_time_factor:
        :return:
        """
        if mode is None:
            mode = self.mode
        if w_star is None:
            w_star = self.w_star
        if thermal_height is None:
            thermal_height = self.thermal_height
        if decay_by_time_factor is None:
            decay_by_time_factor = self.decay_by_time_factor
        center_x = centers[0]
        center_y = centers[1]
        center_z = centers[2]
        # determine the center by height and time
        if mode == "moving":
            # moving thermal
            center_x_func = lambda z, t: center_x + self.horizontal_wind[0] * t
            center_y_func = lambda z, t: center_y + self.horizontal_wind[1] * t
        elif mode == "fixed":
            center_x_func = lambda z, t: center_x
            center_y_func = lambda z, t: center_y
        else:
            # static thermal of mountain
            incline = [0, 0, 0]
            if w_star != 0:
                # constant from linearity of wind model and max approximation
                # 0.4576*w_star is the max vz(velocity upward)
                incline = self.horizontal_wind / (w_star * 0.4576)
            center_x_func = lambda z, t: center_x + (z - center_z) * incline[0]
            center_y_func = lambda z, t: center_y + (z - center_z) * incline[1]
        new_thermal = Thermal(x_center=center_x_func,
                              y_center=center_y_func,
                              w_star=w_star, zi=thermal_height, decay_by_time_factor=decay_by_time_factor)

        self.thermals.append(new_thermal)

    def get_noise(self, curr_pos, t):
        """
        return the noise in the wind
        :param t: time
        :param curr_pos: position
        :return: the noise in the wind
        """
        return self.noise_in_time[
            int(t / self.time_for_noise_change) % int(self.total_time / self.time_for_noise_change)]

    def get_wind_vel(self, curr_pos, t):
        """
        get the total wind velocity
        :param curr_pos: [x,y,z]
        :param t: time
        :return: total wind velocity from all thermals
        """
        total_wind = np.array([0., 0., 0.])
        for thermal in self.thermals:
            total_wind += thermal.get_wind_vel(curr_pos, t)

        # add the horizontal wind
        total_wind += self.horizontal_wind
        # add gaussian noise from gusts
        total_wind += self.get_noise(curr_pos, t)
        return total_wind

    def get_thermals_centers(self, z, t):
        """
        :param t: time
        :param z:
        :return: the centers of thermal(strongest wind) in given height
        """
        centers = []
        for thermal in self.thermals:
            centers.append(thermal.get_center(z, t))

        return np.array(centers)

    def get_max_wind_z(self):
        """
        get the max possible wind velocity in z direction
        :return:
        """
        max_wind_z = 0
        for thermal in self.thermals:
            thermal_max_wind_z = thermal.get_max_wind_z()
            if thermal_max_wind_z > max_wind_z:
                max_wind_z = thermal_max_wind_z
        return max_wind_z


class Thermal:
    def __init__(self, x_center=lambda z, t: 0, y_center=lambda z, t: 0, zi=1000, w_star=2.5,
                 decay_by_time_factor=lambda t: 1):
        """
        class for representing single thermal
        implementation from paper https://websites.isae-supaero.fr/IMG/pdf/report.pdf
        cpp code https://github.com/SuReLI/L2F-sim/blob/master/src/flight_zone/thermal/std_thermal.hpp
        :param x_center: function of x-coordinate center of thermal from z
        :param y_center: function of y-coordinate center of thermal from z
        :param zi: convective mixing-layer thickness (height scale)
        :param w_star: height scale of the thermal
        :param decay_by_time_factor: control the velocity by time (decay of the thermal)
        """
        self.x_center = x_center
        self.y_center = y_center
        self.zi = zi  # height scale of the thermal
        self.w_star = w_star  # the convective velocity scale
        self.decay_by_time_factor = decay_by_time_factor  # control the velocity by time (decay of the thermal)

    def get_wind_vel(self, curr_pos, t):
        """
        get the total wind velocity by position and time
        :param curr_pos: [x,y,z]
        :param t: time
        :return: total wind velocity
        """
        # need to add option for more wind models if needed
        z = curr_pos[2]
        r = np.linalg.norm(curr_pos[:2] - np.array([self.x_center(z, t), self.y_center(z, t)]))
        if z <= 0:
            return np.array([0, 0, 0])
        return self.lenschow_model(r, z) * self.decay_by_time_factor(t)

    def get_center(self, z, t):
        """
        :param t: time
        :param z: height
        :return: the center of thermal at given height
        """
        return np.array([self.x_center(z, t), self.y_center(z, t)])

    def lenschow_model(self, r, z):
        """
        implementation of lenschow model
        :param r: radius
        :param z: height
        :return: total wind
        """
        if z > self.zi:
            return 0
        z_zi_ratio = z / self.zi
        z_zi_third_power = np.power(z_zi_ratio, 1 / 3)
        d = 0.16 * z_zi_third_power * (1 - 0.25 * z_zi_ratio) * self.zi  # diameter of thermal
        w_peak = self.w_star * z_zi_third_power * (1 - 1.1 * z_zi_ratio)  # wind velocity in center of gaussian

        w_total = w_peak * np.exp(-np.power(r, 2) / np.power(d / 2, 2)) * (1 - np.power(r, 2) / np.power(d / 2, 2))
        return np.array([0, 0, w_total])

    def draw_xy(self, height):
        """
        draw wind velocity at z=height y=0, -100<x<100
        :param height:
        :return:
        """
        min_place = -1000
        max_place = 1000
        plt.title("Wind Velocity graph at height {0}".format(height))
        plt.xlabel("x/y")
        locations = [i for i in range(min_place, max_place)]
        velocities = [self.get_wind_vel([i, 0, height], 0) for i in range(min_place, max_place)]
        plt.plot(locations, velocities)
        plt.show()

    def draw_z(self, x=0, y=0):
        """
        draw wind velocity at (x,y), 0<z<1000
        :param x:
        :param y:
        :return:
        """
        min_range = 1
        max_range = 1000
        locations = [i for i in range(min_range, max_range)]
        velocities = [self.get_wind_vel([x, y, i], 0)[2] for i in range(min_range, max_range)]
        plt.title(f"Wind Velocity graph at height {x},{y}")
        plt.xlabel("z")
        plt.plot(locations, velocities)
        plt.show()

    def calculate_thermal_place_z_with_wind(self, r, w):
        der_func = lambda z, state: [w / self.lenschow_model(r, z)[2]]
        init_values = [r]
        t_eval = np.linspace(5, 1500, 1000)
        t_span = (5, 1500)
        result = solve_ivp(der_func, t_span, init_values, t_eval=t_eval)
        plt.ylabel("z")
        plt.ylabel("r")
        plt.plot(result.y[0], result.t)

    def get_max_wind_z(self):
        """
        :return: the max wind velocity at the thermal
        """
        # approximation - little bit more than real
        return self.w_star/2


if __name__ == "__main__":
    pass
