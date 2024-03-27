import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.signal import savgol_filter
import scipy.integrate as it
from bokeh.models import ColumnDataSource, GMapOptions, HoverTool
from bokeh.io import show, output_notebook
from bokeh.plotting import gmap, save
from bokeh.transform import linear_cmap
from bokeh.palettes import Plasma256 as palette
from bokeh.models import ColorBar
import os
import plotly.express as px
import plotly.graph_objects as go
import random

# output_notebook()

"""
Explanation about Annontated1HzData:

Vertical speed - calculated by dv/dt

Thermals compuation:
1 - gliding - v_z < 0
2 - thermalling - v_z > 0 and intersections
3 - linear soaring - v_z > 0 and no intersections


all_data.pkl df:

date - the date of the measure
time_delta - the delta between 2 consecutive measurements
is_continuous - True iff 0 < time_delta < num_seconds(3)
route_num - every continouns measurements are count as route
x,y - the x,y coordinates from lat and lon
d{n}{param} - the n'th derivative of param
bank_angle - estimated bank angle by the side force
"""

# need to put the dataframes in this path
dataframes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataFrames")


class VulturesData:
    def __init__(self, read_data=True, data_type="regular"):
        # for progress bar
        tqdm.pandas()

        self.poly_degree = 6
        self.num_samples = 13
        # the number of samples to smooth a thermal and take all the relevant parts
        self.thermal_number_of_samples_smooth = 15
        # the intersection distance threshold for thermal classifier
        self.intersect_distance_threshold = 20

        self.read_data = read_data
        self.info_params = ["YearBorn", "Weight", "Sex", "WingSpan", "WingArea"]
        # data type can be "regular" - the data we got from Roi, "human_walter" - the data from walter flight
        # "new" - the bigger data from the csv
        self.data_type = data_type

        self.thermal_name = fr"{dataframes_path}\{self.data_type}_thermal_data_{self.num_samples}samples_{self.poly_degree}degree.pkl"
        is_filter_continues = False  # if True then filter only continues data

        print(f"loading data {data_type}")
        if data_type == "regular":
            name = rf"{dataframes_path}\all_data_{0}samples_{1}degree.pkl".format(
                self.num_samples, self.poly_degree)
        elif data_type == "new":
            name = fr"{dataframes_path}\new_big_data_{self.num_samples}samples_{self.poly_degree}degree.pkl"
            is_filter_continues = True  # for the data to be not so big
        elif data_type == "human_walter":
            # human_walter
            name = rf"{dataframes_path}\real_gliding_data_{self.num_samples}samples_{self.poly_degree}degree.pkl"

        if not read_data:
            tqdm.pandas()  # to see bars in python
            # the gliding_data is the data from gliding with walter

            if data_type == "regular":
                self.df = self.load_data_to_df()
            elif data_type == "new":
                self.df = self.load_data_from_big_csv()
            else:
                # human_walter
                self.df = self.load_data_from_walter_files()
            print("finished first loading")
            if data_type != "human_walter":
                self.add_info()
            print("finished adding info")
            self.filter_by_continuity(filter_continues=is_filter_continues)
            print("finished continuity filter")
            self.add_x_y()
            print("finished adding x and y")
            self.calculate_derivatives()
            print("finished calculating derivatives")
            if data_type == "new":
                # filter the standing data (when the vulture is not moving) and add ClsSmthTrack(thermal classification)
                self.filter_standing_add_ClsSmthTrack()
                print("finished filter standing")
            self.calculate_bank()
            print("finished calculating angles")
            self.add_thermal_classification()
            print("finished thermal classification")
            self.add_velocity()
            print("finished adding velocity")
            self.df.to_pickle(name)
            print("finished saving the df, now creating thermal df")
            self.save_thermal_df()
            print("finished all")
        else:
            if data_type != "human_walter":
                self.thermal_df = pd.read_pickle(self.thermal_name)
            self.df = pd.read_pickle(name)

    def load_data_from_big_csv(self):
        """
        load the data from the new big csv
        :return:
        """
        df = pd.read_csv(
            rf"{dataframes_path}\E-obs Gsm Vultures Israel.csv",
            usecols=["height-above-ellipsoid", "location-lat", "location-long", "tag-local-identifier", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.rename(columns={"height-above-ellipsoid": "z", "location-lat": "lat", "location-long": "lon",
                                "tag-local-identifier": "tag", "timestamp": "date"})
        df = df.dropna()
        df["is_young(2013)"] = df["tag"].isin([3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188])

        return df

    def load_data_from_walter_files(self):
        """
        data from sensebox flight
        :return:
        """
        # read the csv
        gliding_df = pd.read_csv(
            rf"{dataframes_path}\SenseBoxFlight_15062022.csv",
            encoding="ISO-8859-1", delimiter=";")
        # drop NA
        gliding_df = gliding_df.dropna()
        gliding_df = gliding_df.reset_index()

        # change type to float
        for c in gliding_df.columns:
            gliding_df[c] = gliding_df[c].astype("float")

        # lon lat for x,y
        gliding_df["lon"] = gliding_df["Longitude"] / 1e7
        gliding_df["lat"] = gliding_df["Latitude"] / 1e7
        gliding_df["z"] = gliding_df["Height_msl"]
        gliding_df["date"] = pd.to_datetime(gliding_df["Time"].astype(float), unit='s')

        # combine data to 1HZ
        group_by_date = gliding_df.groupby(["date"])
        all_values = {}
        for c in gliding_df.columns:
            if c not in ["date"]:
                new_col = group_by_date[c].mean()
                all_values[c] = new_col
        df = pd.DataFrame(all_values).reset_index()

        # add flight info
        df["flight"] = "15.6.2022"
        return df

    def load_data_to_df(self):
        mat = scipy.io.loadmat(
            fr'{dataframes_path}\Annontated1HzData_2016.mat')
        mdata = mat['Annontated1HzData']
        mdtype = mdata.dtype  # all columns names
        ndata = {n: mdata[n][0] for n in mdtype.names}  # dict of params, every param is array with length of tags
        columns = np.array(list(ndata.keys()))  # names of columns
        columns = np.delete(columns, np.argwhere(columns == "runningVerticalStraight"))  # delete param without data

        dfs = []
        for i, tag in tqdm(enumerate(ndata["tag"])):
            size = len(ndata["lat"][i])
            arr_for_df = [np.array([[tag[0][0]] for i in range(size)])]
            for c in columns:
                if c != "tag":
                    if len(ndata[c][i]) == size:
                        arr_for_df.append(ndata[c][i])
                    else:
                        arr_for_df.append(np.array([[None for i in range(size)]]))
            lens_set = {len(ndata[c][i]) for c in columns if
                        c != "tag"}  # check if all of the data is in the same length
            if len(lens_set) == 1:
                df = pd.DataFrame(np.concatenate(arr_for_df, axis=1), columns=columns)
                dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df["date"] = df.apply(lambda x: self.datenum_to_time(x['ptime'] + x['dtnum']), axis=1)
        df = df.rename(columns={"ele": "z"})

        return df

    def datenum_to_time(self, matlab_datenum):
        return datetime.fromordinal(int(matlab_datenum) - 366) + timedelta(days=matlab_datenum % 1)

    def add_info(self):
        # add the tag info to the df
        mat = scipy.io.loadmat(
            fr'{dataframes_path}\TagInfo.mat')
        mdata = mat["TagInfo"]
        mdtype = mdata.dtype  # all columns names
        ndata = {n: mdata[n][0] for n in mdtype.names}  # dict of params, every param is array with length of tags
        columns = np.array(list(ndata.keys()))  # names of columns
        tag_info = pd.DataFrame(ndata)

        # take the relevant data
        print("loading vultures tags information")
        tag_to_data = dict()
        for i in tag_info.index:
            for tag in tag_info["OrigID"][i][0]:
                tag_to_data[tag] = {param: tag_info[f"{param}"][i][0][0] for param in self.info_params}
                tag_to_data[tag]["Sex"] = tag_info["Sex"][i][0][0][0]

        for param in self.info_params:
            self.df[f"{param}"] = self.df.progress_apply(lambda x: tag_to_data[x["tag"]][param], axis=1)

    def filter_by_continuity(self, num_rows=20, num_seconds=1.1, filter_continues=False):
        """
        @param num_rows - the number of rows that needed for continuity
        @param num_seconds - the number of seconds that preserves continuity
        @param filter_continues:
        @return - new dataframe indexed by continuity
        """
        diffs = self.df['date'].diff()
        self.df["time_delta"] = diffs
        self.df['is_continuous'] = ((diffs < pd.Timedelta(num_seconds, unit='s')) & (diffs > pd.Timedelta(0, unit='s')))

        # every continues route will have the same number
        self.df["route_num"] = (~self.df["is_continuous"]).cumsum()

        if filter_continues:
            print(f"the total length is {len(self.df)} rows")
            self.df['filter_continuous'] = False
            count_rows = 0
            for index, row in tqdm(self.df.iterrows()):
                if row["is_continuous"]:
                    count_rows += 1
                else:
                    if count_rows > num_rows:
                        self.df.loc[(self.df.index >= index - 1 - count_rows) & (
                                self.df.index < index), 'filter_continuous'] = True
                    count_rows = 0

            self.df = self.df[self.df["filter_continuous"]]

    def latlon_to_xy(self, lat, lon, lat_rel, lon_rel):
        """
        return x,y from lat,lon relative to lat_rel,lon_rel
        """
        earth_circle = 40075
        dx = 1000 * (lon - lon_rel) * earth_circle * np.cos((lat + lat_rel) * np.pi / 360) / 360
        dy = 1000 * (lat - lat_rel) * earth_circle / 360
        return dx, dy

    def add_x_y(self):
        """
        @param df - a df with lon lat and route_num
        return the df with x and y relative to first location
        *** there is change of only 1 deg so the approximate will be okay ***
        can be improved if will do calculations for every route
        """
        x, y = self.latlon_to_xy(self.df["lat"], self.df["lon"], self.df.iloc[0]["lat"], self.df.iloc[0]["lon"])
        self.df["x"] = x
        self.df["y"] = y

    def calculate_derivatives(self):
        """
        calculting derivatives of each param and order and add it to df
        """
        self.df = calculate_derivatives_savgol(self.df, derivative_params=["x", "y", "z"], derivative_order=[0, 1, 2],
                                               num_samples=self.num_samples, poly_degree=self.poly_degree,
                                               group_by="route_num")

    def filter_standing_add_ClsSmthTrack(self):
        """
        doing some changes in the data, delete standings and add ClsSmthTrack
        in order for the data to be like the old, that the old functions can run
        :return:
        """
        self.df["velocity"] = np.sqrt(self.df["d1x"] ** 2 + self.df["d1y"] ** 2 + self.df["d1z"] ** 2)
        self.df = self.df[self.df["velocity"] > 2]
        self.filter_by_continuity(filter_continues=True)
        self.df = self.df.reset_index(drop=True)

        # add intersections for ClsSmthTrack
        intersect_by_group = self.df.groupby("route_num").progress_apply(
            lambda x: self.intersect_distance(np.array(x["x"]), np.array(x["y"])))
        intersections = np.concatenate([intersect_by_group.loc[index] for index in intersect_by_group.index])
        self.df["intersect_distance"] = intersections

        # add ClsSmthTrack like in the old data
        self.df["ClsSmthTrack"] = 3
        self.df.loc[
            (self.df.intersect_distance < self.intersect_distance_threshold) & (self.df.d1z > 0), "ClsSmthTrack"] = 2
        self.df.loc[self.df.d1z < 0, "ClsSmthTrack"] = 1

    def calculate_bank(self):
        """
        calculating bank angles from derviatives and locations
        """
        calculate_with_tan = True
        m = 8  # the mass of the vulture
        g = 9.81
        dot_product = self.df["d2x"] * (-self.df["d1y"]) + self.df["d2y"] * self.df["d1x"]
        norm = np.sqrt(self.df["d1x"] ** 2 + self.df["d1y"] ** 2)
        radial_acc = dot_product / norm
        if calculate_with_tan:
            bank_angle = np.arctan(radial_acc / g)
            self.df["bank_angle"] = np.rad2deg(bank_angle)
        else:
            sin_bank = radial_acc / g
            # bound the values for arcsin
            sin_bank[sin_bank > 1] = 1
            sin_bank[sin_bank < -1] = -1
            bank_angle = np.arcsin(sin_bank)
            self.df["bank_angle"] = np.rad2deg(bank_angle)

        # add bank diffs - one of the most complicated operations I made, for bank angle action
        # in every second it is the future difference that the vulture is going to do
        bank_diff = self.df.groupby("route_num").progress_apply(lambda x: pd.concat(
            [x["bank_angle"].diff().shift(periods=-1)[:-1], pd.Series({x.index[-1]: 0})])).reset_index(level=[0, 1])[0]
        self.df["bank_angle_action"] = bank_diff

    def add_thermal_classification(self):
        """
        add smoothed thermal classification by the first classification
        """
        # option for condition add
        # abs_avg_bank = df['bank_angle'].rolling(50, min_periods=1).mean().abs()
        thermal_bank_angle = 7
        if self.data_type == "human_walter":
            self.df["is_thermal_helper"] = (
                    (self.df['bank_angle'].rolling(50, min_periods=1).mean().abs() > thermal_bank_angle) &
                    (self.df['d0z'].rolling(50, min_periods=1).mean() > 0))
        else:
            self.df["is_thermal_helper"] = ((self.df["z"] != 0) & (self.df["ClsSmthTrack"] == 2)).astype(
                int)  # .rolling(15, min_periods=1).max()  #.astype(bool)
        self.df["is_thermal"] = self.df.groupby('route_num')["is_thermal_helper"].progress_apply(
            lambda x: x.rolling(self.thermal_number_of_samples_smooth, min_periods=1).max())
        del self.df["is_thermal_helper"]

    def add_velocity(self):
        """
        add the velocity calculated by savgol velocities
        :return:
        """
        self.df["velocity"] = (np.sqrt(self.df["d1x"] ** 2 + self.df["d1y"] ** 2 + self.df["d1z"] ** 2))

    def get_df(self):
        """
        function that returns the df for ML
        :return:
        """
        return self.df

    def get_thermal_df(self):
        """
        function that returns the thermal df for ML
        :return:
        """
        return self.thermal_df

    def save_thermal_df(self, num_seconds=1.1, num_rows=100):
        """
        return the thermal data that is continues and bigger than num_rows
        :param num_seconds:
        :param num_rows: - the number of rows that needed for continuity
        :return:
        """
        thermal_df = self.df[(self.df["z"] != 0) & (self.df["is_thermal"] == 1)].copy()  # delete values of z==0

        if self.data_type == "regular":
            # drop unnecessary columns
            thermal_df.drop(['speed', 'ptime', 'dtnum', 'TimeDiffSec',
                             'Azimuth', 'runningElevationProfile', 'runningVerticalSpeed',
                             'runningAngularSpeed', 'SegmentIntersect', 'runningHeadingVar',
                             'runningSpeed', 'AzDiffForwd', 'runningSegDispForCumForwd'], axis=1, inplace=True)

        # calculate continues thermal data
        diffs = thermal_df['date'].diff()
        thermal_df["new_time_delta"] = diffs
        thermal_df['is_continuous_thermal'] = (
                (diffs < pd.Timedelta(num_seconds, unit='s')) & (diffs > pd.Timedelta(0, unit='s')))

        # every continues route will have the same number
        thermal_df["thermal_num"] = (~thermal_df["is_continuous_thermal"]).cumsum()

        # leaves only continuity routes
        route_count = thermal_df.groupby("thermal_num")["thermal_num"].count()
        long_routes = route_count[route_count > num_rows].index
        continuity_thermal_df = thermal_df[thermal_df["thermal_num"].isin(long_routes)]

        # cut every first samples in every thermal num
        continuity_thermal_df = continuity_thermal_df.groupby("thermal_num").progress_apply(
            lambda x: x.iloc[self.thermal_number_of_samples_smooth - 5:-self.thermal_number_of_samples_smooth + 5,
                      :]).reset_index(level=[0, 1], drop=True)

        # save the df
        self.thermal_df = continuity_thermal_df

        # in order to take the right values for the direction take values at the begining of the route
        # and the end of the route
        begin_point_at_route = 10
        end_point_at_route = -10
        group_by_thermal_num = continuity_thermal_df.groupby("thermal_num")
        wind_direction_x = group_by_thermal_num["x"].progress_apply(
            lambda x: pd.Series([x.iloc[end_point_at_route] - x.iloc[begin_point_at_route]] * len(x), index=x.index))
        wind_direction_y = group_by_thermal_num["y"].progress_apply(
            lambda y: pd.Series([y.iloc[end_point_at_route] - y.iloc[begin_point_at_route]] * len(y), index=y.index))
        wind_direction_z = group_by_thermal_num["z"].progress_apply(
            lambda z: pd.Series([z.iloc[end_point_at_route] - z.iloc[begin_point_at_route]] * len(z), index=z.index))
        time_diff_route = group_by_thermal_num["date"].progress_apply(
            lambda t: pd.Series([(t.iloc[end_point_at_route] - t.iloc[begin_point_at_route]).total_seconds()] * len(t),
                                index=t.index))
        continuity_thermal_df["wind_direction_x"] = wind_direction_x
        continuity_thermal_df["wind_direction_y"] = wind_direction_y
        continuity_thermal_df["wind_direction_z"] = wind_direction_z
        continuity_thermal_df["time_diff_route"] = time_diff_route

        # save the approax wind velocity
        continuity_thermal_df["wind_velocity"] = (np.sqrt(continuity_thermal_df["wind_direction_x"] ** 2 +
                                                          continuity_thermal_df["wind_direction_y"] ** 2) /
                                                  continuity_thermal_df["time_diff_route"])

        # add the wind angle from the wind direction
        continuity_thermal_df["wind_angle"] = np.rad2deg(np.arctan2(continuity_thermal_df["wind_direction_y"],
                                                                    continuity_thermal_df["wind_direction_x"]))

        # add the angle of the height angle in order to see the diagonality of the thermal
        continuity_thermal_df["height_angle"] = np.rad2deg(np.arctan2(continuity_thermal_df["wind_direction_z"],
                                                                      np.sqrt(
                                                                          continuity_thermal_df[
                                                                              "wind_direction_x"] ** 2 +
                                                                          continuity_thermal_df[
                                                                              "wind_direction_y"] ** 2)))

        # calculate the side angle of the velocity
        continuity_thermal_df["side_angle"] = np.rad2deg(
            np.arctan2(continuity_thermal_df["d1y"], continuity_thermal_df["d1x"]))

        # add time back variables to adjust for simulation data
        vars_for_time_back = {"bank_angle": "bank_angle", "velocity": "velocity", "d1z": "vz",
                              "side_angle": "side_angle"}
        for var in vars_for_time_back:
            for timeback in range(10):
                continuity_thermal_df[f"info_{vars_for_time_back[var]}_timeback{timeback}"] = group_by_thermal_num[
                    f"{var}"].progress_apply(
                    lambda x: x.shift(periods=timeback))

        # add relative angle to wind
        continuity_thermal_df["relative_angle"] = (2 * (
                continuity_thermal_df["info_bank_angle_timeback0"] >= 0) - 1) * (
                                                          ((continuity_thermal_df["info_side_angle_timeback0"] -
                                                            continuity_thermal_df["wind_angle"]) % 360) - 180)

        remove_nans = True
        if remove_nans:
            # remove rows with Nans, mostly because of the time back variables
            relevant_columns = [c for c in continuity_thermal_df.columns if c not in self.info_params]
            continuity_thermal_df = continuity_thermal_df.dropna(axis=0, subset=relevant_columns)

        # save the data to pickle
        continuity_thermal_df.to_pickle(self.thermal_name)

        print("saved thermal df")

        return continuity_thermal_df

    def check_savgol(self):
        """
            function for checking which savitsky-golay params is better
            the function draws start_val-end_val graph after savgol and integration of derivatives
        """
        start_val = 800
        end_val = 1000
        savgol_num = 13
        savgol_poly = 6
        params = ["x"]  # , "y", "z"]
        new_df = pd.DataFrame()  # dataframe to calculate derivative and bank angle
        for param in params:
            t = np.linspace(1, end_val - start_val, end_val - start_val)
            x = self.df[start_val:end_val]["{0}".format(param)]
            new_df["d0x"] = savgol_filter(x, savgol_num, savgol_poly, mode="nearest", deriv=0)
            new_df["d1x"] = savgol_filter(x, savgol_num, savgol_poly, mode="nearest", deriv=1)
            new_df["d2x"] = savgol_filter(x, savgol_num, savgol_poly, mode="nearest", deriv=2)
            #     d0x = df[start_val:end_val]["d0{0}".format(param)]
            #     d1x = df[start_val:end_val]["d1{0}".format(param)]
            #     d2x = df[start_val:end_val]["d2{0}".format(param)]
            #  relevant graphs
            evald1x2 = it.cumtrapz(new_df["d2x"], t, initial=0) + new_df["d1x"][0]
            evalx2 = it.cumtrapz(evald1x2, t, initial=0) + x[start_val]
            evalx1 = it.cumtrapz(new_df["d1x"], t, initial=0) + x[start_val]
            plt.plot(t, new_df["d0x"], label="d0{0}".format(param), color="red")
            plt.plot(t, evalx1, label="d1{0}".format(param))
            plt.plot(t, evalx2, label="d2{0}".format(param))
            plt.scatter(t, x, label="real {0}".format(param), marker=".")
            plt.legend()
            plt.show()

            # plt.plot(t, evalx2-x, label="diff between d2 to real")
            # plt.plot(t, evald1x2, label="eval d1x")

            #         plt.plot(t, new_df["d1x"], label="d1{0}".format(param))
            #         plt.plot(t, new_df["d2x"], label="d2{0}".format(param))
            #         plt.title("values of {0} by var".format(param))
            #         plt.legend()
            #         plt.show()

    @staticmethod
    def plot_vulture_maps(df, zoom=13, map_type='satellite'):
        """
        plotting vultures route on satellite map, saves as HTML
        :param df: the df of points to show
        :param zoom:
        :param map_type:
        :return:
        """
        # some settings
        api_key = os.environ["GOOGLE_API_KEY"]
        bokeh_width, bokeh_height = 1600, 800
        lng = np.mean(df["lon"])
        lat = np.mean(df["lat"])
        gmap_options = GMapOptions(lat=lat, lng=lng, map_type=map_type, zoom=zoom)

        # data for hovering
        hover = HoverTool(
            tooltips=[
                ('z', '@z'),
                ('date', '@date{%F %H:%M:%S}'),
                ('is_thermal', '@is_thermal'),
                ('Vz', '@d1z')
            ],
            formatters={'@date': 'datetime'}
        )

        p = gmap(api_key, gmap_options, title='Vultures Data',
                 width=bokeh_width, height=bokeh_height,
                 tools=[hover, 'reset', 'wheel_zoom', 'pan'])

        # adding the data
        source = ColumnDataSource(df)

        # show(p)

        # add colors
        color_var = "z"
        mapper = linear_cmap(color_var, palette, df[color_var].min(), df[color_var].max())
        color_bar = ColorBar(color_mapper=mapper['transform'], location=(0, 0))
        p.add_layout(color_bar, 'right')

        # plotting
        center = p.circle('lon', 'lat', size=4, alpha=0.2, color=mapper, source=source)

        # path to save
        path = os.path.join(rf"{dataframes_path}",
                            "DataFrames/bokeh_map.html")
        save(p, path)

        print(f"successfully saved to: {path}")
        return p

    def intersect_distance(self, x, y, intersect_min_bound=5, intersect_max_bound=25, initial_value=300):
        """
        takes 2 lists and finds the minimum distance from another timestamp that can be intersected with
        it is useful to see if there is an intersection in each point

        the intersection can only be with timestamps close enough but not too close
        therefore the variables intersect_min_bound and intersect_max_bound are used
        :param x:
        :param y:
        :param intersect_min_bound:
        :param intersect_max_bound:
        :param initial_value:
        :return: distance to closest point
        """
        intersect_distance = np.full(len(x), initial_value)
        for i in range(intersect_max_bound, len(x) - intersect_max_bound):
            min_dist1 = np.min(np.sqrt((x[i - intersect_max_bound: i - intersect_min_bound] - x[i]) ** 2 + (
                    y[i - intersect_max_bound: i - intersect_min_bound] - y[i]) ** 2))
            min_dist2 = np.min(np.sqrt((x[i + intersect_min_bound: i + intersect_max_bound] - x[i]) ** 2 + (
                    y[i + intersect_min_bound: i + intersect_max_bound] - y[i]) ** 2))
            intersect_distance[i] = min(min_dist1, min_dist2)
        return intersect_distance


def calculate_derivatives_savgol(df, derivative_params=["x", "y", "z"], derivative_order=[0, 1, 2], num_samples=13,
                                 poly_degree=6, group_by="route_num"):
    """
    add derivatives to the df
    :param df:
    :param derivative_params:
    :param derivative_order:
    :param num_samples:
    :param poly_degree:
    :param group_by:
    :return:
    """
    for order in derivative_order:
        for param in derivative_params:
            print("calculate d{0} of {1}".format(order, param))
            # using Savitzky-Golay, need to check if need to change delta from 1 (currently know that sample is at 1HZ)
            # there is a problem with the derivative at the end and the beginning of the flight because of the "nearest" mode
            derivative_by_group = df.groupby(group_by).progress_apply(
                lambda x: savgol_filter(x[param], num_samples, poly_degree, mode="nearest", deriv=order))
            derivatives = np.concatenate([derivative_by_group.loc[index] for index in derivative_by_group.index])
            df["d{0}{1}".format(order, param)] = derivatives
    return df


def draw_route(df, x="x", y="y", z="z", hover_data=["date", 'd1z', 'bank_angle', 'd2x', 'd2y',
                                                    'd2z', 'd1x', 'd1y', 'velocity', 'date'], color="is_thermal",
               symbol="tag", title=""):
    """
    draw a route on a 3D plot of route
    :param title:
    :param df:
    :param x:
    :param y:
    :param z:
    :param hover_data:
    :param color:
    :param symbol:
    :return:
    """
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=color,
                        hover_data=hover_data, symbol=symbol,
                        color_continuous_scale=px.colors.sequential.Bluered, title=title)
    fig.update_layout(scene_aspectmode='data')  # ,scene_aspectratio=dict(x=1, y=1, z=1))
    return fig


def draw_glider_route(df, x="info_x_timeback0", y="info_y_timeback0", z="info_z_timeback0",
                      hover_data=["info_bank_angle_timeback0", "info_attack_angle_timeback0", "info_time_timeback0",
                                  "info_wingspan_timeback0", "info_vz_timeback0"], color="is_thermal",
                      symbol=None, title=""):
    """
    draw a route on a 3D plot of route for glider data in simulation
    :param df:
    :param x:
    :param y:
    :param z:
    :param hover_data:
    :param color:
    :param symbol:
    :return:
    """
    fig = draw_route(df, x=x, y=y, z=z, hover_data=hover_data, color=color, symbol=symbol, title=title)
    return fig


def add_thermal_classification_for_glider(df, num_samples_for_climb=10, vz_column="info_vz_timeback0",
                                          distance_column="real_distance", old_version=True):
    """
    add thermal classification to the df - good for gliding model not for vultures
    :param distance_column:
    :param old_version: if computed from climb rate
    :param vz_column:
    :param num_samples_for_climb:
    :param df:
    :return:
    """
    if old_version:
        if "route_num" in df.columns:
            df["is_thermal"] = (df.groupby('route_num')[vz_column].apply(
                lambda x: x.rolling(num_samples_for_climb).mean()) > 0)
        else:
            df["is_thermal"] = (df[vz_column].rolling(num_samples_for_climb).mean() > 0)
    else:
        df["is_thermal"] = (df[distance_column] < 80)
    return df


def add_distance_from_center_for_glider(df, center_column="info_fixed_thermal_center_0_timeback0",
                                        x_column="info_x_timeback0", y_column="info_y_timeback0"):
    """
    add distance from center by centers, x and y
    :param df:
    :param center_column:
    :param x_column:
    :param y_column:
    :return:
    """
    thermal_x_coords = df[center_column].map(lambda x: x[0])
    thermal_y_coords = df[center_column].map(lambda x: x[1])
    df["real_distance"] = np.sqrt((df[x_column] - thermal_x_coords) ** 2 + (
            df[y_column] - thermal_y_coords) ** 2)

    return df


if __name__ == "__main__":
    data = VulturesData(read_data=True, data_type="new")
    data.save_thermal_df()
    # print(pd.__version__)
    # data.save_thermal_df()
    # print(df.columns)
