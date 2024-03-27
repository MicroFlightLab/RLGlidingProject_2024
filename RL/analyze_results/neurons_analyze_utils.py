import pandas as pd
import numpy as np
import plotly.express as px
import wandb
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
from RL import training_utils
from RL import wandb_utils
from Data import data_utils
from captum.attr import IntegratedGradients, LayerActivation
import torch
from sklearn.cluster import KMeans
from kneed import KneeLocator
from utils import graphical_utils


def cluster_by_neurons(run_path, model_full_path_in_wandb, episodes=5, n_clusters=4, layers_to_conclude=[],
                       model_dict_and_df=None, title="", log=False, add_feature_importance=True,
                       add_correlations_dict=True, update_hyperparameters=dict()):
    """
    cluster time by neurons values
    :param add_correlations_dict:
    :param add_feature_importance: add feature importance by integrated gradients
    :param run_path:
    :param model_full_path_in_wandb:
    :param episodes:
    :param n_clusters: number of clusters in k-means, if list - choose the k by knee method on the list
    :param layers_to_conclude: layers to ignore
    :param model_dict_and_df: model_dict_and_df from wandb_utils
    :param title:
    :param log:
    :return: dict with figure, models_and_df_dict, neuron activation df
    """
    # get the model and the environment
    if model_dict_and_df is None:
        run_paths = run_path.split("/")
        model_dict_and_df = wandb_utils.get_df_and_model_dict_by_run_id(run_paths[1], run_paths[2],
                                                                        model_full_path_in_wandb=model_full_path_in_wandb,
                                                                        episodes=episodes, log=log,
                                                                        update_hyperparameters=update_hyperparameters)

    # get the states input for neurons analysis
    model_dict = model_dict_and_df["model_dict"]
    run_data_df = model_dict_and_df["df"]
    model = model_dict["model"]
    states_input = torch.stack(run_data_df["obs_tensor"].to_list())

    # get the neurons activation
    net = model.policy.actor
    inner_relu_layers = [module for module in net.modules() if isinstance(module, torch.nn.ReLU)]
    layer_act = LayerActivation(net, inner_relu_layers)
    attribution = layer_act.attribute(states_input)

    # create dataframe of activated neurons
    dict_of_activated_neurons = {}  # time: neurons after relu
    for i in range(len(attribution[0])):  # for each second
        lst_of_neurons = []
        for j in range(len(attribution)):  # for each layer
            if j not in layers_to_conclude:
                lst_of_neurons += attribution[j][i].tolist()
        dict_of_activated_neurons[i] = lst_of_neurons

    neurons_df = pd.DataFrame.from_dict(dict_of_activated_neurons)

    # cluster
    if type(n_clusters) == list:
        sse = []
        for i in n_clusters:
            kmeans = KMeans(n_clusters=i).fit(neurons_df.T)
            sse.append(kmeans.inertia_)

        n_clusters = KneeLocator(n_clusters, sse, curve="convex", direction="decreasing").elbow

    kmeans = KMeans(n_clusters=n_clusters).fit(neurons_df.T)
    run_data_df["kmeans_neurons_cluster"] = kmeans.labels_.astype(str)

    # add fig of the cluster
    fig = data_utils.draw_glider_route(run_data_df[run_data_df["route_num"] == 0], color="kmeans_neurons_cluster",
                                       title=title)

    # add feature importance by integrated gradients
    feature_importance = None
    if add_feature_importance:
        route_len = len(run_data_df[run_data_df["route_num"] == 0])
        ig = IntegratedGradients(model.policy.actor)
        attributions, delta = ig.attribute(states_input[:route_len], target=0, return_convergence_delta=True)
        feature_importance = {"ig": ig, "ig_attributions": attributions, "ig_delta": delta}

    correlations_dict = None
    if add_correlations_dict:
        correlations_dict = get_neurons_correlation_dict(neurons_df.T, run_data_df)

    ret_dict = {
        "fig": fig,
        "model_dict_and_df": model_dict_and_df,
        "neurons_df": neurons_df,
        "kmeans": kmeans,
        "net": net,
        "env": model_dict["env"],
        "states_input": states_input,
        "run_data_df": run_data_df,
        "feature_importance": feature_importance,
        "correlations_dict": correlations_dict,
        "n_clusters": n_clusters
    }

    return ret_dict


def get_frame_by_model_cluster_ret_dict(cluster_by_neurons_ret_dict, title="", pca=None, figs_to_conclude=[]):
    """
    create dashboard of couple subplots by cluster_by_neurons_ret_dict
    the graphs are:
    glider_3d_fig, importance_fig, state_dist, pca_fig, actions_fig
    :param figs_to_conclude:
    :param pca: if not None, use this pca
    :param cluster_by_neurons_ret_dict:
    :param title: the title of the dashboard
    :return:
    """
    run_data_df = cluster_by_neurons_ret_dict["run_data_df"]
    attributions = cluster_by_neurons_ret_dict["feature_importance"]["ig_attributions"]
    env = cluster_by_neurons_ret_dict["env"]
    activated_neurons_df = cluster_by_neurons_ret_dict["neurons_df"]
    glider_3d_fig, importance_fig, state_dist, pca_fig, actions_fig, add_polar_histogram_by_cluster = None, None, None, None, None, None

    # create df of attributions - attribution in each state for each time
    columns_by_feature = env.obs_to_param
    att_df = pd.DataFrame(np.abs(attributions.numpy()))
    att_df.rename(columns_by_feature, axis=1, inplace=True)
    params_names = list(set(["_".join(param.split("_")[:-1]) for param in columns_by_feature.values()]))
    timeback_name = "timeback"
    params_times = list(
        set(["_".join(param.split("_")[-1:])[len(timeback_name):] for param in columns_by_feature.values()]))

    # check the importance values of each feature - time and state
    for feature in params_names + params_times:
        feature_columns = [col for col in columns_by_feature.values() if feature in col]
        att_df[feature] = att_df[feature_columns].sum(axis=1)

    att_df["kmeans_neurons_cluster"] = run_data_df.kmeans_neurons_cluster

    feature_importance_by_cluster = True
    if feature_importance_by_cluster:
        att_df_by_cluster = att_df.groupby("kmeans_neurons_cluster")[params_times + params_names].mean()
        att_df_by_cluster = att_df_by_cluster.T.sort_index()
        importance_fig = px.bar(att_df_by_cluster, barmode='group')
        importance_fig.update_layout(
            title="Feature Importance by Integrated Gradients",
            xaxis_title="features",
            yaxis_title="sum")

    add_glider_route_by_cluster = True
    if add_glider_route_by_cluster:
        glider_3d_fig = data_utils.draw_glider_route(run_data_df[run_data_df["route_num"] == 0],
                                                     color="kmeans_neurons_cluster")

    add_states_distribution = True
    if add_states_distribution:
        data_col = [f"{param_name}_timeback0" for param_name in params_names]
        data_for_policy_network = run_data_df[data_col]
        state_dist = px.histogram(data_for_policy_network, nbins=500)

    add_pca_by_cluster = True
    if add_pca_by_cluster:
        if pca is None:
            pca = PCA(n_components=3)
            pca.fit(activated_neurons_df.T)
        df_of_pca_components = pd.DataFrame(pca.transform(activated_neurons_df.T))
        df_of_pca_components["cluster"] = run_data_df["kmeans_neurons_cluster"]
        pca_fig = px.scatter_3d(df_of_pca_components, x=0, y=1, z=2, color="cluster")
        pca_fig.update_traces(marker=dict(size=3))

    add_action_by_cluster = True
    if add_action_by_cluster:
        action_cols = [col for col in run_data_df.columns if "action_real" in col]
        action_cols_and_clusters = action_cols + ["kmeans_neurons_cluster"]
        cluster_df = run_data_df[action_cols_and_clusters].groupby("kmeans_neurons_cluster")[action_cols].agg(
            ["mean", "std"])
        cluster_df.columns = ["_".join(a) for a in cluster_df.columns.to_flat_index()]
        cluster_df["kmeans_neurons_cluster"] = cluster_df.index
        if len(action_cols) == 1:
            action_spec = "bar"
            actions_fig = px.bar(cluster_df, x="kmeans_neurons_cluster", y=f"{action_cols[0]}_mean",
                                 error_y=f"{action_cols[0]}_std")
        elif len(action_cols) == 2:
            action_spec = "scatter"
            actions_fig = px.scatter(cluster_df, x=f"{action_cols[0]}_mean", y=f"{action_cols[1]}_mean",
                                     color="kmeans_neurons_cluster",
                                     error_x=f"{action_cols[0]}_std", error_y=f"{action_cols[1]}_std")
        else:
            action_spec = "scatter3d"
            actions_fig = px.scatter_3d(cluster_df, x=f"{action_cols[0]}_mean", y=f"{action_cols[1]}_mean",
                                        z=f"{action_cols[2]}_mean",
                                        color="kmeans_neurons_cluster", error_x=f"{action_cols[0]}_std",
                                        error_y=f"{action_cols[1]}_std",
                                        error_z=f"{action_cols[2]}_std")

    add_polar_histogram_by_cluster = True
    if add_polar_histogram_by_cluster:
        polar_histogram_fig = training_utils.param_in_polar_by_angle(run_data_df, num_bins=20,
                                                                     angle_param="info_angle_from_wind_timeback0",
                                                                     color="kmeans_neurons_cluster",
                                                                     param_to_show="info_angle_from_wind_timeback0",
                                                                     func_name="count", is_bar=True)

    figs_dict = {
        "glider_3d_fig": {"fig": glider_3d_fig, "spec": {"type": "scatter3d"}},
        "importance_fig": {"fig": importance_fig, "spec": {"type": "bar"}},
        "state_dist": {"fig": state_dist, "spec": {"type": "histogram"}},
        "pca_fig": {"fig": pca_fig, "spec": {"type": "scatter3d"}},
        "actions_fig": {"fig": actions_fig, "spec": {"type": action_spec}},
        "polar_histogram_fig": {"fig": polar_histogram_fig, "spec": {"type": "barpolar"}}
    }

    figs_matrix = [[]]
    specs = [[]]
    for fig in figs_dict.keys():
        if fig not in figs_to_conclude:
            if len(figs_matrix[-1]) == 3:
                figs_matrix.append([])
                specs.append([])
            figs_matrix[-1].append(figs_dict[fig]["fig"])
            specs[-1].append(figs_dict[fig]["spec"])

    # in case there is more than one row
    if len(figs_matrix) != 1:
        figs_matrix[-1] += [None] * (3 - len(figs_matrix[-1]))
        specs[-1] += [None] * (3 - len(specs[-1]))

    combined_fig = graphical_utils.create_subplot_fig(figs_matrix, specs)
    combined_fig.update_layout(title_text=title)

    ret_dict = {
        "combined_fig": combined_fig,
        "pca": pca
    }
    return ret_dict


def get_neuron_clustering_dashboard(run_path, models_full_paths, cluster_results_dict={}, log=True,
                                    n_clusters=list(range(1, 10)), episodes=5, figs_to_conclude=[],
                                    use_same_pca_for_all=True, update_hyperparameters=dict(),):
    """
    get neuron clustering dashboard of a run and a process in the run
    :param update_hyperparameters: dict to update hyperparameters
    :param use_same_pca_for_all: if true use the pca of the last model for all
    :param figs_to_conclude:
    :param run_path:
    :param models_full_paths:
    :param cluster_results_dict:
    :param log:
    :param n_clusters:
    :param episodes:
    :return:
    """
    # get the models data
    for model_full_path in models_full_paths:
        model_dict_and_df = None
        if len(cluster_results_dict) is not None:
            if model_full_path in cluster_results_dict.keys():
                if "model_dict_and_df" in cluster_results_dict[model_full_path].keys():
                    model_dict_and_df = cluster_results_dict[model_full_path]["model_dict_and_df"]
        cluster_results_dict[model_full_path] = cluster_by_neurons(run_path, model_full_path,
                                                                   log=log,
                                                                   n_clusters=n_clusters,
                                                                   model_dict_and_df=model_dict_and_df,
                                                                   episodes=episodes,
                                                                   update_hyperparameters=update_hyperparameters)

    if log:
        print("finished getting runs cluster data")

    # get the frames for the animation
    frames = []
    blank_figure = None
    curr_pca = None  # use the pca from the last show
    for i, model_full_path in enumerate(models_full_paths[::-1]):
        ret_dict = get_frame_by_model_cluster_ret_dict(
            cluster_by_neurons_ret_dict=cluster_results_dict[model_full_path], title=model_full_path,
            figs_to_conclude=figs_to_conclude, pca=curr_pca)
        frame = ret_dict["combined_fig"]
        # use the pca from the end of learning
        if i == 0 and use_same_pca_for_all:
            curr_pca = ret_dict["pca"]
        frames.append(frame)

    animation = graphical_utils.create_animation_from_figs(frames[::-1])

    ret_dict = {
        "animation": animation,
        "cluster_results_dict": cluster_results_dict,
        "model_full_paths": models_full_paths,
        "run_path": run_path,
        "frames": frames
    }

    return ret_dict


def get_neurons_correlation_dict(transposed_neurons_df, run_data_df):
    """
    get dictionary of correlation dataframes between the neurons and between state and time from the neuron correlations
    :param transposed_neurons_df: (time x neurons) - activated neurons in each time
    :param run_data_df: (time x data) - df of actions, states and info from:
     training_utils.get_df_for_analysis_by_model_env
    :return:
    """
    # prepare the relevant cols to correlation
    # this is from training_utils.get_df_for_analysis_by_model_env definition - not so much generic might change this
    state_cols = [col for col in run_data_df.columns if ("_real" in col) and ("action_real" not in col)]
    action_cols = [col for col in run_data_df.columns if "action_real" in col]
    relevant_cols_for_correlation = state_cols + action_cols
    data_df_for_correlations = run_data_df[relevant_cols_for_correlation]

    # create df of correlations
    df_of_neurons_and_data = pd.concat([data_df_for_correlations, transposed_neurons_df], axis=1)
    neurons_corr_df = df_of_neurons_and_data.corr().fillna(0)  # dropna(how="all").dropna(how="all", axis=1)
    neurons_rows_state_action_cols_corr_df = neurons_corr_df[neurons_corr_df.index.isin(transposed_neurons_df.columns)][
        relevant_cols_for_correlation]

    # add the correlations by feature (time/state)
    # get the params for the features
    state_params_names = list(set(["_".join(param.split("_")[:-2]) for param in state_cols]))
    timeback_name = "timeback"
    state_params_times = sorted(list(
        set([name_part[len(timeback_name)] for param in state_cols for name_part in param.split("_") if
             timeback_name in name_part])))

    # sum the correlations for each feature
    state_features = state_params_names + state_params_times
    for feature in state_features:
        feature_columns = [col for col in state_cols if feature in col]
        neurons_rows_state_action_cols_corr_df[feature] = neurons_rows_state_action_cols_corr_df[
            feature_columns].abs().sum(axis=1)

    # find correlation for each time
    correlation_by_time = transposed_neurons_df.dot(neurons_rows_state_action_cols_corr_df)

    # normalize correlations in related columns
    correlation_cols_kinds = {"action_cols": action_cols, "state_params_names": state_params_names,
                              "state_params_times": state_params_times, "state_cols": state_cols}

    for cols_kind in correlation_cols_kinds.keys():
        correlation_by_time[correlation_cols_kinds[cols_kind]] = correlation_by_time[
            correlation_cols_kinds[cols_kind]].div(
            correlation_by_time[correlation_cols_kinds[cols_kind]].abs().sum(axis=1), axis=0)

    correlations_ret_dict = {
        "correlation_by_time": correlation_by_time,  # (state, time) correlation matrix
        "correlation_cols_kinds": correlation_cols_kinds,  # all columns names divided to logic lists
        "neurons_corr_df": neurons_corr_df  # all to all correlations
    }

    return correlations_ret_dict


def get_cluster_classification_by_run_data_df(runs_number_to_df, thermal_threshold=50):
    """
    classify modes to be search modes and thermal modes
    :param runs_number_to_df:
    :param thermal_threshold:
    :return:
    """
    dict_of_clusters_classification = {"run_number": [], "thermal_clusters": [], "search_clusters": [],
                                       "average_thermal_time": []}
    for run_number in tqdm(runs_number_to_df):
        run_data_df = runs_number_to_df[run_number]
        out_thermal = run_data_df[run_data_df["real_distance"] > thermal_threshold].groupby("kmeans_neurons_cluster")[
            "kmeans_neurons_cluster"].count()
        in_thermal = run_data_df[run_data_df["real_distance"] < thermal_threshold].groupby("kmeans_neurons_cluster")[
            "kmeans_neurons_cluster"].count()

        clusters = set(list(in_thermal.index) + list(out_thermal.index))
        for cluster in clusters:
            if cluster not in in_thermal.index:
                in_thermal[cluster] = 0
            if cluster not in out_thermal.index:
                out_thermal[cluster] = 0

        # classify between thermal and search clusters by count the times near thermal
        is_thermal_cluster = out_thermal / in_thermal < 0.5
        num_thermal_clusters = sum(is_thermal_cluster)
        num_search_clusters = len(is_thermal_cluster) - num_thermal_clusters

        # calculate the average thermal time
        average_thermal_time = sum(run_data_df["is_thermal"]) / len(run_data_df)

        dict_of_clusters_classification["run_number"].append(run_number)
        dict_of_clusters_classification["thermal_clusters"].append(num_thermal_clusters)
        dict_of_clusters_classification["search_clusters"].append(num_search_clusters)
        dict_of_clusters_classification["average_thermal_time"].append(average_thermal_time)

    cluster_classification_df = pd.DataFrame(dict_of_clusters_classification)
    cluster_classification_df["num_clusters"] = cluster_classification_df["thermal_clusters"] + \
                                                cluster_classification_df["search_clusters"]

    return cluster_classification_df
