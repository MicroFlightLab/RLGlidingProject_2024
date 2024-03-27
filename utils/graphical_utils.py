import plotly.express as px
import plotly.graph_objects as go
import random

from matplotlib import pyplot as plt
import numpy as np
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
import pickle
import pandas as pd

figures_list_for_dahboard_path = r"put_here_path"


def combine_graphs_from_dict(graph_kind, graphs_dict, df_for_all_graphs=None):
    """
    combine graphs from dictionary
    :param graph_kind: the function of the graph like px.line
    :param graphs_dict: {fig_name: figs_params_dict, ...}
    :param df_for_all_graphs: not None if same df for all graphs
    :return:
    """
    figs = dict()
    for graph_name in graphs_dict.keys():
        # determine the df for the graph from the values in the graphs_dict
        df_for_graph = df_for_all_graphs
        curr_graph_dict = graphs_dict[graph_name]
        if df_for_graph is None and "df_for_graph" in curr_graph_dict.keys():
            df_for_graph = curr_graph_dict["df_for_graph"]

        # create dict without df
        curr_graph_dict_without_df = curr_graph_dict.copy()
        if "df_for_graph" in curr_graph_dict_without_df.keys():
            del curr_graph_dict_without_df["df_for_graph"]

        if df_for_graph is None:
            figs[graph_name] = graph_kind(**curr_graph_dict_without_df)
        else:
            figs[graph_name] = graph_kind(df_for_graph, **curr_graph_dict_without_df)

    final_fig = combine_graphs(figs)
    return final_fig


def combine_graphs(figs):
    """
    combine plotly graphs into one graph, the graphs will be shown with same x,y axis
    :param figs: dictionary of figures, {name: plotly figure}
    :return:
    """
    all_data = []
    for tag_num in figs.keys():
        layout = figs[tag_num].layout
        all_data += figs[tag_num].data
    fig = go.Figure(data=all_data, layout=layout)

    for i, tag_num in enumerate(figs.keys()):
        fig["data"][i]["showlegend"] = True
        fig["data"][i]["name"] = str(tag_num)
        fig["data"][i]["line"]["color"] = px.colors.qualitative.Plotly[i]
        fig["data"][i]["marker"]["color"] = px.colors.qualitative.Plotly[i]

    return fig


def draw_multiple_graphs(df, x, y_list):
    """
    draw multiple graphs in one figure
    :param df:
    :param x:
    :param y_list:
    :return:
    """
    fig = go.Figure()
    for y in y_list:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='lines', name=y))

    return fig


def frame_args(duration):
    """
    set animation frame duration
    :param duration:
    :return:
    """
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"}
    }


def create_subplot_fig(figs_matrix, specs, same_color_for_clusters=True):
    """
    create subplot from list in plotly
    :param same_color_for_clusters: to make sure that same legends have the same color
    :param figs_matrix: the matrix of the figs, will be plotted this way in the final figure
    :param specs: the type of the subplot, for example [[{"type": "scatter3d"}, {"type": "bar"}]]
    :return:combined figure
    """
    combined_fig = make_subplots(
        rows=len(figs_matrix), cols=len(figs_matrix[0]),
        vertical_spacing=0.02,
        specs=specs
    )

    # add each trace (or traces) to its specific subplot
    for row, figs_row in enumerate(figs_matrix):
        for col, fig in enumerate(figs_row):
            if fig is not None:
                for i in fig.data:
                    # i["legendgroup"] = row * len(figs_row) + col
                    combined_fig.add_trace(i, row=row + 1, col=col + 1)

    combined_fig.update_layout(scene_aspectmode='data')

    if same_color_for_clusters:
        # change same clusters to same color
        color_legend_graphs = dict()
        # find all legends
        all_legends_groups = []
        for graph in combined_fig['data']:
            all_legends_groups.append(graph["legendgroup"])

        all_legends_groups = sorted(list(set(all_legends_groups)))
        # map each legend to color
        for i, legend_group in enumerate(all_legends_groups):
            color_legend_graphs[legend_group] = px.colors.qualitative.Dark24[i]

        for graph in combined_fig['data']:
            graph["marker"]["color"] = color_legend_graphs[graph["legendgroup"]]

    return combined_fig


def create_animation_from_figs(frame_figs):
    """
    create animation from given figs
    :param frame_figs: the figs to create animation from
    :return:
    """
    animation = go.Figure(frame_figs[-1])
    # animation.data = []
    animation.update_layout(legend_orientation="v")  # providing legend and colorbar from overlap

    frames = [go.Frame(data=frame_fig["data"], name=i) for i, frame_fig in enumerate(frame_figs)]

    animation.update(frames=frames)

    sliders = [
        {"pad": {"b": 10, "t": 60},
         "len": 0.9,
         "x": 0.1,
         "y": 0,

         "steps": [
             {"args": [[f.name], frame_args(10)],
              "label": f"{k : .3f}",
              "method": "animate",
              } for k, f in enumerate(animation.frames)
         ]
         }
    ]

    animation.update_layout(

        updatemenus=[{"buttons": [
            {
                "args": [None, frame_args(50)],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [[None], frame_args(0)],
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

    animation.update_layout(sliders=sliders)

    # animation_new = go.Figure(data=data, layout=animation["layout"], frames=frames)
    for frame in animation.frames:
        frame.layout.legend.orientation = 'v'

    # update the titles of the subplots
    for k in range(len(animation.frames)):
        animation.frames[k]['layout'].update(title_text=frame_figs[k]['layout']['title']['text'])

    return animation


def get_tab_dashboard_app():
    """
    dashboard from the figures in the pickle file [fig1, fig2, ...]
    :return:
    """
    app = Dash(__name__)
    # Create a list of figures
    with open(figures_list_for_dahboard_path, 'rb') as f:
        figures = pickle.load(f)

    # Create a list of tabs
    tabs = []
    for i in range(len(figures)):
        tabs.append(dcc.Tab(label=f'{figures[i]["layout"]["title"]["text"]}', value=f'figure{i + 1}',
                            children=[dcc.Graph(figure=figures[i])]))

    app.layout = html.Div([dcc.Tabs(id='tabs', value='figure1', children=tabs)])
    return app


def plot_3d_matplotlib(x, y, z, c, s=3, fig_path=None, show=False, save=True, elev=35, azim=-142, roll=0,
                       discrete_color=False, title="", axis=True, zlim=None, axis_type="equal", legend=True,
                       ylim=None, keep_aspect_ratio=False, color_by_cluster=None):
    """
    plot 3d scatter plot in matplotlib
    :param title:
    :param discrete_color:
    :param roll:
    :param azim:
    :param elev:
    :param fig_path:
    :param x:
    :param y:
    :param z:
    :param c:
    :param s:
    :param show:
    :param save:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if keep_aspect_ratio:
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.zaxis.set_major_locator(plt.MaxNLocator(2))
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # change ticks fontsize
        ax.tick_params(axis='both', which='major', labelsize=6)

    if discrete_color:
        if color_by_cluster is None:
            p = ax.scatter(x, y, z, c=c, s=s, cmap="tab10")
        else:
            for cluster in set(c):
                p = ax.scatter(x[c == cluster], y[c == cluster], z[c == cluster], c=color_by_cluster[str(cluster)], s=s)
        ax.plot(x, y, z, linewidth=0.5, color="black")
        if legend:
            legend1 = ax.legend(*p.legend_elements(),
                                loc="lower left", title="Clusters")
            ax.add_artist(legend1)
    else:
        p = ax.scatter(x, y, z, c=c, s=s, cmap="Wistia")
        # set color axis min and max to -2 and 6
        p.set_clim(-2, 6)
        fig.colorbar(p, ax=ax)

    if not axis:
        ax.set_axis_off()
        # set axis on with only x,y,z axis with no ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])

    if zlim is not None:
        ax.set_zlim(zlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_aspect(axis_type)
    ax.set_title(title)

    # make white background
    ax.set_facecolor((1.0, 1.0, 1.0))

    if show:
        plt.show()

    if save and fig_path is not None:
        # Save the plot to SVG format
        plt.savefig(fig_path, format='svg')
        print("saved")


if __name__ == '__main__':
    app = get_tab_dashboard_app()
    app.run_server(debug=True)
