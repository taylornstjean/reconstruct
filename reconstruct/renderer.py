import plotly.graph_objs as go
from plotly.graph_objects import Layout
import numpy as np


layout = Layout(
    paper_bgcolor='rgba(0,0,0,1)',
    plot_bgcolor='rgba(0,0,0,1)',
    font_color="white",
    title_font_color="white",
    legend_title_font_color="white"
)


def plot(points):

    """
    Simple plotter to display points.

    :param points: Points to plot.
    :type points: list | tuple | np.ndarray
    """

    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker={
                "size": 3,
                "color": "dodgerblue"
            }
        )
    )

    fig.write_html("plot_data.html", include_plotlyjs="cdn")

    fig.show()


def plot_transform(lines, points, prime_range, dl):
    """
    Simple plotter to display detected lines alongside the input point cloud.

    :param lines: Lines to plot (detected lines) with point cloud.
    :type lines: list | tuple | np.ndarray

    :param points: The points in the point cloud.
    :type points: list | tuple | np.ndarray
    """

    v = np.arange(*prime_range, dl)

    fig = go.Figure(layout=layout)
    for [point, b, _, _] in lines:
        fig.add_trace(
            go.Scatter3d(
                x=point[0] + v * b[0],
                y=point[1] + v * b[1],
                z=point[2] + v * b[2],
                mode="lines",
                line={
                    "width": 2,
                    "colorscale": "plasma"
                }
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker={
                "size": 3,
                "color": "black"
            }
        )
    )

    fig.write_html("transform.html", include_plotlyjs="cdn")

    fig.show()
