import plotly.graph_objs as go
from plotly.graph_objects import Layout
import numpy as np


class Plot3D:
    _layout: Layout = Layout(
        paper_bgcolor='rgba(0,0,0,1)',
        plot_bgcolor='rgba(0,0,0,1)',
        font_color="white",
        title_font_color="white",
        legend_title_font_color="white"
    )

    def __init__(self) -> None:
        """
        Initialize a 3D plotter.
        """
        self._fig: go.Figure = go.Figure(layout=self._layout)

    def points(self, points, color="dodgerblue", size=3) -> None:
        """
        Add points to the plot.

        :param points: Points to plot.
        :type points: list | tuple | np.ndarray

        :param color: The color to apply to the set of points (see plotly documentation for options). Defaults to ``"dodgerblue"``.
        :type color: str

        :param size: The size of each point marker. Defaults to ``3``.
        :type size: int
        """
        self._fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "size": size,
                    "color": color
                }
            )
        )

    def lines(self, lines, domain, step, colorscale="plasma", width=2) -> None:
        """
        Add lines to the plot.

        :param lines: Points to plot.
        :type lines: list | tuple | np.ndarray

        :param domain: The ``[x,y]`` domains over which to plot each line.
        :type domain: list[float | int]

        :param step: The step size for the line plotter.
        :type step: int | float

        :param colorscale: The plotly colorscale to apply to the set of lines (see plotly documentation for options). Defaults to ``"plasma"``.
        :type colorscale: str

        :param width: The width of each line. Defaults to ``2``.
        :type width: int
        """
        v: np.ndarray = np.arange(*domain, step)

        for [point, b] in lines:
            self._fig.add_trace(
                go.Scatter3d(
                    x=point[0] + v * b[0],
                    y=point[1] + v * b[1],
                    z=point[2] + v * b[2],
                    mode="lines",
                    line={
                        "width": width,
                        "colorscale": colorscale
                    }
                )
            )

    def save(self, path) -> None:
        self._fig.write_html(path, include_plotlyjs="cdn")

    def show(self) -> None:
        self._fig.show()
