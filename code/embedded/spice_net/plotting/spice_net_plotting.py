from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt

from ..spice_net_hcm import SpiceNetHcm
from ..spice_net_som import SpiceNetSom


def plot_som(som: SpiceNetSom, select_all_button: bool = False):
    stem_x_values = som.get_as_matrix()[:, 0]
    x_min = stem_x_values.min()
    x_max = stem_x_values.max()
    distance = x_max - x_min
    x_min -= distance * 0.15
    x_max += distance * 0.15
    x_plt_vals = np.linspace(x_min, x_max, 500)
    activation_values = som.calculate_activation_values(x_plt_vals.tolist())
    df = pd.DataFrame(activation_values[:, 2:].transpose(), index=x_plt_vals)
    fig = px.line(df)
    if select_all_button:
        fig.update_layout(dict(updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=["visible", "legendonly"],
                        label="Deselect All",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", True],
                        label="Select All",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=False,
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            ),
        ]
        ))
    fig.show(title='Activation values')


def plot_hcm(hcm: SpiceNetHcm, normed: bool = False):
    matrix = hcm.get_matrix() if normed is False else np.where(hcm.get_matrix() > 0, 1, -1)
    fig = px.imshow(matrix, text_auto=False, aspect="auto")
    fig['layout']['yaxis']['autorange'] = "min"
    fig.update_layout(yaxis=dict(scaleanchor='x'))
    fig.show()


def plot_som_weights_stem(som: SpiceNetSom):
    stem_x_values = som.get_as_matrix()[:, 0]
    stem_y_values = som.get_as_matrix()[:, 1]

    plt.stem(stem_x_values, stem_y_values)
    plt.show()


def plot_som_comparison_to_data(som: SpiceNetSom, data: list, n_bins: Optional[int] = None):
    bins = n_bins if n_bins is not None else len(som)
    stem_x_values = som.get_as_matrix()[:, 0]
    stem_y_values = som.get_as_matrix()[:, 1]

    fig, axs = plt.subplots(2, 2, sharey=False, tight_layout=True)
    fig.set_size_inches(18.5, 21.0, forward=True)

    axs[0, 0].stem(stem_x_values, stem_y_values)

    # first row of plots
    xmax = axs[0, 0].viewLim.xmax
    xmin = axs[0, 0].viewLim.xmin

    # second row of plots
    x_plt_vals = np.linspace(xmin, xmax, 500)
    activation_values = som.calculate_activation_values(x_plt_vals)
    num_rows, num_cols = activation_values.shape
    for i in range(num_rows):
        axs[1, 0].plot(x_plt_vals, activation_values[i, 2:], linewidth=2)
    axs[0, 1].hist(data, bins=bins)
    axs[1, 1].hist(data, bins=bins)
    plt.show()
