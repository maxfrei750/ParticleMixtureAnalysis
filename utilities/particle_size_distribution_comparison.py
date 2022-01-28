from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from matplotlib.container import BarContainer
from scipy import stats

from measure_particle_size_distribution import CSV_FILE_NAME
from paddle.custom_types import AnyPath
from paddle.data import MaskRCNNDataset
from paddle.statistics import gmean, gstd
from paddle.visualization import get_viridis_colors

# TODO: Improve documentation.


def gather_psd_data_for_class(
    data_sets: List[MaskRCNNDataset],
    legend_labels: List[str],
    class_name: str,
    score_threshold: float = 0,
    equivalent_diameter: str = "feret_diameter_max",
):

    particle_size_lists = []
    score_lists = []
    psd_property_lists = []
    for data_set, legend_label in zip(data_sets, legend_labels):
        class_label = data_set.map_class_name_to_label[class_name]
        data_csv_path = data_set.subset_path / CSV_FILE_NAME
        data_csv_path = pd.read_csv(data_csv_path, dtype={"image_name": str})

        class_data = data_csv_path.loc[data_csv_path["label"] == class_label]

        particle_sizes = class_data[equivalent_diameter]
        scores = class_data["score"]

        particle_sizes, scores = apply_score_threshold(particle_sizes, scores, score_threshold)

        particle_size_lists.append(particle_sizes)
        score_lists.append(scores)

        geometric_mean = gmean(particle_sizes, weights=scores)
        geometric_standard_deviation = gstd(particle_sizes, weights=scores)
        number = len(particle_sizes)

        psd_property_lists.append(
            pd.Series(
                data={
                    "$d_g$": geometric_mean,
                    "$\sigma_g$": geometric_standard_deviation,
                    "$N$": number,
                },
                name=legend_label,
            )
        )
    return particle_size_lists, score_lists, psd_property_lists


def visualize_particle_size_distribution_property_errors(
    psd_properties_list: List[pd.Series],
    psd_properties_reference: pd.Series,
    output_root: AnyPath,
    class_name: str,
    do_display_bar_labels: bool = False,
):

    psd_properties = pd.DataFrame(psd_properties_list)

    reference_means = psd_properties_reference.loc[["$d_g$", "$\sigma_g$"]]

    uncertainty_geometric_diameter = (
        psd_properties_reference.loc["$\sigma_{d_g}$"] / psd_properties_reference.loc["$d_g$"]
    )

    uncertainty_geometric_standard_deviation = (
        psd_properties_reference.loc["$\sigma_{\sigma_g}$"]
        / psd_properties_reference.loc["$\sigma_g$"]
    )

    psd_properties_errors = psd_properties.div(reference_means) - 1

    psd_properties_errors_full = psd_properties_errors.copy()

    psd_properties_errors_full["$\sigma_{d_g}$"] = uncertainty_geometric_diameter
    psd_properties_errors_full["$\sigma_{\sigma_g}$"] = uncertainty_geometric_standard_deviation

    psd_properties_errors_full.to_csv(
        output_root / f"psd_property_error_comparison_class_{class_name}.csv"
    )

    num_methods = len(psd_properties_errors)
    num_colors = num_methods

    psd_properties_errors_full.to_latex(
        output_root / f"psd_property_error_comparison_class_{class_name}.tex",
        escape=False,
        float_format="%.4f",
    )
    colors = get_viridis_colors(num_colors)

    psd_properties_errors = psd_properties_errors.drop(columns="$N$").sort_index(
        ascending=False, axis=1
    )

    psd_properties_errors = psd_properties_errors.rename(
        columns={"$d_g$": "Geometric mean\ndiameter", "$\sigma_g$": "Geometric standard\ndeviation"}
    )

    fig = psd_properties_errors.T.plot.bar(
        rot=0,
        color=colors,
        yerr=[[uncertainty_geometric_diameter, uncertainty_geometric_standard_deviation]]
        * num_methods,
        capsize=2,
        error_kw=dict(capthick=1, linewidth=1),
    )
    fig.axhline(
        0,
        color="black",
        markersize=0,
        linewidth=fig.spines["left"].get_linewidth(),
    )
    fig.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    plt.ylabel("Relative error")

    if do_display_bar_labels:
        for container in fig.containers:
            if isinstance(container, BarContainer):
                labels = [f"{value:.0%}" for value in container.datavalues]
                fig.bar_label(container, labels, label_type="center")


def visualize_particle_size_distribution_properties(
    psd_property_lists: List[pd.Series], output_root: AnyPath, class_name: str
):
    psd_properties = pd.DataFrame(psd_property_lists)
    psd_properties.to_csv(output_root / f"psd_property_comparison_class_{class_name}.csv")
    psd_properties.to_latex(
        output_root / f"psd_property_comparison_class_{class_name}.tex",
        escape=False,
        float_format="%.2f",
    )
    return psd_properties


def apply_score_threshold(particle_sizes, scores, score_threshold):
    particle_sizes = particle_sizes[scores > score_threshold]
    scores = scores[scores > score_threshold]
    return particle_sizes, scores


def plot_average_particle_size_distribution(
    particle_size_lists: List,
    label: str = "Average",
    unit: str = "px",
    do_display_number: bool = False,
    do_display_geometric_mean: bool = True,
    do_display_geometric_standard_deviation: bool = True,
    **kwargs,
):

    geometric_mean_diameters = [
        gmean(particle_size_list) for particle_size_list in particle_size_lists
    ]
    geometric_mean_diameter_mean = np.mean(geometric_mean_diameters)
    geometric_mean_diameter_std = np.std(geometric_mean_diameters)

    geometric_standard_deviations = [
        gstd(particle_size_list) for particle_size_list in particle_size_lists
    ]
    geometric_standard_deviation_mean = np.mean(geometric_standard_deviations)
    geometric_standard_deviation_std = np.std(geometric_standard_deviations)

    numbers = [len(particle_size_list) for particle_size_list in particle_size_lists]
    number_mean = np.mean(numbers)
    number_std = np.std(numbers)

    label_base = label

    if do_display_number:
        label += f"\n  $N={number_mean:.0f}\pm{number_std:.0f}$"

    if do_display_geometric_mean:
        label += f"\n  $d_g={geometric_mean_diameter_mean:.1f}\pm{geometric_mean_diameter_std:.1f}$ {unit}"

    if do_display_geometric_standard_deviation:
        label += f"\n  $\sigma_g={geometric_standard_deviation_mean:.2f}\pm{geometric_standard_deviation_std:.2f}$"

    particle_sizes = np.concatenate(particle_size_lists)

    num_supports = 400
    support = np.linspace(1, max(particle_sizes) * 2, num_supports)
    support_log = np.log(support)

    # TODO: Apply log before KDE.
    probability_density_lists = [
        stats.gaussian_kde(np.log(particle_size_list))(support_log) / support
        for particle_size_list in particle_size_lists
    ]

    plot_data = pd.concat(
        [
            pd.DataFrame({"x": support, "y": probability_density_list})
            for probability_density_list in probability_density_lists
        ]
    )

    default_kwargs = {"color": "gray", "markersize": 0, "ci": "sd", "n_boot": 2000, "seed": 1}
    plot_kwargs = default_kwargs.copy()
    plot_kwargs.update(kwargs)

    sns.lineplot(data=plot_data, x="x", y="y", label=label, **plot_kwargs)

    psd_properties = pd.Series(
        data={
            "$d_g$": geometric_mean_diameter_mean,
            "$\sigma_g$": geometric_standard_deviation_mean,
            "$\sigma_{d_g}$": geometric_mean_diameter_std,
            "$\sigma_{\sigma_g}$": geometric_standard_deviation_std,
        },
        name=label_base,
    )

    return psd_properties


def get_data_sets(data_roots, subsets):
    data_roots = [Path(data_root) for data_root in data_roots]
    data_sets = [
        MaskRCNNDataset(
            data_root,
            subset=subset,
        )
        for data_root in data_roots
        for subset in subsets
    ]
    class_names = set(sum([data_set.class_names for data_set in data_sets], start=[]))
    return class_names, data_sets


def get_psd_x_label(equivalent_diameter):
    if equivalent_diameter == "feret_diameter_max":
        measurand_name = "Maximum Feret diameter"
    elif equivalent_diameter == "area_equivalent_diameter":
        measurand_name = "Area equivalent diameter"
    else:
        raise ValueError(f"Unknown equivalent diameter: {equivalent_diameter}")
    return measurand_name
