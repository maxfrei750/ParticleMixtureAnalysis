from pathlib import Path
from typing import List

from matplotlib import pyplot as plt

from paddle.custom_types import AnyPath
from paddle.visualization import plot_particle_size_distributions, save_figure
from utilities.particle_size_distribution_comparison import (
    gather_psd_data_for_class,
    get_data_sets,
    get_psd_x_label,
    plot_average_particle_size_distribution,
    visualize_particle_size_distribution_properties,
)


def compare_manual_psd_measurements(
    data_roots: List[AnyPath],
    subsets: List[str],
    output_root: AnyPath,
    equivalent_diameter: str = "feret_diameter_max",
):
    """Compares the particle size distributions (PSDs) of manual measurements by:
    * plotting the PSDs class-wise
    * calculating characteristic PSD properties: geometric mean diameter, geometric standard
      deviation and number (optional with a threshold for the detection score).
    * calculating the "average" PSD and adding it to the plot.

    :param data_roots: List of paths of the data set folders. All data sets must have the specified
        subset.
    :param subsets: List of names of subsets_tested_methods to use.
    :param output_root: Path, where plots and tables are saved to
    :param equivalent_diameter: equivalent diameter to use for the study (feret_diameter_max or
        area_equivalent_diameter), optional
    """

    measurand_name = get_psd_x_label(equivalent_diameter)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    class_names, data_sets = get_data_sets(data_roots, subsets)

    legend_labels = [f"Manual reference {i}" for i in range(1, len(data_sets) + 1)]

    for class_name in class_names:

        if class_name == "background":
            continue

        particle_size_lists, score_lists, psd_properties_list = gather_psd_data_for_class(
            data_sets,
            legend_labels,
            class_name,
            equivalent_diameter=equivalent_diameter,
        )

        plot_particle_size_distributions(
            particle_size_lists=particle_size_lists,
            score_lists=score_lists,
            measurand_name=measurand_name,
            labels=legend_labels,
            kind="kde",
            linewidth=1,
            linestyle="--",
            do_display_number=False,
            do_display_geometric_mean=False,
            do_display_geometric_standard_deviation=False,
        )

        average_psd_properties = plot_average_particle_size_distribution(
            particle_size_lists,
            do_display_geometric_mean=False,
            do_display_geometric_standard_deviation=False,
        )

        plt.ylim([0, 0.02])
        plt.xlim([0, 250])

        save_figure(output_root, f"psd_comparison_class_{class_name}")
        plt.close()

        psd_properties_list = [average_psd_properties] + psd_properties_list

        _ = visualize_particle_size_distribution_properties(
            psd_properties_list, output_root, class_name
        )


if __name__ == "__main__":
    compare_manual_psd_measurements(
        data_roots=[
            "output/Manual",
        ],
        subsets=["validation_real_max", "validation_real_kevin", "validation_real_venator"],
        output_root="studies/PsdComparisonManual/results",
        equivalent_diameter="area_equivalent_diameter",
    )
