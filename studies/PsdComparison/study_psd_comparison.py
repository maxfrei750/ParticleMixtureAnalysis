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
    visualize_particle_size_distribution_property_errors,
)


def compare_data_set_particle_size_distributions(
    data_roots_tested_methods: List[AnyPath],
    subsets_tested_methods: List[str],
    legend_labels_tested_methods: List[str],
    data_roots_reference: List[AnyPath],
    subsets_reference: List[str],
    output_root: AnyPath,
    legend_label_reference: str = "Manual reference",
    score_threshold: float = 0,
    equivalent_diameter: str = "feret_diameter_max",
):
    """Compares the particle size distributions (PSDs) of multiple data sets by:
    * plotting the PSDs class-wise
    * calculating characteristic PSD properties: geometric mean diameter, geometric standard
      deviation and number (optional with a threshold for the detection score).
    * calculating the errors of the PSD properties, using the first PSD as reference
    * plotting the relative PSD property errors and saving them as csv and tex tables

    :param data_roots_tested_methods: List of paths of the data set folders. All data sets must have the specified
        subset.
    :param subsets_tested_methods: List of names of subsets to use.
    :param legend_labels_tested_methods: Labels to use in the legends
    :param data_roots_reference: Data roots of the references.
    :param subsets_reference: Subsets of the references.
    :param output_root: Path, where plots and tables are saved to
    :param legend_label_reference: Label for the reference in the legends
    :param score_threshold: detections with a score below this threshold are discarded, optional
    :param equivalent_diameter: equivalent diameter to use for the study (feret_diameter_max or
        area_equivalent_diameter), optional
    """

    measurand_name = get_psd_x_label(equivalent_diameter)

    if len(data_roots_tested_methods) * len(subsets_tested_methods) != len(
        legend_labels_tested_methods
    ):
        raise ValueError(
            "Number of `data_roots_tested_methods` times number of `subsets_tested_methods` must "
            "be equal to number of `legend_labels_tested_methods`."
        )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    class_names_reference, data_sets_reference = get_data_sets(
        data_roots_reference, subsets_reference
    )

    class_names_tested_methods, data_sets_tested_methods = get_data_sets(
        data_roots_tested_methods, subsets_tested_methods
    )

    if sorted(class_names_tested_methods) != sorted(class_names_reference):
        raise ValueError("Tested methods and reference data use different classes.")

    class_names = class_names_tested_methods

    for class_name in class_names:

        if class_name == "background":
            continue

        legend_labels_reference = [
            f"Manual reference {i}" for i in range(1, len(data_sets_reference) + 1)
        ]

        particle_size_lists, score_lists, psd_properties_list = gather_psd_data_for_class(
            data_sets_reference,
            legend_labels_reference,
            class_name,
            equivalent_diameter=equivalent_diameter,
        )

        psd_properties_reference = plot_average_particle_size_distribution(
            particle_size_lists, label=legend_label_reference, linestyle="--"
        )

        particle_size_lists, score_lists, psd_properties_list = gather_psd_data_for_class(
            data_sets_tested_methods,
            legend_labels_tested_methods,
            class_name,
            score_threshold,
            equivalent_diameter=equivalent_diameter,
        )

        plot_particle_size_distributions(
            particle_size_lists=particle_size_lists,
            score_lists=score_lists,
            measurand_name=measurand_name,
            labels=legend_labels_tested_methods,
            kind="kde",
            do_display_number=False,
        )

        plt.xlim([0, 250])
        plt.ylim([0, 0.06])

        save_figure(output_root, f"psd_comparison_class_{class_name}")
        plt.close()

        visualize_particle_size_distribution_properties(
            [psd_properties_reference] + psd_properties_list, output_root, class_name
        )

        visualize_particle_size_distribution_property_errors(
            psd_properties_list, psd_properties_reference, output_root, class_name
        )

        plt.ylim([-0.3, 0.1])

        save_figure(output_root, f"psd_property_error_comparison_class_{class_name}")
        plt.close()


if __name__ == "__main__":
    compare_data_set_particle_size_distributions(
        data_roots_tested_methods=[
            "output/MultiClassSynthetic400SyntheticValidation400Detections",
            "output/HoughTransform",
        ],
        subsets_tested_methods=["validation_real_max"],
        legend_labels_tested_methods=[
            "Proposed method",
            "Hough transform",
        ],
        data_roots_reference=[
            "output/Manual",
        ],
        subsets_reference=[
            "validation_real_max",
            "validation_real_venator",
            "validation_real_kevin",
        ],
        output_root="studies/PsdComparison/results",
        equivalent_diameter="area_equivalent_diameter",
    )
