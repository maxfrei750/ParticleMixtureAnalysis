import fire

from paddle.custom_types import AnyPath
from paddle.inspection import inspect_dataset


def inspect_catalyst_dataset(subset: str, data_root: AnyPath = "data/complete"):
    inspect_dataset(
        subset=subset,
        data_root=data_root,
        do_display_box=False,
        do_display_label=False,
        do_display_score=False,
        score_threshold=0.5,
    )


if __name__ == "__main__":
    fire.Fire(inspect_catalyst_dataset)
