# /// script
# dependencies = [
#     "polars>=0.20.0",
#     "numpy>=1.26.0",
# ]
# ///
from pathlib import Path

import numpy as np
import polars as pl
from polars import DataFrame

FILE_PATH = "data/main_2026-02/_full_fhvhv_tripdata_2026-02.parquet"
TARGET_ROWS = 5_000_000


def sample_parquet(input_str: str, target: int) -> None:
    in_path = Path(input_str).resolve()
    out_path = in_path.with_name(f"cut_{in_path.name}")

    lf = pl.scan_parquet(in_path)

    total_rows: DataFrame = lf.select(pl.len()).collect().item()

    if total_rows <= target:
        lf.collect().write_parquet(out_path)
        return

    rng = np.random.default_rng()

    sampled_indices = rng.choice(total_rows, size=target, replace=False)
    sampled_indices.sort()

    df_cut = (
        lf.with_row_index("__row_idx__")
        .filter(pl.col("__row_idx__").is_in(sampled_indices))
        .drop("__row_idx__")
        .collect()
    )

    df_cut.write_parquet(out_path)


if __name__ == "__main__":
    sample_parquet(FILE_PATH, TARGET_ROWS)
