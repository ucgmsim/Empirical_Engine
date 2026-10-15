"""Summarise value changes in the benchmark files between a base ref and the working tree.

Prints a markdown report, used by the benchmark-diff workflow as a PR comment.
Run from the repo root, e.g. `python tests/benchmark_data/compare_benchmarks.py origin/master`.
"""

import io
import subprocess
import sys

import numpy as np
import pandas as pd

DATA_DIR = "tests/benchmark_data/data"
# Same tolerance as assert_frame_equal in test_gmm_benchmark.py
RTOL, ATOL = 1e-5, 1e-8
MAX_COMMENT_CHARS = 60_000  # GitHub's limit is 65536


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def main(base: str) -> str:
    merge_base = git("merge-base", base, "HEAD").decode().strip()
    # Diff against the working tree, so uncommitted (tracked) changes are included
    name_status = git(
        "diff", "--name-status", "--no-renames", merge_base, "--", DATA_DIR
    )

    rows, other, n_unchanged = [], [], 0
    for line in name_status.decode().splitlines():
        status, path = line.split("\t")
        name = path.removeprefix(f"{DATA_DIR}/").removesuffix(".parquet")
        if status != "M":
            other.append(f"- `{name}`: {'added' if status == 'A' else 'deleted'}")
            continue

        old = pd.read_parquet(io.BytesIO(git("show", f"{merge_base}:{path}")))
        new = pd.read_parquet(path)
        if not (old.columns.equals(new.columns) and old.index.equals(new.index)):
            other.append(
                f"- `{name}`: columns/index changed ({old.shape} -> {new.shape})"
            )
            continue

        # Residuals assume ln means; linear IMs are never negative
        for df in (old, new):
            assert (df.filter(regex="_mean$") < 0).to_numpy().any(), (
                f"{name}: mean columns don't look like ln values (none negative)"
            )

        close = np.isclose(new, old, rtol=RTOL, atol=ATOL, equal_nan=True)
        changed_cols = old.columns[~close.all(axis=0)]
        if changed_cols.empty:
            n_unchanged += 1
            continue

        diff = old[changed_cols] - new[changed_cols]
        is_mean = changed_cols.str.endswith("_mean")
        # Signed ln residual (old - new) with the largest magnitude
        residuals = diff.loc[:, is_mean].to_numpy().ravel()
        max_residual = (
            f"{residuals[np.nanargmax(np.abs(residuals))]:+.3g}"
            if is_mean.any()
            else "-"
        )
        max_std_diff = (
            f"{diff.loc[:, ~is_mean].abs().max().max():.3g}"
            if (~is_mean).any()
            else "-"
        )
        rows.append(
            f"| `{name}` | {len(changed_cols)}/{len(old.columns)} "
            f"| {max_residual} | {max_std_diff} "
            f"| {(~close).any(axis=1).sum()}/{len(old)} |"
        )

    report = [
        "<!-- benchmark-diff -->",
        f"### Benchmark value changes vs `{base}`",
        "",
        (
            f"**{len(rows)}** modified files have value changes beyond the benchmark "
            f"test tolerance (rtol={RTOL:g}, atol={ATOL:g}). **{n_unchanged}** modified "
            "files have no value changes (only bytes differ, e.g. a different "
            "pandas/pyarrow writer)."
        ),
    ]
    if rows:
        report += [
            "",
            (
                "*Max ln residual*: the signed residual lnIM<sub>old</sub> − "
                "lnIM<sub>new</sub> with the largest magnitude, over the changed `_mean` "
                "columns. *Max \\|Δ\\| std*: the largest absolute difference in the "
                "changed standard deviation columns (ln units)."
            ),
            "",
            "| File | Changed columns | Max ln residual | Max \\|Δ\\| std | Changed rows |",
            "|---|---|---|---|---|",
            *rows,
        ]
    if other:
        report += ["", "**Other changes:**", *other]

    text = "\n".join(report)
    if len(text) > MAX_COMMENT_CHARS:
        text = text[:MAX_COMMENT_CHARS].rsplit("\n", 1)[0] + "\n\n*(truncated)*"
    return text


if __name__ == "__main__":
    print(main(sys.argv[1]))
