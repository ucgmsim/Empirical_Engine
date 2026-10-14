"""Summarise value changes in the benchmark files between a base ref and HEAD.

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
    name_status = git(
        "diff", "--name-status", "--no-renames", merge_base, "HEAD", "--", DATA_DIR
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

        close = np.isclose(new, old, rtol=RTOL, atol=ATOL, equal_nan=True)
        changed_cols = old.columns[~close.all(axis=0)]
        if changed_cols.empty:
            n_unchanged += 1
            continue

        abs_diff = (new[changed_cols] - old[changed_cols]).abs().max()
        is_mean = abs_diff.index.str.endswith("_mean")
        rows.append(
            f"| `{name}` | {len(changed_cols)}/{len(old.columns)} "
            f"| {abs_diff[is_mean].max() if is_mean.any() else 0:.3g} "
            f"| {abs_diff[~is_mean].max() if (~is_mean).any() else 0:.3g} "
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
            "| File | Changed columns | Max \\|Δ\\| mean | Max \\|Δ\\| std | Changed rows |",
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
