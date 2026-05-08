"""Entry point for dataset construction package."""

import argparse
import os

from . import (
    build_credit_dataset,
    build_medical_dataset,
    build_tabular_dataset,
    build_text_dataset,
    verify_credit_side_channels,
    verify_medical_side_channels,
    verify_side_channels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["tabular", "text", "medical"],
        choices=["tabular", "text", "medical", "credit"],
        help="Datasets to build",
    )
    parser.add_argument("--credit-source", help="Optional local CSV path for credit data")
    args = parser.parse_args()

    base = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    ml_data_root = os.path.join(base, "data", "ml")
    if "tabular" in args.datasets:
        out_dir = os.path.join(ml_data_root, "tabular")
        build_tabular_dataset(out_dir)
        verify_side_channels(out_dir)
    if "text" in args.datasets:
        out_dir = os.path.join(ml_data_root, "text")
        build_text_dataset(out_dir)
        verify_side_channels(out_dir)
    if "medical" in args.datasets:
        out_dir = os.path.join(ml_data_root, "medical")
        build_medical_dataset(out_dir)
        verify_medical_side_channels(out_dir)
    if "credit" in args.datasets:
        out_dir = os.path.join(ml_data_root, "credit")
        build_credit_dataset(out_dir, source_csv=args.credit_source)
        verify_credit_side_channels(out_dir)


if __name__ == "__main__":
    main()
