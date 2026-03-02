#!/usr/bin/env python3
"""
Test script for AudioFolder dataset.

Usage:
    python test_audiofolder.py /path/to/dataset
"""

import os
import sys
import argparse

import lps_utils.quantities as lps_qty
import lps_ml.datasets as ml_db
import lps_ml.audio_processors as ml_procs
import lps_ml.datasets.selection as ml_sel

def _main():
    parser = argparse.ArgumentParser(
        description="Test AudioFolder dataset"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Root directory containing class subfolders"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: directory {args.input_dir} does not exist.")
        sys.exit(1)

    print(f"\n[INFO] Building AudioFolder from: {args.input_dir}\n")

    fs=lps_qty.Frequency.khz(16)
    duration=lps_qty.Time.s(1)
    overlap=lps_qty.Time.s(0)

    dm = ml_db.AudioFolder(
        file_processor=ml_procs.TimeProcessor(
                fs_out=fs,
                duration=duration,
                overlap=overlap,
                pipelines=[ml_procs.ToFloatConverter()]
            ),
        input_dir=args.input_dir,
        selection = ml_sel.Selector(ml_sel.LabelTarget(column="Class", values= ["cargo"])),
        batch_size=4
    )

    dm.prepare_data()
    dm.setup()

    print("\n========== to_df() ==========\n")
    df = dm.to_df()
    print(df)

    print("\n========== to_compile_df() ==========\n")
    compile_df = dm.to_compile_df()
    print(compile_df)

    print("\n[INFO] Number of targets:", dm.get_n_targets())

    print("\n[INFO] Sample shape:", dm.get_sample_shape())


if __name__ == "__main__":
    _main()
