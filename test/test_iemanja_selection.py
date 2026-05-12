"""
Iemanja dataset combination inspector.
"""

import pandas as pd

import lps_utils.quantities as lps_qty

import lps_ml.datasets as ml_db
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv


def main():

    fs_out = lps_qty.Frequency.khz(16)
    duration = lps_qty.Time.s(1)
    overlap = lps_qty.Time.s(0)

    results = []

    for dynamic_selection in ml_db.DynamicSelection:

        for channel_selection in ml_db.ChannelSelection:

            dm = ml_db.Iemanja(
                file_processor=ml_procs.TimeProcessor(
                    fs_out=fs_out,
                    duration=duration,
                    overlap=overlap,
                ),

                cv=ml_cv.FiveByTwo(),

                dynamic_selection=dynamic_selection,
                channel_selection=channel_selection,

                batch_size=16
            )

            compile_df = dm.to_compile_df()

            print("\n")
            print("=" * 80)
            print(
                f"Dynamic: {dynamic_selection.name} | "
                f"Channel: {channel_selection.name}"
            )
            print("=" * 80)

            print(compile_df)

            results.append({
                "dynamic": dynamic_selection.name,
                "channel": channel_selection.name,
                "Qtd()": len(dm.to_df())
            })

    summary_df = pd.DataFrame(results)

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(summary_df)

    summary_df.to_csv(
        "./result/iemanja_selection_summary.csv",
        index=False
    )


if __name__ == "__main__":
    main()