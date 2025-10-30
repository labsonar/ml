""" Generate .csv with qtds of cargo ship classifier in each DC of IARA
"""
import os
import pandas as pd

import lps_ml.datasets.iara as iara

def main():
    """Main function for the dataset info tables."""

    output_dir = "./result/cargo_ship"
    os.makedirs(output_dir, exist_ok=True)

    dcs = [
        iara.DC.A,
        iara.DC.B,
        iara.DC.C,
        iara.DC.D,
        iara.DC.OS,
        iara.DC.F,
        iara.DC.G,
        iara.DC.GLIDER,
    ]

    for classifier in [iara.CargoShipClassifier.IDENTIFIED,
                       iara.CargoShipClassifier.SIMILAR_EXCL_IDENTIFIED,
                       iara.CargoShipClassifier.GENERAL_EXCL_SIMILAR]:
        parts = []

        for dc in dcs:
            df = dc.to_df()

            selector = classifier.as_selector()
            df2 = selector.apply(df)
            part = df2.groupby(['Ship ID']).size().reset_index(name=str(dc))

            parts.append(part)

        merged = parts[0]
        for p in parts[1:]:
            merged = pd.merge(merged, p, on="Ship ID", how="outer")

        merged = merged.fillna(0).sort_values("Ship ID").reset_index(drop=True)

        print(f"\n########## {classifier.name} ##########")
        print(merged)

        merged.to_csv(os.path.join(output_dir,f"{classifier.name.lower()}.csv"))

if __name__ == "__main__":
    main()
