import os
import argparse
import collections
import pandas as pd
import torch

import lps_ml.utils.device as ml_device
import lps_ml.utils.sonar_loss as ml_loss
import lps_ml.model.audio_vae as lps_audio_vae
import lps_ml.datasets as ml_db
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
import lps_utils.quantities as lps_qty


def eval_loader(loader, model, sonar_loss, device):

    acc = collections.defaultdict(list)

    for x, _ in loader:

        x = x.to(device)

        with torch.no_grad():
            y = model(x)

        loss_dict = sonar_loss.compute_all_losses(x, y)

        for key, values in loss_dict.items():
            for i, v in enumerate(values):
                acc[f"{key}[{i}]"].append(v.item())

    result = {}
    for k, v in acc.items():
        result[k] = sum(v) / len(v)

    for group in ["stft_loss", "mel_loss", "lofar_loss", "demon_loss"]:
        keys = [k for k in result if k.startswith(group)]
        if keys:
            result[f"mean_{group}"] = sum(result[k] for k in keys) / len(keys)

    return result

def _main():

    parser = argparse.ArgumentParser(description="Compare múltiplos modelos CONV_VAE")

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Lista de checkpoints (.ckpt)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Diretório com áudios (mesmo formato do AudioFolder)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./result/model_comparison.csv"
    )

    args = parser.parse_args()
    device = ml_device.get_available_device()


    fs = lps_qty.Frequency.khz(16)
    n_samples = int(2**17)
    overlap = int(2**16)

    dm = ml_db.AudioFolder(
        file_processor=ml_procs.SampleProcessor(
            fs_out=fs,
            n_samples=n_samples,
            overlap=overlap,
            pipelines=[ml_procs.ToFloatConverter()]
        ),
        cv=ml_cv.SimpleSplitCV(),
        input_dir=args.input_dir,
        batch_size=16,
        num_workers=1
    )

    sonar_loss = ml_loss.SonarLoss()

    dm.setup()
    trn_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    # test_loader = dm.test_dataloader()

    rows = []

    for model_path in args.models:

        model = lps_audio_vae.CONV_VAE.load_from_checkpoint(model_path)
        model.to(device)
        model.eval()

        parent = os.path.basename(os.path.dirname(model_path))
        name = os.path.splitext(os.path.basename(model_path))[0]

        row = {
            "model": f"{parent}/{name}"
        }

        trn_res = eval_loader(trn_loader, model, sonar_loss, device)
        for k, v in trn_res.items():
            row[f"train/{k}"] = v

        val_res = eval_loader(val_loader, model, sonar_loss, device)
        for k, v in val_res.items():
            row[f"val/{k}"] = v

        # test_res = eval_loader(test_loader, model, sonar_loss, device)
        # for k, v in test_res.items():
        #     row[f"test/{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("model")

    cols = sorted([c for c in df.columns if "mean" not in c]) + \
           sorted([c for c in df.columns if "mean" in c])

    df = df[cols]

    print("\n Resultado:")
    print(df)

    df.to_csv(args.output_csv)
    print(f"\n Salvo em: {args.output_csv}")

if __name__ == "__main__":
    _main()
