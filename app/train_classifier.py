"""
Simple classifier over iemanja dataset
"""
import os
import argparse

import torch
import lps_ml.model as ml_model
import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.utils.default as ml_default
import lps_ml.utils.metrics as ml_metrics

def _main():
    """Main function for the dataset info tables."""

    parser = argparse.ArgumentParser(description="Train an MLP classifier on iara.")
    parser.add_argument("--only-info", action="store_true", help="Only print database info")
    parser.add_argument("--paired", action="store_true", help="Only print database info")

    builder = ml_db.IemanjaBuilder()
    builder.add_argparse_args(parser=parser)
    ml_default.add_model_args(parser, models=["mlp", "cnn1d", "cnn2d"])
    ml_default.add_training_args(parser, default_output_dir="./result/iemanja_classifier")

    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    if args.paired:
        dm = builder.paired_from_argparse_args(args)
    else:
        dm = builder.from_argparse_args(args)

    print(ml_utils.format_header(60,"Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Training"))
    print()
    print(dm.to_df())

    if args.only_info:
        dm.setup()
        dl = dm.train_dataloader()
        for x, y in dl:
            if isinstance(x, list):
                print("x: ", len(x))
                for i in x:
                    print("\ti: ", i.shape)
            else:
                print("x: ", x.shape)

            print("y: ", y.shape)
            print(y)
            break

    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        trainer, ckpt = ml_default.trainer_from_args(args)


        if os.path.exists(ckpt.get_best()):
            dm.setup()

        else:

            model = ml_default.model_from_args(args, dm)

            trainer.fit(model, dm)
            trainer.test(model, datamodule=dm)

        if args.model == "mlp":
            best_model = ml_model.MLP.load_from_checkpoint(ckpt.get_best())
        elif args.model == "cnn1d":
            best_model = ml_model.CNN1D.load_from_checkpoint(ckpt.get_best())
        else:
            best_model = ml_model.CNN2D.load_from_checkpoint(ckpt.get_best())

        loaders = {
            "train": dm.train_dataloader(),
            "val": dm.val_dataloader(),
            "test": dm.test_dataloader(),
        }

        with ml_utils.evaluating(best_model):
            metrics_df = ml_metrics.evaluate_splits(best_model, loaders)

        metrics_path = os.path.join(output_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path)

        print(ml_utils.format_header(60))
        print()
        print(ml_utils.format_header(60,"Results"))
        print(metrics_df)
        print(ml_utils.format_header(60))

if __name__ == "__main__":
    _main()
