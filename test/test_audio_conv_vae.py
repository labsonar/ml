import os
import argparse
import torch
import matplotlib.pyplot as plt

import lightning
import lightning.pytorch.callbacks as lightning_call

import lps_utils.quantities as lps_qty
import lps_ml.utils.lightning as lps_light
import lps_ml.utils.general as ml_utils
import lps_ml.datasets as ml_db
import lps_ml.core.cv as ml_cv
import lps_ml.audio_processors as ml_procs
import lps_ml.model.audio_vae as lps_audio_vae
import lps_ml.datasets.selection as ml_sel

def _main():
    builder = ml_db.IemanjaBuilder(time_exclusive=True)

    parser = argparse.ArgumentParser(
        description="Test AudioFolder dataset"
    )
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--pqmf_bands", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--ratios",
                        type=int,
                        nargs='+',
                        default=[4, 4, 4, 2],
                        help="Lista de fatores de downsampling para o ruído"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_kl", type=float, default=0.1)
    parser.add_argument("--stft_factor", type=float, default=1)
    parser.add_argument("--mel_factor", type=float, default=0)
    parser.add_argument("--lofar_factor", type=float, default=0)
    parser.add_argument("--demon_factor", type=float, default=0)

    parser.add_argument("--resume_ckpt",
                        type=str,
                        default=None,
                        help="Checkpoint para retomar treinamento"
    )
    parser.add_argument("--pretrained_ckpt",
                        type=str,
                        default=None,
                        help="Checkpoint pré-treinado para fine-tuning"
    )

    parser.add_argument("--kl_warmup_steps", type=int, default=1)

    parser.add_argument("--cls_warmup_steps", type=int, default=1)
    parser.add_argument("--cls_factor", type=float, default=1e-3)

    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min_delta", type=float, default=0.001)

    parser.add_argument("--output_dir", type=str, default="./result/audio_conv_vae/test")

    builder.add_argparse_args(parser=parser)
    args = parser.parse_args()

    if args.resume_ckpt and args.pretrained_ckpt:
        raise ValueError("Use apenas um: --resume_ckpt OU --pretrained_ckpt")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    dm = builder.from_argparse_args(args)

    if args.pretrained_ckpt is not None:
        model = lps_audio_vae.CONV_VAE.load_from_checkpoint(
            args.pretrained_ckpt,
            beta_kl=args.beta_kl,
            stft_factor=args.stft_factor,
            mel_factor=args.mel_factor,
            lofar_factor=args.lofar_factor,
            demon_factor=args.demon_factor,
            lr=args.lr,
            kl_warmup_steps=args.kl_warmup_steps,
            cls_warmup_steps=args.cls_warmup_steps,
            cls_factor=args.cls_factor,
            n_classes=dm.get_n_targets(),
        )
    else:
        model = lps_audio_vae.CONV_VAE(
            n_bands=args.pqmf_bands,
            capacity=args.capacity,
            latent_dim=args.latent_dim,
            ratios=args.ratios,

            beta_kl=args.beta_kl,
            stft_factor=args.stft_factor,
            mel_factor=args.mel_factor,
            lofar_factor=args.lofar_factor,
            demon_factor=args.demon_factor,
            lr=args.lr,
            kl_warmup_steps=args.kl_warmup_steps,
            cls_warmup_steps=args.cls_warmup_steps,
            cls_factor=args.cls_factor,
            n_classes=dm.get_n_targets(),
        )

    trainer = lps_light.default_trainer(
            output_dir = output_dir,
            max_epochs = args.max_epochs,
            check_val_every_n_epoch = args.check_val_every_n_epoch,
            patience = args.patience,
            min_delta = args.min_delta,
        )

    dm.num_workers = 0
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt)
    print("Treino concluído. Gerando reconstruções finais...")


if __name__ == "__main__":
    _main()
