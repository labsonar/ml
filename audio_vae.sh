#!/bin/bash

python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/no_wu --max_epochs 20
python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/wu --max_epochs 20 --kl_warmup_steps 10
python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/wu_cls --max_epochs 20 --cls_warmup_steps 10 --kl_warmup_steps 10
