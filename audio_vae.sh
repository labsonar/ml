#!/bin/bash

python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/no_wu --max_epochs 200
python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/wu --max_epochs 200 --kl_warmup_steps 100
python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/wu_cls --max_epochs 200 --cls_warmup_steps 100 --kl_warmup_steps 100
python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/wu_cls_01 --max_epochs 200 --cls_warmup_steps 100 --kl_warmup_steps 100 --cls_factor 0.1
