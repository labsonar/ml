#!/bin/bash

# python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/no_wu --max_epochs 200
# python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/wu --max_epochs 200 --kl_warmup_steps 100
# python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/wu_cls --max_epochs 200 --cls_warmup_steps 100 --kl_warmup_steps 100
# python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/wu_cls_01 --max_epochs 200 --cls_warmup_steps 100 --kl_warmup_steps 100 --cls_factor 0.1

# python test/test_ldm.py --model /data/models/v0_80.ts --ldm-steps 1000 --lr 0.00001 --output-dir ./result/ldm_test/v0_80
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 1000 --lr 0.00001 --output-dir ./result/ldm_test/v0_95
# python test/test_ldm.py --model /data/models/v0_99.ts --ldm-steps 1000 --lr 0.00001 --output-dir ./result/ldm_test/v0_99
# python test/test_ldm.py --model /data/models/v0_995.ts --ldm-steps 1000 --lr 0.00001 --output-dir ./result/ldm_test/v0_995
# python test/test_ldm.py --model /data/models/v0_999.ts --ldm-steps 1000 --lr 0.00001 --output-dir ./result/ldm_test/v0_999

# python test/test_audio_conv_vae.py --output_dir ./result/conv_vae/wu_cls_01 --max_epochs 2000 --cls_warmup_steps 500 --kl_warmup_steps 1000 --cls_factor 0.1

# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 50 --output-dir ./result/ldm_v0_95/50
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 100 --output-dir ./result/ldm_v0_95/100
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 200 --output-dir ./result/ldm_v0_95/200
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 300 --output-dir ./result/ldm_v0_95/300
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 400 --output-dir ./result/ldm_v0_95/400
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 500 --output-dir ./result/ldm_v0_95/500
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 600 --output-dir ./result/ldm_v0_95/600
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 700 --output-dir ./result/ldm_v0_95/700
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 800 --output-dir ./result/ldm_v0_95/800
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 900 --output-dir ./result/ldm_v0_95/900
# python test/test_ldm.py --model /data/models/v0_95.ts --ldm-steps 1000 --output-dir ./result/ldm_v0_95/1000

# python test/test_ldm.py --model /data/models/c0_95.ts --ldm-steps 300 --output-dir ./result/ldm_c0_95/300
# python test/test_ldm.py --model /data/models/c0_95.ts --ldm-steps 500 --output-dir ./result/ldm_c0_95/500

# # python test/test_ldm.py --model /data/models/c0_95.ts --ldm-steps 500 --output-dir ./result/ldm_c0_95/500_fixed --dynamic_selection FIXED_ONLY
# python test/test_ldm.py --model /data/models/c0_95.ts --ldm-steps 500 --output-dir ./result/ldm_c0_95/500_mobile --dynamic_selection MOBILE_ONLY
# python test/test_ldm.py --model /data/models/c0_95.ts --ldm-steps 500 --output-dir ./result/ldm_c0_95/500_all --dynamic_selection ALL


python test/test_latent_separability.py --model /data/models/c0_95.ts --output_dir ./result/latent_separability/mobile --dynamic_selection MOBILE_ONLY
python test/test_latent_separability.py --model /data/models/c0_95.ts --output_dir ./result/latent_separability/all --dynamic_selection ALL