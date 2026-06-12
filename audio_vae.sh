#!/bin/bash
# time python scripts/train.py --config ./configs/cls_0.gin --db_path ./runs/dataset/ --out_path ./runs/models/test1 --name default --channels 1 --workers 1 --batch 32 --max_steps 300

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --models /data/models/cls_5/cls_5_1M1_95.ts --output_dir ./result/latent_test/all --export_umap

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --models /data/models/cls_5/cls_5_1M1_95.ts --output_dir ./result/latent_test/train --export_umap --fold_role TRAIN

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --models /data/models/cls_5/cls_5_1M1_95.ts --output_dir ./result/latent_test/val --export_umap --fold_role VALIDATION

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --models /data/models/cls_5/cls_5_1M1_95.ts --output_dir ./result/latent_test/test --export_umap --fold_role TEST

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --models /data/models/cls_5/cls_5_1M1_95.ts --output_dir ./result/latent_test/val_samples --export_umap --fold_role VALIDATION --latent_mode samples

# python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --models /data/models/cls_5/* --output_dir ./result/cls_5/channel
# python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --models /data/models/cls_5/* --output_dir ./result/cls_5/ship

# python test/test_vae_export.py --models /data/models/cls_5/cls_5_best_9* --output_dir ./result/cls_5/test_audios/ /data/frags_ie_fix/


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


# python test/test_latent_separability.py --model /data/models/c0_95.ts --output_dir ./result/latent_separability/mobile --dynamic_selection MOBILE_ONLY
# python test/test_latent_separability.py --model /data/models/c0_95.ts --output_dir ./result/latent_separability/all --dynamic_selection ALL


# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-16 --base-channels 16

# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32 --base-channels 32

# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-64 --base-channels 64

# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm-128 --base-channels 128

# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-256 --base-channels 256

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-16/eval --ldm-checkpoint ./result/ldm/ldm-16/best.ckpt

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32/eval --ldm-checkpoint ./result/ldm/ldm-32/best.ckpt

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-64/eval --ldm-checkpoint ./result/ldm/ldm-64/best.ckpt

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-128/eval --ldm-checkpoint ./result/ldm/ldm-128/best.ckpt

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-256/eval --ldm-checkpoint ./result/ldm/ldm-256/best.ckpt




# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-huber --base-channels 32 --loss HUBER

# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-l1 --base-channels 32 --loss L1

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-huber/eval --ldm-checkpoint ./result/ldm/ldm-32-huber/best.ckpt

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-l1/eval --ldm-checkpoint ./result/ldm/ldm-32-l1/best.ckpt



# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-1b --base-channels 32 --num-res-blocks 1

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-1b/eval --ldm-checkpoint ./result/ldm/ldm-32-1b/best.ckpt

# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-3b --base-channels 32 --num-res-blocks 3

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-3b/eval --ldm-checkpoint ./result/ldm/ldm-32-3b/best.ckpt

# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-4b --base-channels 32 --num-res-blocks 4

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-4b/eval --ldm-checkpoint ./result/ldm/ldm-32-4b/best.ckpt

# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-r/12 --base-channels 32 --channel-ratios 1 2

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-r/12/eval --ldm-checkpoint ./result/ldm/ldm-32-r/12/best.ckpt

# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-r/1248 --base-channels 32 --channel-ratios 1 2 4 8

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-r/1248/eval --ldm-checkpoint ./result/ldm/ldm-32-r/1248/best.ckpt

# python test/test_ldm.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-r/421 --base-channels 32 --channel-ratios 4 2 1

# python test/test_ldm_eval.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/fix_cls_test/fix_cls_10_30.ts --ie-channel-selection REFERENCE_ONLY --output-dir ./result/ldm/ldm-32-r/421/eval --ldm-checkpoint ./result/ldm/ldm-32-r/421/best.ckpt

