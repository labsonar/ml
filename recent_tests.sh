#!/bin/bash
# time python scripts/train.py --config ./configs/cls_0.gin --db_path ./runs/dataset/ --out_path ./runs/models/test1 --name default --channels 1 --workers 1 --batch 32 --max_steps 300

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --models /data/models/cls2_ch_no_demon.ts --output_dir ./result/cls2_ch_no_demon/train_ship --export_umap --fold_role TRAIN

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --models /data/models/cls2_ch_no_demon.ts --output_dir ./result/cls2_ch_no_demon/val_ship --export_umap --fold_role VALIDATION

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --models /data/models/cls2_ch_no_demon.ts --output_dir ./result/cls2_ch_no_demon/train --export_umap --fold_role TRAIN

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --models /data/models/cls2_ch_no_demon.ts --output_dir ./result/cls2_ch_no_demon/val --export_umap --fold_role VALIDATION

python test/test_latent_extrapolation.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/cls2_ch_no_demon.ts --output_dir ./result/cls2_ch_no_demon/val_ship_cross --fold_role VALIDATION --umap-model ./result/cls2_ch_no_demon/train_ship/umap_cls2_ch_no_demon.pkl

python test/test_latent_extrapolation.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --ie-latent-model /data/models/cls2_ch_no_demon.ts --output_dir ./result/cls2_ch_no_demon/val_cross --fold_role VALIDATION --umap-model ./result/cls2_ch_no_demon/train/umap_cls2_ch_no_demon.pkl



# python scripts/train.py --config ./configs/cls10.gin --db_path ./runs/dataset_channel/ --out_path ./runs/models/cls2_ch_no_demon --name cls2_ch_no_demon --channels 1 --workers 1 --batch 32 --max_steps 25000 --save_every 25000 --gpu 0


python test/test_ldm.py --ie-target "NAME_(US)" --ie-latent-model /data/models/cls_5_5M_95.ts --output-dir ./result/ldm_test/cls5/0-1 --early-stopping-patience 10

python test/test_ldm_eval.py --ie-target "NAME_(US)" --ie-latent-model /data/models/cls_5_5M_95.ts --output-dir ./result/ldm_test/cls5/0-1 --ldm-checkpoint ./result/ldm_test/cls5/0-1/best-v1.ckpt