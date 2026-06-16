#!/bin/bash
# time python scripts/train.py --config ./configs/cls_0.gin --db_path ./runs/dataset/ --out_path ./runs/models/test1 --name default --channels 1 --workers 1 --batch 32 --max_steps 300

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --models /data/models/cls10_ch.ts --output_dir ./result/cls10_ch/train_ship --export_umap --fold_role TRAIN

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --models /data/models/cls10_ch.ts --output_dir ./result/cls10_ch/val_ship --export_umap --fold_role VALIDATION

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --models /data/models/cls10_ch.ts --output_dir ./result/cls10_ch/train --export_umap --fold_role TRAIN

python test/test_latent_separability.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --models /data/models/cls10_ch.ts --output_dir ./result/cls10_ch/val --export_umap --fold_role VALIDATION

python test/test_latent_extrapolation.py --ie-dataset-dir /data/iemanja_fixed/ --ie-latent-model /data/models/cls10_ch.ts --output_dir ./result/cls10_ch/val_ship_cross --fold_role VALIDATION --umap-model ./result/cls10_ch/train_ship/umap_cls10_ch.pkl

python test/test_latent_extrapolation.py --ie-dataset-dir /data/iemanja_fixed/ --ie-target "NAME_(US)" --ie-latent-model /data/models/cls10_ch.ts --output_dir ./result/cls10_ch/val_cross --fold_role VALIDATION --umap-model ./result/cls10_ch/train/umap_cls10_ch.pkl



# python scripts/train.py --config ./configs/cls10.gin --db_path ./runs/dataset_channel/ --out_path ./runs/models/cls10_ch --name cls10_ch --channels 1 --workers 1 --batch 32 --max_steps 25000 --save_every 25000 --gpu 0

