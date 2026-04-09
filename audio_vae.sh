#!/bin/bash

# python test/test_audio_conv_vae.py --output_dir ./result/iemanja/cap64_lat128 --ratios 8 8 8 4 --capacity 64 --latent_dim 128 /data/iemanja/data/
# python test/test_audio_conv_vae.py --output_dir ./result/iemanja/cap64_lat64 --ratios 8 8 8 4 --capacity 64 --latent_dim 64 /data/iemanja/data/
# python test/test_audio_conv_vae.py --output_dir ./result/iemanja/cap64_lat32 --ratios 8 8 8 4 --capacity 64 --latent_dim 32 /data/iemanja/data/

# python test/test_audio_conv_vae.py --output_dir ./result/iemanja/beta001 --beta_kl 0.1 --ratios 8 8 8 4 --capacity 64 --latent_dim 128 /data/iemanja/data/
# python test/test_audio_conv_vae.py --output_dir ./result/iemanja/beta01 --beta_kl 0.01 --ratios 8 8 8 4 --capacity 64 --latent_dim 128 /data/iemanja/data/
# python test/test_audio_conv_vae.py --output_dir ./result/iemanja/beta0001 --beta_kl 0.001 --ratios 8 8 8 4 --capacity 64 --latent_dim 128 /data/iemanja/data/

python test/test_audio_conv_vae.py --output_dir ./result/iemanja/r4442 --beta_kl 0.001 --ratios 4 4 4 2 --capacity 64 --latent_dim 128 /data/iemanja/data/
python test/test_audio_conv_vae.py --output_dir ./result/iemanja/r8442 --beta_kl 0.001 --ratios 8 4 4 2 --capacity 64 --latent_dim 128 /data/iemanja/data/
python test/test_audio_conv_vae.py --output_dir ./result/iemanja/r8844 --beta_kl 0.001 --ratios 8 8 4 4 --capacity 64 --latent_dim 128 /data/iemanja/data/
python test/test_audio_conv_vae.py --output_dir ./result/iemanja/r8888 --beta_kl 0.001 --ratios 8 8 8 8 --capacity 64 --latent_dim 128 /data/iemanja/data/