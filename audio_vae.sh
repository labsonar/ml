#!/bin/bash
# python test/test_audio_vae.py --beta_kl 0.1 --stft_factor 1 --mel_factor 0 --output_dir ./result/audio_vae/stft /data/unique_sample/env/
# python test/test_audio_vae.py --beta_kl 0.1 --stft_factor 0 --mel_factor 1 --output_dir ./result/audio_vae/mel /data/unique_sample/env/
# python test/test_audio_vae.py --beta_kl 0.1 --stft_factor 1 --mel_factor 1 --output_dir ./result/audio_vae/comb /data/unique_sample/env/

# python test/test_audio_vae.py --beta_kl 1 --stft_factor 1 --mel_factor 1 --output_dir ./result/audio_vae/comb_kl_p /data/unique_sample/env/
# python test/test_audio_vae.py --beta_kl 0.0001 --stft_factor 1 --mel_factor 1 --output_dir ./result/audio_vae/comb_kl_m /data/unique_sample/env/

# python test/test_audio_vae.py --capacity 8 --output_dir ./result/audio_vae/capacity_8 /data/unique_sample/env/
# python test/test_audio_vae.py --capacity 32 --output_dir ./result/audio_vae/capacity_32 /data/unique_sample/env/
# python test/test_audio_vae.py --capacity 64 --output_dir ./result/audio_vae/capacity_64 /data/unique_sample/env/

# python test/test_audio_vae.py --latent_dim 8 --output_dir ./result/audio_vae/latent_8 /data/unique_sample/env/
# python test/test_audio_vae.py --latent_dim 16 --output_dir ./result/audio_vae/latent_16 /data/unique_sample/env/
# python test/test_audio_vae.py --latent_dim 64 --output_dir ./result/audio_vae/latent_64 /data/unique_sample/env/

# python test/test_audio_vae.py --pqmf_bands 4 --output_dir ./result/audio_vae/pqmf_4 /data/unique_sample/env/
# python test/test_audio_vae.py --pqmf_bands 16 --output_dir ./result/audio_vae/pqmf_16 /data/unique_sample/env/

# python test/test_audio_vae.py --noise_bands 4 --output_dir ./result/audio_vae/4_band /data/unique_sample/env/
# python test/test_audio_vae.py --noise_bands 16 --output_dir ./result/audio_vae/16_band /data/unique_sample/env/

# python test/test_audio_vae.py --noise_ratios 4 4 4 4 --output_dir ./result/audio_vae/res_0s128 /data/unique_sample/env/
# python test/test_audio_vae.py --noise_ratios 8 4 4 4 --output_dir ./result/audio_vae/res_0s256 /data/unique_sample/env/
# python test/test_audio_vae.py --noise_ratios 8 8 8 4 --output_dir ./result/audio_vae/res_1s /data/unique_sample/env/
# python test/test_audio_vae.py --noise_ratios 8 8 8 8 --output_dir ./result/audio_vae/res_2s /data/unique_sample/env/

python test/test_audio_vae.py --capacity 8 --stft_factor 1 --mel_factor 0 --output_dir ./result/audio_vae_nb/stft /data/unique_sample/nb/
# python test/test_audio_vae.py --capacity 8 --stft_factor 0 --mel_factor 1 --output_dir ./result/audio_vae_nb/mel /data/unique_sample/nb/
# python test/test_audio_vae.py --capacity 8 --stft_factor 0 --mel_factor 0 --lofar_factor 1 --output_dir ./result/audio_vae_nb/lofar /data/unique_sample/nb/
# python test/test_audio_vae.py --capacity 8 --stft_factor 1 --mel_factor 1 --lofar_factor 1 --output_dir ./result/audio_vae_nb/comb_loss /data/unique_sample/nb/