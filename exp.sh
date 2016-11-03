#CUDA_VISIBLE_DEVICES=1 nohup python autoencoder.py --nb_epoch 100 --output nvae_nvae_conv_e100_i128_l64_f128_ebn --dataset celeba --model nvae_conv --intermediate_dim 128 --latent_dim 64 > nvae_conv_e100_i128_l64_f128_ebn.cout 2> nvae_conv_e100_i128_l64_f128_ebn.cerr &
CUDA_VISIBLE_DEVICES=1 nohup python autoencoder.py --nb_epoch 10 --output classic_resi --dataset celeba --model nvae_conv --intermediate_dim 128 --latent_dim 64 > classic_resi.cout 2> classic_resi.cerr &


