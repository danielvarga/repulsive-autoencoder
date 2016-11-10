#CUDA_VISIBLE_DEVICES=1 nohup python autoencoder.py --nb_epoch 100 --output nvae_nvae_conv_e100_i128_l64_f128_ebn --dataset celeba --model nvae_conv --intermediate_dim 128 --latent_dim 64 > nvae_conv_e100_i128_l64_f128_ebn.cout 2> nvae_conv_e100_i128_l64_f128_ebn.cerr &
#CUDA_VISIBLE_DEVICES=1 nohup python autoencoder.py --nb_epoch 300 --output classic_resi_resi --dataset celeba --model nvae_conv --intermediate_dim 128 --latent_dim 64 --frequency 10 > classic_resi_resi.cout 2> classic_resi_resi.cerr &
#CUDA_VISIBLE_DEVICES=1 nohup python autoencoder.py --nb_epoch 100 --output classic_resinob_dense --dataset celeba --model nvae_conv --intermediate_dim 128,256,512 --latent_dim 64 --frequency 10 > classic_resinob_dense.cout 2> classic_resinob_dense.cerr &
#CUDA_VISIBLE_DEVICES=1 nohup python autoencoder.py --nb_epoch 100 --output classic_resinob_resinobd --dataset celeba --model nvae_conv --intermediate_dim 128 --latent_dim 64 --frequency 10 > classic_resinob_resinobd.cout 2> classic_resinob_resinobd.cerr &



#CUDA_VISIBLE_DEVICES=1 nohup python autoencoder.py --nb_epoch 100 --output classic_resi_trash --dataset celeba --model nvae_conv --intermediate_dim 128 --latent_dim 64 --frequency 10 > classic_resi_trash.cout 2> classic_resi_trash.cerr &

#CUDA_VISIBLE_DEVICES=0 nohup python autoencoder.py --nb_epoch 100 --output classic_resi_l512_trash --dataset celeba --model nvae_conv --intermediate_dim 128 --latent_dim 512 --frequency 10 > classic_resi_l512_trash.cout 2> classic_resi_l512_trash.cerr &
CUDA_VISIBLE_DEVICES=0 nohup python autoencoder.py --nb_epoch 100 --output disc_1 --dataset celeba --model nvae_conv --intermediate_dim 128 --latent_dim 512 --frequency 10 > disc_1.cout 2> disc_1.cerr &
