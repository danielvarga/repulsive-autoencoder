CUDA_VISIBLE_DEVICES=0 nohup python autogenerator.py --nb_epoch 20 --output genx --dataset celeba --model nvae --intermediate_dim 128,128 --latent_dim 64 > genx.cout 2> genx.cerr &


