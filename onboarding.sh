# Create github account.

###########
# Setting up environment on one's own computer

# Python setup
# We don't use conda. You might try, it will probably work, but no promises.

# Install python 3.
# Install pip
pip3 install virtualenv

# Create local virtualenv:
virtualenv venv
# Activate virtualenv:
. venv/bin/activate
# -> or something similar for Windows, google it.

# Install modules:
# Must have modules:
pip install tensorflow keras scikit-learn h5py matplotlib pillow annoy pyemd
# Nice to have modules:
pip install scikit-image pandas seaborn networkx torch jupyterlab

# Install git command line

# Ask danielvarga for write access to repulsive-autoencoder repo if necessary.

###########
# Try network training code on toy example:

git clone https://github.com/danielvarga/repulsive-autoencoder.git
# ...or even better, set up ssh access to github (see below), and use this:
git clone git@github.com:danielvarga/repulsive-autoencoder.git

cd repulsive-autoencoder
mkdir pictures
python autoencoder.py ini/vae_syn-clocks-hand1.ini
# -> look for output in pictures/vae_syn-clocks-hand1/

############
# Migrating to a real computer

# Ask Renyi sysadmins for renyi.hu account
ssh -p 2820 account@renyi.hu
renyi> mkdir -p www/tmp # -> this is where we'll check our charts.

# TODO What's up with Authenticator?

# Ask Deep Learning Group for geforce accounts with the same user name:
renyi> ssh geforce  # aka 10.0.4.222 on the VPN
renyi> ssh geforce2 # aka 10.0.4.223 on the VPN

# TODO Request VPN access for users. Document VPN tunneling.
# TODO Having VPN access ready, set up and document X-based direct video access.

# Set up passwordless ssh access in all 2x3 directions:
renyi> ssh-keygen # ...then press enter three times.
renyi> ssh-copy-id geforce
renyi> ssh-copy-id geforce2
geforce*> ssh-keygen # ...then press enter three times.
geforce*> ssh-copy-id -p 2820 renyi.hu
geforce*> ssh-copy-id other_geforce # Here other_geforce means either geforce or geforce2

############
# Setting up bash environment

# on both geforce machines:
cp /home/daniel/.bashrc ~
# logout-login

# This takes care of activating the best virtualenv

############
# Adding public keys to github:

open https://github.com/settings/keys
# for both machines:
# click New SSH Key
# add as title geforce1 or geforce2
# add as key the public key geforce*:./.ssh/id_rsa.pub

############
# Our first GPU training


# on geforce:
mkdir experiments
git clone git@github.com:danielvarga/repulsive-autoencoder.git
cd repulsive-autoencoder
mkdir pictures

# check GPU availability:
nvidia-smi
# -> Bijection between CUDA_DEVICE numbering and nvidia-smi numbering is ad hoc.
#    Currently on geforce: 0->2 1->1 2->0, on geforce2: 0->1 1->0

# Let's say CUDA_DEVICE 0 is free:
CUDA_VISIBLE_DEVICE=0 python autoencoder.py ini/dcgan_vae.ini
# -> this currently assumes that data files are present at scattered places like
#    /home/zombori/datasets/celeba_72_64_color.npy
# -> output is in pictures/dcgan_vae/

# To prevent termination of process after logging out from geforce:
CUDA_VISIBLE_DEVICE=0 nohup python autoencoder.py ini/dcgan_vae.ini > dcgan_vae.cout 2> dcgan_vae.cerr &
# -> output now in dcgan_vae.cout

scp -P 2820 pictures/dcgan_vae_lsun/dcgan_vae-interp.png renyi.hu:./www/tmp/
# on local machine:
open https://old.renyi.hu/~accountname/tmp/

# When training is finished, trained network is saved, and we can use it for post-hoc analysis:
CUDA_VISIBLE_DEVICE=0 nohup python project_latent.py ini/dcgan_vae.ini
# -> Output shows that lots of charts and images are created. Let's check them.

# TODO Tensorboard

# TODO Figure out how to share exps without giving 777 write access for everything.
# (The issue is that project_latent.py writes to the directory where it takes the networks from.)
