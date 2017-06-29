import os
import argparse

import exp
from model_gaussian import get_latent_dim

def str2bool(v):
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# architecture
parser.add_argument('--spherical', dest="spherical", type=str2bool, default=True, help="spherical (True/False)")
parser.add_argument('--sampling', dest="sampling", type=str2bool, default=False, help="use sampling")
parser.add_argument('--sampling_std', dest="sampling_std", type=float, default=-1.0, help="sampling std, if < 0, then we learn std")
parser.add_argument('--intermediate_dims', dest="intermediate_dims", default="1000,1000", help="Intermediate dimensions")
parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension")
parser.add_argument('--activation', dest="activation", default="relu", help="activation function")
parser.add_argument('--depth', dest="depth", default=3, type=int, help="Depth of conv vae model")
parser.add_argument('--base_filter_num', dest="base_filter_num", default=32, type=int, help="Initial number of filter in the conv model")

# training
parser.add_argument('--optimizer', dest="optimizer", type=str, default="adam", help="Optimizer, adam or rmsprop or sgd")
parser.add_argument('--lr', dest="lr", default="0.001", type=float, help="Learning rate for the optimizer.")
parser.add_argument('--batch_size', dest="batch_size", default=200, type=int, help="Batch size.")
parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=200, help="Number of epochs")
parser.add_argument('--nb_iter', dest="nb_iter", type=int, default=1300, help="Number of iterations") #TODO
parser.add_argument('--verbose', dest="verbose", type=int, default=2, help="Logging verbosity: 0-silent, 1-verbose, 2-perEpoch (default)")
parser.add_argument('--nesterov', dest="nesterov", default="0.0", type=float, help="Nesterov momentum")
parser.add_argument('--weight_schedules', dest="weight_schedules", default='', help="Comma separated list of loss schedules, ex size_loss|5|1|0.2|0.8 means that size_loss has has initial weight 5, final weight 1 and the weight is adjusted linearly after finishing the first 20% of the training and before finishing the 80% of training")
parser.add_argument('--losses', dest="losses", default="mse_loss", help="list of losses")
parser.add_argument('--metrics', dest="metrics", default="mse_loss", help="list of metrics")
parser.add_argument('--lr_decay_schedule', dest="lr_decay_schedule", default='0.5,0.8', help="Comma separated list floats from [0,1] indicating where to decimate the learning rate. Ex 0.2,0.5 means we decimate the learning rate at 20% and 50% of the training")

# dataset
parser.add_argument('--dataset', dest="dataset", default="celeba", help="Dataset to use")
parser.add_argument('--color', dest="color", default=True, type=str2bool, help="color(True/False)")
parser.add_argument('--trainSize', dest="trainSize", type=int, default=0, help="Train set size (0 means default size)")
parser.add_argument('--testSize', dest="testSize", type=int, default=0, help="Test set size (0 means default size)")
parser.add_argument('--shape', dest="shape", default="64,64", help="image shape")

# encoder
parser.add_argument('--encoder', dest="encoder", default="conv", help="encoder type (dense/conv)")
parser.add_argument('--encoder_wd', dest="encoder_wd", type=float, default=0.0, help="Weight decay param for the encoder")
parser.add_argument('--encoder_use_bn', dest="encoder_use_bn", type=str2bool, default=False, help="Use batch norm in encoder")
parser.add_argument('--use_nat', dest="use_nat", default=False, type=str2bool, help="If true, we sample points from the latent prior, match them with images and try to push latent points to the corresponding sampled points")

# decoder/generator
parser.add_argument('--decoder', dest="decoder", default="conv", help="decoder type (dense/conv)") #TODO
parser.add_argument('--generator', dest="generator", default="dcgan", help="generator type (dense/dcgan)")
parser.add_argument('--decoder_wd', dest="decoder_wd", type=float, default=0.0, help="Weight decay param for the decoder") #TODO
parser.add_argument('--generator_wd', dest="generator_wd", type=float, default=0.0, help="Weight decay param for generator")
parser.add_argument('--decoder_use_bn', dest="decoder_use_bn", type=str2bool, default=False, help="Use batch norm in decoder") # TODO
parser.add_argument('--use_bn_gen', dest="use_bn_gen", type=str2bool, default=False, help="Use batch normalization in generator")
parser.add_argument('--dcgan_size', dest="dcgan_size", default="small", help="tiny/small/large")

# discriminator
parser.add_argument('--discriminator', dest="discriminator", default="dcgan", help="discriminator type (dense/dcgan)")
parser.add_argument('--discriminator_wd', dest="discriminator_wd", type=float, default=0.0, help="Weight decay param for discriminator")
parser.add_argument('--clipValue', dest="clipValue", type=float, default=0.01, help="Critic clipping range is (-clipValue, clipValue)")
parser.add_argument('--use_bn_disc', dest="use_bn_disc", type=str2bool, default=False, help="Use batch normalization in discriminator")
parser.add_argument('--disc_size', dest="disc_size", default="large", help="tiny/small/large")

# NAT
parser.add_argument('--ornstein', dest="ornstein", default="1.0", type=float, help="Ornstein process coefficient (1 means no movement")
parser.add_argument('--matching_frequency', dest="matching_frequency", default="1", type=int, help="After how many epoch do we rematch the data")
parser.add_argument('--min_items_in_matching', dest="min_items_in_matching", default="-1", type=int, help="Minimum number of items to run the matching (-1 means batch size)")
parser.add_argument('--use_labels_as_latent', dest="use_labels_as_latent", default=False, type=str2bool, help="Only available for celeba, the latent points are the labels")
parser.add_argument('--greedy_matching', dest="greedy_matching", default=False, type=str2bool, help="If True, then matching is greedy")
parser.add_argument('--projection', dest="projection", default="0", type=int, help="if > 0 then project images to the specified dimension before computing distance matrix")
parser.add_argument('--no_update_epochs', dest="no_update_epochs", type=int, default=0, help="Number of epochs during which we do not perform gradient update (only matching)")
parser.add_argument('--no_matching_epochs', dest="no_matching_epochs", type=int, default=0, help="Number of epochs during which we do not perform rematching (only gradient update)")
parser.add_argument('--use_augmentation', dest="use_augmentation", type=str2bool, default=False, help="If True we use data augmentation specified in tranform_images.py")
parser.add_argument('--oversampling', dest="oversampling", type=int, default=0, help="How many extra latent points should we use (oversampling)")

# natAE
parser.add_argument('--distance_space', dest="distance_space", default="latent", help="The space in which we compute distances (latent/pixel)")

# radial basis model
parser.add_argument('--gaussianParams', dest="gaussianParams", default="10,1,10", help="main_channel,dots,side_channel - this overrides latent_dim param")
parser.add_argument('--gaussianVariance', dest="gaussianVariance", default=0.1, type=float, help="Maximum variance of the dots.")


# locations
parser.add_argument('ini_file', nargs='*', help="Ini file to use for configuration")
parser.add_argument('--prefix', dest="prefix", default="trash", help="File prefix for the output visualizations and models.")
parser.add_argument('--callback_prefix', dest="callback_prefix", default="same", help="File prefix for the results renerated by callbacks ('same' means same as prefix)")
parser.add_argument('--modelPath', dest="modelPath", default=None, help="Path to saved networks. If none, build networks from scratch.")
parser.add_argument('--latent_point_file', dest="latent_point_file", default=None, help="npy file that contains NAT latent points")

# micellaneous
parser.add_argument('--memory_share', dest="memory_share", type=float, default=0.45, help="fraction of memory that can be allocated to this process")
parser.add_argument('--frequency', dest="frequency", type=int, default=20, help="image saving frequency")
parser.add_argument('--layers_to_monitor', dest="layers_to_monitor", default=1, help="comma separated list of layers to monitor")
parser.add_argument('--monitor_frequency', dest="monitor_frequency", type=int, default=0, help="After how many batches should we save the activations of monitored layers (0 means no saving")

# deprecated
parser.add_argument('--xent_weight', dest="xent_weight", default=-200, type=int, help="weight of the crossentropy loss")
parser.add_argument('--gen_size', dest="gen_size", default=None, help="tiny/small/large") # dcgan_size instead


args_param = parser.parse_args()
args = exp.mergeParamsWithInis(args_param)
ini_file = args.prefix + ".ini"
exp.dumpParams(args, ini_file)

def getArgs():

    if args.xent_weight != -200:
        print "xent_weight argument is deprecated and it is not doing anything!"
    if args.gen_size is not None:
        print "gen_size argument is deprecated, please use dcgan_size instead! Now forwarding its value to dcgan_size"

    # put output files in a separate directory
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    prefix_parts = args.prefix.split("/")
    prefix_parts.append(prefix_parts[-1])
    args.prefix = "/".join(prefix_parts)

    if args.callback_prefix == "same":
        args.callback_prefix = args.prefix
    args.shape = tuple(map(int, str(args.shape).split(",")))
    if type(args.intermediate_dims) is int:
        args.intermediate_dims = [args.intermediate_dims]
    elif len(args.intermediate_dims) > 0:
        args.intermediate_dims = map(int, str(args.intermediate_dims).split(","))
    else:
        args.intermediate_dims = []

    if type(args.lr_decay_schedule) is float:
        args.lr_decay_schedule = [args.lr_decay_schedule]
    elif len(args.lr_decay_schedule) > 0:
        args.lr_decay_schedule = map(float, str(args.lr_decay_schedule).split(","))
    else:
        args.lr_decay_schedule = []

    args.losses = str(args.losses).split(",")
    args.metrics = str(args.metrics).split(",")
    args.metrics = sorted(set(args.metrics + args.losses))

    weight_schedules = []
    if len(args.weight_schedules) > 0:
        import keras.backend as K
        for schedule in str(args.weight_schedules).split(","):
            schedule_list = schedule.split("|")
            schedule_list[1:5] = map(float,schedule_list[1:5])
            var = K.variable(value=schedule_list[1])
            schedule_list.append(var)
            weight_schedules.append(schedule_list)
    args.weight_schedules = weight_schedules

    if type(args.layers_to_monitor) is int:
        args.layers_to_monitor = [args.layers_to_monitor]
    elif len(args.layers_to_monitor) > 0:
        args.layers_to_monitor = map(int, str(args.layers_to_monitor).split(","))
    else:
        args.layers_to_monitor = []

    args.gaussianParams = map(int, str(args.gaussianParams).split(","))
    assert len(args.gaussianParams) == 3

    return args
