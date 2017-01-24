import argparse
import exp
from model_gaussian import get_latent_dim

parser = argparse.ArgumentParser()

parser.add_argument('ini_file', nargs='*', help="Ini file to use for configuration")

parser.add_argument('--prefix', dest="prefix", default="trash", help="File prefix for the output visualizations and models.")
parser.add_argument('--callback_prefix', dest="callback_prefix", default="same", help="File prefix for the results renerated by callbacks ('same' means same as prefix)")
parser.add_argument('--lr', dest="lr", default="0.001", type=float, help="Learning rate for RMS prop.")
parser.add_argument('--dataset', dest="dataset", default="celeba", help="Dataset to use: mnist/celeba")
parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=200, help="Number of epochs")
parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension")
parser.add_argument('--intermediate_dims', dest="intermediate_dims", default="1000,1000", help="Intermediate dimensions")
parser.add_argument('--frequency', dest="frequency", type=int, default=20, help="image saving frequency")
parser.add_argument('--encoder', dest="encoder", default="conv", help="encoder type (dense/conv)")
parser.add_argument('--decoder', dest="decoder", default="conv", help="decoder type (dense/conv)")
parser.add_argument('--color', dest="color", default=1, type=int, help="color(0/1)")
parser.add_argument('--sampling', dest="sampling", type=int, default=1, help="use sampling (0/1)")
parser.add_argument('--spherical', dest="spherical", type=int, default=0, help="spherical (0/1)")
parser.add_argument('--depth', dest="depth", default=3, type=int, help="Depth of conv model")
parser.add_argument('--base_filter_num', dest="base_filter_num", default=32, type=int, help="Initial number of filter in the conv model")
parser.add_argument('--batch_size', dest="batch_size", default=200, type=int, help="Batch size.")
parser.add_argument('--memory_share', dest="memory_share", type=float, default=0.98, help="fraction of memory that can be allocated to this process")
parser.add_argument('--xent_weight', dest="xent_weight", default=1, type=int, help="weight of the crossentropy loss")
parser.add_argument('--losses', dest="losses", default="xent_loss", help="list of losses")
parser.add_argument('--metrics', dest="metrics", default="xent_loss", help="list of metrics")
parser.add_argument('--activation', dest="activation", default="relu", help="activation function")
parser.add_argument('--decoder_wd', dest="decoder_wd", type=float, default=0.0, help="Weight decay param for the decoder")
parser.add_argument('--encoder_wd', dest="encoder_wd", type=float, default=0.0, help="Weight decay param for the encoder")
parser.add_argument('--decoder_use_bn', dest="decoder_use_bn", type=int, default=0, help="Use batch norm in decoder")
parser.add_argument('--optimizer', dest="optimizer", type=str, default="adam", help="Optimizer, adam or rmsprop.")
parser.add_argument('--verbose', dest="verbose", type=int, default=2, help="Logging verbosity: 0-silent, 1-verbose, 2-perEpoch (default)")
parser.add_argument('--weight_schedules', dest="weight_schedules", default='', help="Comma separated list of loss schedules, ex size_loss|5|1|0.2|0.8 means that size_loss has has initial weight 5, final weight 1 and the weight is adjusted linearly after finishing the first 20% of the training and before finishing the 80% of training")
parser.add_argument('--trainSize', dest="trainSize", type=int, default=0, help="Train set size (0 means default size)")
parser.add_argument('--testSize', dest="testSize", type=int, default=0, help="Test set size (0 means default size)")
parser.add_argument('--gaussianParams', dest="gaussianParams", default="10,1,10", help="main_channel,dots,side_channel - this overrides latent_dim param")


args_param = parser.parse_args()
args = exp.mergeParamsWithInis(args_param)
ini_file = args.prefix + ".ini"
exp.dumpParams(args, ini_file)

def getArgs():
    assert args.encoder in ("dense", "conv")

    if args.callback_prefix == "same":
        args.callback_prefix = args.prefix
    if args.color == 1:
        args.color = True
    else:
        args.color = False
    if args.sampling == 1:
        args.sampling = True
    else:
        args.sampling = False
    if args.spherical == 1:
        args.spherical = True
    else:
        args.spherical = False
    if args.dataset == 'celeba':
        args.shape = (72, 64)
    else:
        args.shape = None
    if args.decoder_use_bn == 1:
        args.decoder_use_bn = True
    else:
        args.decoder_use_bn = False
    if type(args.intermediate_dims) is int:
        args.intermediate_dims = [args.intermediate_dims]
    elif len(args.intermediate_dims) > 0:
        args.intermediate_dims = map(int, str(args.intermediate_dims).split(","))
    else:
        args.intermediate_dims = []

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

    # if the decoder is gaussian, update latent_dim
    args.gaussianParams = map(int, str(args.gaussianParams).split(","))
    assert len(args.gaussianParams) == 3
    if args.decoder == "gaussian":
        args.latent_dim = get_latent_dim(args.gaussianParams)
    else: print "!!!!!!!!!!!!!!!!!!!!!!!!"
    return args
