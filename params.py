import argparse
import exp

parser = argparse.ArgumentParser()

parser.add_argument('ini_file', nargs='*', help="Ini file to use for configuration")

parser.add_argument('--prefix', dest="prefix", default="trash", help="File prefix for the output visualizations and models.")
parser.add_argument('--callback_prefix', dest="callback_prefix", default="same", help="File prefix for the results renerated by callbacks ('same' means same as prefix)")
parser.add_argument('--lr', dest="lr", default="0.001", help="Learning rate for RMS prop.")
parser.add_argument('--dataset', dest="dataset", default="mnist", help="Dataset to use: mnist/celeba")
parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=10, help="Number of epochs")
parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension")
parser.add_argument('--intermediate_dims', dest="intermediate_dims", default="256", help="Intermediate dimensions")
parser.add_argument('--frequency', dest="frequency", type=int, default=10, help="image saving frequency")
parser.add_argument('--encoder', dest="encoder", default="dense", help="encoder type (dense/conv)")
parser.add_argument('--decoder', dest="decoder", default="dense", help="decoder type (dense/conv)")
parser.add_argument('--color', dest="color", default=0, type=int, help="color(0/1)")
parser.add_argument('--sampling', dest="sampling", type=int, default=1, help="use sampling (0/1)")
parser.add_argument('--spherical', dest="spherical", type=int, default=0, help="spherical (0/1)")
parser.add_argument('--depth', dest="depth", default=2, type=int, help="Depth of conv model")
parser.add_argument('--base_filter_num', dest="base_filter_num", default=32, type=int, help="Initial number of filter in the conv model")
parser.add_argument('--batch_size', dest="batch_size", default=1000, type=int, help="Batch size.")
parser.add_argument('--memory_share', dest="memory_share", type=float, default=0.7, help="fraction of memory that can be allocated to this process")
parser.add_argument('--xent_weight', dest="xent_weight", default=1, type=int, help="weight of the crossentropy loss")
parser.add_argument('--losses', dest="losses", default="xent_loss", help="list of losses")
parser.add_argument('--activation', dest="activation", default="relu", help="activation function")

args_param = parser.parse_args()
args = exp.mergeParamsWithInis(args_param)
ini_file = args.prefix + ".ini"
exp.dumpParams(args, ini_file)

def getArgs():
    assert args.encoder in ("dense", "conv")
    assert args.decoder in ("dense", "conv")

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

    args.intermediate_dims = map(int, str(args.intermediate_dims).split(","))
    args.losses = str(args.losses).split(",")
    return args



# assert args.model in ("ae", "rae", "vae", "nvae", "vae_conv", "nvae_conv", "vae_conv_sym", "nvae_conv_sym", "universal"), "Unknown model type."
# print "Training model of type %s" % args.model
