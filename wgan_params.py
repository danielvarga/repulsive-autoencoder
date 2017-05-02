import argparse
import exp
from model_gaussian import get_latent_dim

parser = argparse.ArgumentParser()

parser.add_argument('ini_file', nargs='*', help="Ini file to use for configuration")

parser.add_argument('--prefix', dest="prefix", default="dcgan/trash", help="File prefix for the output visualizations and models.")
parser.add_argument('--lr', dest="lr", default="0.00005", type=float, help="Learning rate for RMS prop.")
parser.add_argument('--generator', dest="generator", default="dcgan", help="generator type (dense/dcgan)")
parser.add_argument('--discriminator', dest="discriminator", default="dcgan", help="discriminator type (dense/dcgan)")
parser.add_argument('--generator_wd', dest="generator_wd", type=float, default=0.0, help="Weight decay param for generator")
parser.add_argument('--discriminator_wd', dest="discriminator_wd", type=float, default=0.0, help="Weight decay param for discriminator")
parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=100, help="Latent dimension")
parser.add_argument('--batch_size', dest="batch_size", default=100, type=int, help="Batch size.")
parser.add_argument('--nb_iter', dest="nb_iter", type=int, default=1300, help="Number of iterations")
parser.add_argument('--dataset', dest="dataset", default="mnist", help="Dataset to use: mnist/celeba")
parser.add_argument('--trainSize', dest="trainSize", type=int, default=0, help="Train set size (0 means default size)")
parser.add_argument('--testSize', dest="testSize", type=int, default=0, help="Test set size (0 means default size)")
parser.add_argument('--color', dest="color", default=1, type=int, help="color(0/1)")
parser.add_argument('--shape', dest="shape", default="64,64", help="image shape, currently only 64,64 supported")
parser.add_argument('--frequency', dest="frequency", type=int, default=20, help="image saving frequency")
parser.add_argument('--memory_share', dest="memory_share", type=float, default=0.45, help="fraction of memory that can be allocated to this process")
parser.add_argument('--verbose', dest="verbose", type=int, default=0, help="Logging verbosity: 0-silent, 1-verbose, 2-perEpoch")
parser.add_argument('--optimizer', dest="optimizer", type=str, default="adam", help="Optimizer, adam, rmsprop, sgd.")
parser.add_argument('--clipValue', dest="clipValue", type=float, default=0.01, help="Critic clipping range is (-clipValue, clipValue)")
parser.add_argument('--use_bn_gen', dest="use_bn_gen", type=int, default=0, help="Use batch normalization in generator 0/1")
parser.add_argument('--use_bn_disc', dest="use_bn_disc", type=int, default=0, help="Use batch normalization in discriminator 0/1")
parser.add_argument('--gen_size', dest="gen_size", default="small", help="tiny/small/large")
parser.add_argument('--disc_size', dest="disc_size", default="large", help="tiny/small/large")
parser.add_argument('--modelPath', dest="modelPath", default=None, help="Path to saved networks. If none, build networks from scratch.")
parser.add_argument('--sampling', dest="sampling", type=int, default=0, help="use sampling (0/1)")
parser.add_argument('--nesterov', dest="nesterov", default="0.0", type=float, help="Nesterov momentum")
parser.add_argument('--ornstein', dest="ornstein", default="1.0", type=float, help="Ornstein process coefficient (1 means no movement")
parser.add_argument('--matching_frequency', dest="matching_frequency", default="1", type=int, help="After how many epoch do we rematch the data")
parser.add_argument('--min_items_in_matching', dest="min_items_in_matching", default="-1", type=int, help="Minimum number of items to run the matching (-1 means batch size)")


args_param = parser.parse_args()
args = exp.mergeParamsWithInis(args_param)
ini_file = args.prefix + ".ini"
exp.dumpParams(args, ini_file)

def getArgs():
    if args.color == 1:
        args.color = True
    else:
        args.color = False

    if args.sampling == 1:
        args.sampling = True
    else:
        args.sampling = False

    args.use_bn_gen = (args.use_bn_gen == 1)
    args.use_bn_disc = (args.use_bn_disc == 1)

    args.shape = tuple(map(int, str(args.shape).split(",")))

    if args.min_items_in_matching == -1:
        args.min_items_in_matching = args.batch_size
    return args
