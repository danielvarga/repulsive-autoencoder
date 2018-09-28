from __future__ import print_function
import os

# Causes memory leak below python 2.7.3
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def dumpParams(params, ini_file):
    if not os.path.exists(os.path.dirname(ini_file)):
        os.makedirs(os.path.dirname(ini_file))
    f = open(ini_file, "w+")
    for k in sorted(params.keys()):
        toPrint = k+"\t"+str(params[k])
        # print >>f, toPrint
        print(toPrint, file=f)

def heuristicCast(s):
    s = s.strip() # Don't let some stupid whitespace fool you.
    if s=="None":
        return None
    elif s=="True":
        return True
    elif s=="False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def paramsFromConf(f):
    params = AttrDict()
    for l in f:
        if l.strip().startswith("#"):
            continue
        try:
            k, v = l.strip().split(None, 1)
        except:
            assert False, "Malformed config line " + l.strip()
        try:
            v = heuristicCast(v)
        except ValueError:
            assert False, "Malformed parameter value " + v
        params[k] = v
    return params

def mergeParamsWithInis(args_param, ini_files_param="ini_file"):

    args = AttrDict()

    args_param_dict = vars(args_param)

    for k in args_param_dict:
        args[k] = args_param_dict[k]

    if args_param_dict[ini_files_param]:
        for ini_file in args_param_dict[ini_files_param]:
            args_ini_dict = paramsFromConf(open(ini_file))
            for k in args_ini_dict:
                if (k in args_param_dict) and isinstance(args[k], bool): # convert parameter from ini file to bool
                    args[k] = str2bool(args_ini_dict[k])
                else:
                    args[k] = args_ini_dict[k]

    return args


def str2bool(v):
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
