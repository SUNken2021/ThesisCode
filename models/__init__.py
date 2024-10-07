import time
import os
import shutil
from .DNN import DNN
from .CNN import CNN
from .Transformer import Transformer
from .EigenNN import EigenNN


def get_model(args):
    if args.model == "DNN":
        return DNN(args.DNN_set)
    elif args.model == "CNN":
        return CNN(args.CNN_set)
    elif args.model == "Transformer":
        return Transformer(args.Transformer_set)
    elif args.model == "EigenNN":
        return EigenNN(args.EigenNN_set)
    pass


def mkdir(args, suffix_form: str = "time"):
    if suffix_form == "time":
        t = time.localtime()
        suffix = "{}_{}_{}_{}_{}".format(
            t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
    dirname = "./save/"+args.model+"_"+suffix

    os.makedirs(dirname+"/figures")
    os.makedirs(dirname+"/state_dict")
    shutil.copyfile("./settings/trainset.json", dirname+"/trainset.json")
    shutil.copyfile("./models/"+args.model+".py", dirname+"/"+args.model+".py")
    return dirname
