import argparse
import sys
import time
import json
from numpy import require

from openea.modules.args.args_hander import check_args, load_args
from openea.modules.load.kgs import read_kgs_from_folder
from openea.models.trans import TransD
from openea.models.trans import TransE
from openea.models.trans import TransH
from openea.models.trans import TransR
from openea.models.semantic import DistMult
from openea.models.semantic import HolE
from openea.models.semantic import SimplE
from openea.models.semantic import RotatE
from openea.models.neural import ConvE
from openea.models.neural import ProjE
from openea.approaches import AlignE
from openea.approaches import BootEA
from openea.approaches import BootEAPro
from openea.approaches import BootEABase
from openea.approaches import JAPE
from openea.approaches import Attr2Vec
from openea.approaches import MTransE
from openea.approaches import IPTransE
from openea.approaches import GCN_Align
from openea.approaches import AttrE
from openea.approaches import IMUSE
from openea.approaches import SEA
from openea.approaches import MultiKE
from openea.approaches import RSN4EA
from openea.approaches import GMNN
from openea.approaches import KDCoE
from openea.approaches import RDGCN
from openea.approaches import BootEA_RotatE
from openea.approaches import BootEA_TransH
from openea.approaches import AliNet
from openea.models.basic_model import BasicModel


class ModelFamily(object):
    BasicModel = BasicModel

    TransE = TransE
    TransD = TransD
    TransH = TransH
    TransR = TransR

    DistMult = DistMult
    HolE = HolE
    SimplE = SimplE
    RotatE = RotatE

    ProjE = ProjE
    ConvE = ConvE

    MTransE = MTransE
    IPTransE = IPTransE
    Attr2Vec = Attr2Vec
    JAPE = JAPE
    AlignE = AlignE
    BootEA = BootEA
    BootEAPro = BootEAPro
    BootEABase = BootEABase
    GCN_Align = GCN_Align
    GMNN = GMNN
    KDCoE = KDCoE

    AttrE = AttrE
    IMUSE = IMUSE
    SEA = SEA
    MultiKE = MultiKE
    RSN4EA = RSN4EA
    RDGCN = RDGCN
    BootEA_RotatE = BootEA_RotatE
    BootEA_TransH = BootEA_TransH
    AliNet = AliNet


def get_model(model_name):
    return getattr(ModelFamily, model_name)

import argparse

parser = argparse.ArgumentParser(description='OpenEA')
parser.add_argument('--verify-range', type=float, nargs=2, required=False)
parser.add_argument('--interact-iter', type=int, default=0)
parser.add_argument('--only-top', type=bool, default=False)
# 单次迭代验证最大数
parser.add_argument('--max-correct', type=int, default=1000)
parser.add_argument('-m', '--embedding_module', type=str, required=False)

if __name__ == '__main__':
    t = time.time()
    args = load_args(sys.argv[1]) # json 读参数
    args.training_data = args.training_data + sys.argv[2] + '/'
    args.dataset_division = sys.argv[3]
    # 最小迭代次数，当前终止条件准确度连续下降两次不太可靠
    # args.min_iter = int(sys.argv[4])

    extra_args = parser.parse_args(sys.argv[4:])
    for k, v in extra_args.__dict__.items():
        if v is not None and hasattr(args, k):
            args.__setattr__(k, v)
        elif not hasattr(args, k):
            args.__setattr__(k, v)

    print(args.embedding_module)
    print('模型参数设置')
    print(json.dumps(args.__dict__, indent=2))
    remove_unlinked = False
    if args.embedding_module == "RSN4EA":
        remove_unlinked = True
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=remove_unlinked)
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    model.run()
    model.test()
    model.save()
    print("Total run time = {:.3f} s.".format(time.time() - t))