#from .net_file_sk import *
from .unets import *
from .unetsb import *
from .unetsc import *
from .unets_big import *
# from .unets_big2 import *
# from .unets_small import *
from .unets_big_type2 import *
from .unets_big_type3 import *
from .unets_big_v2 import *
from .unets_big_v3 import *
from .unets_big_v4 import *

class ModelLoader():
    def __init__(self):
        pass

    @staticmethod
    def load(modelname, *args, **kwargs):
        object = globals()[modelname]
        return object(*args, **kwargs)


if __name__ == "__main__":
    net = ModelLoader.load("SKUNET")
    print(net.cuda())
