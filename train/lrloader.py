from .lr_scheme import *


class LRLoader():
    def __init__(self):
        pass

    @staticmethod
    def load(funcname):
        func = globals()[funcname]
        return func


if __name__ == "__main__":
    lr = LRLoader.load("base_lr")
    print(lr(1,10,10))
