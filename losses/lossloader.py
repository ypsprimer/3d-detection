from .rpn_loss import *
class LossLoader():
    def __init__(self):
        pass

    @staticmethod
    def load(lossname, config):

        object = globals()[lossname]
        return object(config=config)


if __name__ == "__main__":
    loss = LossLoader.load("DiceLoss")
    print(loss)