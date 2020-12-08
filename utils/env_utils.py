import torch


def setEnvironment(envi_para):
    torch_rand = envi_para.env["torch_rand"]

    if torch_rand == 1:
        torch.manual_seed(0)
