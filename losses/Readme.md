Unlike most classification tasks, where you usually do:

    mp = DataParallel(m)
    pred = mp(data)
    loss = loss_fun(pred, label)

In this framework, we warp model and loss as a single module and Dataparallel this warp:

    warp = sequential(m, loss_fun)
    warp_p = DataParallel(warp)
    loss = warp_p(data, label)

The reason is that in former method, the loss_fun is conducted solely in master card, and the "label" and "pred" are also stored in master card, which makes this loss calculating step a bottleneck. Especially when perceptual loss is used, the computation cost at loss_fun can not be ignored.

In later method, the loss calculating step is also parallellized to different cards, making it more efficient.

Check warp_loss.py for detail

When using perceptual loss, one important thing to notice is the relative weight of different losses. The magnitude of features might be different from layer to layer, so tune the weight carefully to balance them. We recommend to try several iterations in debug mode to check it.

Another issue to notice is that the pixel level loss and perceptual loss are usually calculated with different metric, so they are usually not comparable.