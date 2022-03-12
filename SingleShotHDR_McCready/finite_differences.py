import numpy as np

def opDx_(x, axis):
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(0, 1)
    Dx = np.diff(x, axis=axis, append=x[tuple(slc)])
    return Dx

def opDtx_(diffx, axis):
    Dtx = -diffx + np.roll(diffx, 1, axis=axis)
    return Dtx

def opDx(x):
    Dx = opDx_(x, axis=-1)
    Dy = opDx_(x, axis=-2)
    if len(Dx.shape) <= 3:
        return np.stack((Dx, Dy), axis=-3)
    else:
        return np.concatenate((Dx, Dy), axis=-3)

def opDtx(x):
    Dtx = opDtx_(x[..., 0, :, :], axis=-1)
    Dty = opDtx_(x[..., 1, :, :], axis=-2)
    return Dtx + Dty
