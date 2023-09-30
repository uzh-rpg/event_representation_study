import numpy as np
import ot as otpy
from sklearn.metrics import pairwise_distances
import warnings

warnings.filterwarnings("ignore")
import os


def compute_kernel(Cx, Cy, h):
    """
    Compute Gaussian kernel matrices following:

    1. Didong Li, Yulong Lu, Emmanuel Chevallier, and David B Dunson. Density estimation and modeling on symmetric spaces. arXiv preprint arXiv:2009.01983, 2020.
    2. Salem Said, Lionel Bombrun, Yannick Berthoumieu, and Jonathan H Manton. Riemannian gaussian distributions on the space of symmetric positive de€nite matrices. IEEE Transactions on Information Œeory, 63 (4):2153–2170, 2017.

    Parameters:
    Cx: source pairwise distance matrix
    Cy: target pairwise distance matrix
    h : bandwidth

    Return:
    Kx: source kernel
    Ky: target kernel
    """

    std1 = np.sqrt((Cx**2).mean() / 2)
    std2 = np.sqrt((Cy**2).mean() / 2)
    h1 = h * std1
    h2 = h * std2

    # Gaussian kernel (without normalization)
    Kx = np.exp(-((Cx / h1) ** 2) / 2)
    Ky = np.exp(-((Cy / h2) ** 2) / 2)

    return Kx, Ky


class OTMI:
    """
    Solver for OTMI. Source and target can have different dimensions.

    Parameters:
    Xs: source data
    Xt: target data
    h : bandwidth
    reg: weight for entropic regularization
    """

    def __init__(self, Xs, Xt, h, reg=0.05):
        self.Xs = Xs
        self.Xt = Xt
        self.h = h
        self.reg = reg

        # init kernel
        self.Cs = pairwise_distances(Xs, Xs, n_jobs=os.cpu_count())
        self.Ct = pairwise_distances(Xt, Xt, n_jobs=os.cpu_count())
        self.Ks, self.Kt = compute_kernel(self.Cs, self.Ct, h)
        self.P = None

    def solve(self):
        p = otpy.unif(len(self.Xs))
        q = otpy.unif(len(self.Xt))

        gw, log = otpy.gromov.gromov_wasserstein(
            self.Ks, self.Kt, p, q, "kl_loss", log=True, verbose=False
        )
        return gw, log["gw_dist"]


def compute_repr(x, y, t, p, width, height, bins=5):
    voxel_grid = np.zeros((height, width, bins))
    b = (bins - 1) * t
    b_int = b.astype("int")

    for blim in [b_int, b_int + 1]:
        weight = 1 - np.abs(blim - b)
        mask = blim < bins
        np.add.at(voxel_grid, (y[mask], x[mask], blim[mask]), weight[mask] * p[mask])

    return voxel_grid


if __name__ == "__main__":
    import numpy as np
    import cv2

    # np.random.seed(2)
    W = 300
    events = "/data/storage/nzubic/000100.npz"
    fh = np.load(events)
    x = fh["x"] / W
    y = fh["y"] / W
    t = fh["t"]
    t = (t - t[0]) / (t[-1] - t[0])
    p = fh["p"]  # polarity 0, 1

    mask = (fh["x"] < W) & (fh["y"] < W)

    events = np.stack([x[mask], y[mask], t[mask], p[mask]], axis=-1)

    lista = []

    for num_bins in [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        30,
        35,
        40,
        45,
        50,
    ]:
        x_ = fh["x"][mask]
        y_ = fh["y"][mask]
        t_ = t[mask]
        # p_ = 2*p[mask].astype("int32")-1
        p_ = p[mask]

        repr = compute_repr(x=x_, y=y_, t=t_, p=p_, width=W, height=W, bins=num_bins)
        # for b in range(num_bins):
        #    cv2.imshow("A", repr[...,b])
        #    cv2.waitKey(300)

        x_positional_embedding = (
            np.repeat(np.arange(0, 300).reshape(300, 1), repeats=300, axis=1) / 299
        )
        y_positional_embedding = (
            np.repeat(np.arange(0, 300).reshape(1, 300), repeats=300, axis=0) / 299
        )
        repr = np.concatenate(
            (
                repr,
                x_positional_embedding[..., np.newaxis],
                y_positional_embedding[..., np.newaxis],
            ),
            axis=2,
        )

        repr = repr.reshape((-1, num_bins + 2))
        mask1 = np.abs(repr[:, :-2]).sum(-1) > 0
        repr = repr[mask1]

        Xs = events.copy()
        Xt = repr.copy()

        print(Xs.shape, Xt.shape)

        ot = OTMI(Xs, Xt, h=0.7, reg=0.05)
        P, cost = ot.solve()
        lista.append((num_bins, cost))

        print(num_bins, cost)

    print(lista)
    # print(P.sum(0))
    # print(P.mean(), P.max(), P.min())
    # exit(0)


"""
x = reshaped_return_data[:, 0] / self.width
y = reshaped_return_data[:, 1] / self.height
t = reshaped_return_data[:, 2]
t = (t - t[0]) / (t[-1] - t[0])
p = reshaped_return_data[:, 3]
p = (p - min(p)) / (max(p) - min(p))
mask = (reshaped_return_data[:, 0] < self.width) & (reshaped_return_data[:, 1] < self.height)
events = np.stack([x[mask], y[mask], t[mask], p[mask]], axis=-1)

repr = rep.reshape((-1, 12))
mask1 = np.abs(repr).sum(-1) > 0
repr = repr[mask1]

Xs = events.copy()
Xt = repr.copy()

ot = OTMI(Xs, Xt, h=0.7, reg=0.05)
P, cost = ot.solve()

print(cost)
"""
