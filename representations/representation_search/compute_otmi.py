import numpy as np
import ot as otpy
from sklearn.metrics import pairwise_distances


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


def pad_arrays_to_same_shape(A, B):
    # Find the max shape
    max_shape = tuple(map(max, zip(A.shape, B.shape)))

    # Pad A
    pad_A = [(0, max_shape[i] - A.shape[i]) for i in range(len(max_shape))]
    A_padded = np.pad(A, pad_A)

    # Pad B
    pad_B = [(0, max_shape[i] - B.shape[i]) for i in range(len(max_shape))]
    B_padded = np.pad(B, pad_B)

    return A_padded, B_padded


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
        self.Cs = pairwise_distances(Xs, Xs, n_jobs=4)
        self.Ct = pairwise_distances(Xt, Xt, n_jobs=4)
        self.Ks, self.Kt = compute_kernel(self.Cs, self.Ct, h)
        self.P = None

    def loss(self, A, B):
        A_padded, B_padded = pad_arrays_to_same_shape(self.Ks, self.Kt)
        return np.abs(A_padded - B_padded)

    def solve(self):
        p = otpy.unif(len(self.Xs))
        q = otpy.unif(len(self.Xt))

        gw, log = otpy.gromov.sampled_gromov_wasserstein(
            self.Ks,
            self.Kt,
            p,
            q,
            loss_fun=self.loss,
            epsilon=0.1,
            log=True,
            verbose=False,
            max_iter=0,
        )

        return gw, log["gw_dist_estimated"]  # / 100


def otmi(events, rep, height, width, rep_size):
    x_1 = [0, width / 2 - 1]
    y_1 = [0, height / 2 - 1]

    x_2 = [width / 2 - 1, width - 1]
    y_2 = [0, height / 2 - 1]

    x_3 = [0, width / 2 - 1]
    y_3 = [height / 2 - 1, height - 1]

    x_4 = [width / 2 - 1, width - 1]
    y_4 = [height / 2 - 1, height - 1]

    first = events[
        (events[:, 0] >= x_1[0])
        & (events[:, 0] <= x_1[1])
        & (events[:, 1] >= y_1[0])
        & (events[:, 1] <= y_1[1])
    ]
    second = events[
        (events[:, 0] > x_2[0])
        & (events[:, 0] <= x_2[1])
        & (events[:, 1] >= y_2[0])
        & (events[:, 1] <= y_2[1])
    ]
    third = events[
        (events[:, 0] >= x_3[0])
        & (events[:, 0] <= x_3[1])
        & (events[:, 1] > y_3[0])
        & (events[:, 1] <= y_3[1])
    ]
    fourth = events[
        (events[:, 0] > x_4[0])
        & (events[:, 0] <= x_4[1])
        & (events[:, 1] > y_4[0])
        & (events[:, 1] <= y_4[1])
    ]

    shape_sizes = [first.shape[0], second.shape[0], third.shape[0], fourth.shape[0]]
    ind = shape_sizes.index(max(shape_sizes))

    # first stays as expected

    # second to [0, 151] and [0, 119] interval
    second[:, 0] = second[:, 0] - min(second[:, 0])
    second[:, 1] = second[:, 1] - min(second[:, 1])

    third[:, 0] = third[:, 0] - min(third[:, 0])
    third[:, 1] = third[:, 1] - min(third[:, 1])

    fourth[:, 0] = fourth[:, 0] - min(fourth[:, 0])
    fourth[:, 1] = fourth[:, 1] - min(fourth[:, 1])

    # xys = [(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)]
    xys = [
        ([0, rep_size // 2 - 1], [0, rep_size / 2 - 1]),
        ([rep_size / 2 - 1, rep_size - 1], [0, rep_size / 2 - 1]),
        ([0, rep_size / 2 - 1], [rep_size / 2 - 1, rep_size - 1]),
        ([rep_size / 2 - 1, rep_size - 1], [rep_size / 2 - 1, rep_size - 1]),
    ]

    current = [first, second, third, fourth]
    costs = []

    for i in range(len(current)):
        if i == ind:
            continue

        x = current[i][:, 0] / ((width - 1) // 2)
        y = current[i][:, 1] / ((height - 1) // 2)
        t = current[i][:, 2]
        t = (t - t[0]) / (t[-1] - t[0])
        p = current[i][:, 3]
        p = (p - min(p)) / (max(p) - min(p))
        mask = (current[i][:, 0] < (width - 1) // 2) & (
            current[i][:, 1] < (height - 1) // 2
        )
        events = np.stack([x[mask], y[mask], t[mask], p[mask]], axis=-1)

        curr_x, curr_y = xys[i]

        repr = rep[
            int(curr_y[0]) : int(curr_y[1]) + 1, int(curr_x[0]) : int(curr_x[1]) + 1, :
        ]

        x_positional_embedding = np.repeat(
            np.arange(0, repr.shape[0]).reshape(repr.shape[0], 1),
            repeats=repr.shape[1],
            axis=1,
        ) / (repr.shape[0] - 1)
        y_positional_embedding = np.repeat(
            np.arange(0, repr.shape[1]).reshape(1, repr.shape[1]),
            repeats=repr.shape[0],
            axis=0,
        ) / (repr.shape[1] - 1)
        repr = np.concatenate(
            (
                repr,
                x_positional_embedding[..., np.newaxis],
                y_positional_embedding[..., np.newaxis],
            ),
            axis=2,
        )

        repr = repr.reshape((-1, rep.shape[2] + 2))
        mask1 = np.abs(repr[:, :-2]).sum(-1) > 0
        repr = repr[mask1]

        Xs = events.copy()
        Xt = repr.copy()

        ot = OTMI(Xs, Xt, h=0.7, reg=0.05)
        P, cost = ot.solve()
        costs.append(cost)

    return np.mean(costs)
