import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = "16"
plt.rcParams["lines.linewidth"] = 1


# MDES, VoxelGrid together - on x-axis we have number of bins, on y-axis GWD metric on full GEN1 validation dataset
fig, ax = plt.subplots(figsize=(10, 3))
A = ax.plot(
    [1, 2, 3, 9, 12, 15],
    [
        0.74673848532958439,
        0.68001254864549601,
        0.57114556320014550,
        0.42159948575664453,
        0.40281647785577745,
        0.36334685743319348,
    ],
    marker=".",
    c="red",
    label="Voxel Grid",
)
B = ax.plot(
    [1, 2, 3, 9, 12, 15],
    [
        0.74507481102355310,
        0.66752351335679215,
        0.55550015758080901,
        0.39159504580790515,
        0.38314795761716137,
        0.33123495847331357,
    ],
    marker=".",
    c="blue",
    label="MDES",
)
ax.set_xlabel("Number of channels")
ax.set_ylabel("GWD")
ax.legend(
    frameon=True,
    loc="lower left",
    framealpha=1,
    edgecolor="0",
    handletextpad=0.1,
    borderpad=0.2,
    borderaxespad=0.5,
    fontsize=14,
)
ax.set_xticks(np.arange(1, 16))
ax.set_xticklabels(np.arange(1, 16))
fig.savefig(bbox_inches="tight", fname="/home/nzubic/toy_example_num_channels.pdf")


# VG and MDES where on x-axis we have number of sigma kernel blur operations applied on 12 channels and on y-axis we have GWD
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(
    [0, 2, 4],
    [0.40281647785577745, 0.67015056789313240, 0.80191055608035412],
    marker=".",
    c="red",
    label="Voxel Grid",
)
ax.plot(
    [0, 2, 4],
    [0.38314795761716136, 0.6304820476545163, 0.782242035841738],
    marker=".",
    c="blue",
    label="MDES",
)
ax.set_ylim([0.17, 0.82])

ax.set_xlabel("Gaussian blur $\sigma$ [pix]")
ax.set_ylabel("GWD")
ax.legend(
    frameon=True,
    loc="lower left",
    framealpha=1,
    edgecolor="0",
    handletextpad=0.1,
    borderpad=0.2,
    borderaxespad=0.5,
    fontsize=14,
)

ax.set_xticks([0, 2, 4])
ax.set_xticklabels([0, 2, 4])
fig.savefig(bbox_inches="tight", fname="/home/nzubic/toy_example_blur_radius.pdf")

plt.show()
