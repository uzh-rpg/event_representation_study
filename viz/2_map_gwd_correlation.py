import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = "16"
plt.rcParams["lines.linewidth"] = 1


def plot(C_p, methods, mAP, colors, labels, xlim, ylim, dataset_name="Gen1"):
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, key in enumerate(methods):
        ax.scatter(mAP[i], C_p, c=colors[i], label=key)

    ax.set_ylabel("GWD [-]")
    ax.set_xlabel(f"mAP [-]")

    ax.legend(
        frameon=True,
        loc="lower left",
        handletextpad=0.1,
        borderpad=0.2,
        borderaxespad=0.5,
        fontsize=14,
    )

    ax.annotate(
        text=rf"\textbf{{{dataset_name}}}",
        fontsize=21,
        xy=(xlim[1] - 0.03, ylim[1] - 0.03),
    )

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    for idx, label in enumerate(labels):
        ax.annotate(text=label, fontsize=14, xy=(mAP[0][idx] + 0.02, C_p[idx] - 0.005))

    return fig, ax


# gen1
markers = ["*", "^", "s"]
colors = ["r", "g", "b"]
methods = ["Swin-V2", "EfficientRep", "ResNet-50"]
labels = ["EST", "Voxel Grid", "MDES", "Time Surface", "2D Histogram", "TORE"]

C_p = [
    0.3552,
    0.40281647785577745,
    0.38314795761716137,
    0.3252,
    0.621960598186318,
    0.3694,
]
mAP = [
    [0.4531, 0.4249, 0.4375, 0.5007, 0.3598, 0.4465],
    [0.41, 0.3812, 0.3933, 0.4233, 0.3189, 0.4001],
    [0.37, 0.3398, 0.3445, 0.3765, 0.2781, 0.3589],
]

# fig, ax = plot(C_p, methods, mAP, colors, labels, xlim=[0.25, 0.55], ylim=[0.28, 0.65], dataset_name="Gen1")

# uncomment this one only if you want to plot without our method
fig, ax = plot(
    C_p,
    methods,
    mAP,
    colors,
    labels,
    xlim=[0.25, 0.55],
    ylim=[0.28, 0.65],
    dataset_name="Gen1",
)


# plot our method
C_p_ours = 0.30449148912568236
ax.scatter(0.519, C_p_ours, c="r", marker="*", s=200)
ax.annotate(
    text=r"\textbf{ERGO-12 (ours)}", fontsize=14, xy=(0.519 - 0.07, C_p_ours + 0.02)
)
# fig.savefig("/home/nzubic/map_vs_cp_gen1.pdf", bbox_inches="tight")


# gen4
C_p = [
    0.4091,
    0.43281647785577745,
    0.40314795761716137,
    0.4271,
    0.661960598186318,
    0.4129,
]
mAP = [
    [0.3870, 0.38078, 0.38813, 0.3832, 0.32945, 0.3864],
    [0.3705, 0.36511, 0.37234, 0.3672, 0.31235, 0.3691],
    [0.3478, 0.33983, 0.35010, 0.3421, 0.23181, 0.3458],
]

labels = ["EST", "Voxel Grid", "Time Surface", "MDES", "2D Histogram", "TORE"]

fig, ax = plot(
    C_p,
    methods,
    mAP,
    colors,
    labels,
    xlim=[0.20, 0.45],
    ylim=[0.39, 0.7],
    dataset_name="1 Mpx",
)
# fig.savefig("/home/nzubic/map_vs_cp_gen4.pdf", bbox_inches="tight")

plt.show()
