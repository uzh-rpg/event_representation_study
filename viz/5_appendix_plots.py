import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = "16"
plt.rcParams["lines.linewidth"] = 1


def plot(C_p, methods, mAP, colors, labels, xlim, ylim, dataset_name="Gen1"):
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, key in enumerate(methods):
        ax.scatter(mAP[i], C_p, c=colors[i], label=key)

    ax.set_ylabel(r"GWD$_{100}$ [-]")
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
        if label.startswith("ERGO"):
            ax.annotate(
                text=rf"\textbf{{{label}}}", fontsize=14, xy=(mAP[0][idx], C_p[idx])
            )
            ax.scatter(mAP[0][idx], C_p[idx], c="r", marker="*", s=200)
        else:
            ax.annotate(text=label, fontsize=14, xy=(mAP[0][idx], C_p[idx]))

    return fig, ax


# gen1
markers = ["*", "^", "s"]
colors = ["r", "g", "b"]
methods = ["Swin-V2"]
labels = ["Voxel Grid", "MDES", "Time Surface", "TORE", "ERGO-12", "ERGO-9", "ERGO-7"]

C_p = [
    0.517637696535689,
    0.497969176297073,
    0.4725212186799115,
    0.4842212186799116,
    0.46558974599468906,
    0.49039721437555236,
    0.50803210774350944,
]
mAP = [[0.4249, 0.4375, 0.5007, 0.4465, 0.519, 0.4400, 0.4300]]

# fig, ax = plot(C_p, methods, mAP, colors, labels, xlim=[0.25, 0.55], ylim=[0.28, 0.65], dataset_name="Gen1")

# uncomment this one only if you want to plot without our method
fig, ax = plot(
    C_p,
    methods,
    mAP,
    colors,
    labels,
    xlim=[0.42, 0.54],
    ylim=[0.46, 0.525],
    dataset_name="Gen1",
)
fig.savefig("/home/nzubic/gen_one_channels_supp.pdf", bbox_inches="tight")

# gen4
C_p = [
    0.517637696535689,
    0.4725212186799115,
    0.497969176297073,
    0.4842212186799116,
    0.46558974599468906,
    0.49039721437555236,
    0.50803210774350944,
]
mAP = [[0.38078, 0.38813, 0.3832, 0.3864, 0.419, 0.3850, 0.3815]]

labels = ["Voxel Grid", "Time Surface", "MDES", "TORE", "ERGO-12", "ERGO-9", "ERGO-7"]

fig, ax = plot(
    C_p,
    methods,
    mAP,
    colors,
    labels,
    xlim=[0.38, 0.425],
    ylim=[0.46, 0.525],
    dataset_name="1 Mpx",
)
fig.savefig("/home/nzubic/one_mpx_channels_supp.pdf", bbox_inches="tight")

plt.show()
