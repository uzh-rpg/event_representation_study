import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = "16"
plt.rcParams["lines.linewidth"] = 1


def window_to_interval(w):
    return f"[t_{{{w[0]}}}, t_{{{w[1]}}}]"


def plot_cp_overtime(C_p, methods, optimization_results):
    C_p_over_time = [o["C_p"] for o in optimization_results]

    codes = [
        (rf"$p_{{{c+1}}}=$(${o['window']}, {o['function']},$ {o['aggregation']})")
        for c, o in enumerate(optimization_results)
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(C_p_over_time, color="b", marker="*")
    ax.set_xlim([0, 13])
    ax.set_ylim([0.4, 0.85])

    ax.set_xticks(range(len(C_p_over_time)), labels=range(1, 1 + len(C_p_over_time)))
    ax.hlines(C_p, xmin=0, xmax=11, linestyles="dashed", color="gray")

    for cp, method in zip(C_p, methods):
        ax.annotate(method, xy=(11.3, cp - 0.007), fontsize=14)

    ax.scatter(
        [len(C_p_over_time) - 1], [C_p_over_time[-1]], marker="*", s=200, color="b"
    )
    ax.annotate(
        r"\textbf{ERGO-12 (ours)}", xy=(10.6, C_p_over_time[-1] - 0.037), fontsize=14
    )

    for i, c in enumerate(codes):
        ax.annotate(c, xy=(1, 0.8 - 0.035 * i), fontsize=14)

    ax.set_xlabel("Num. Optimized Channels")
    ax.set_ylabel(r"GWD$_{100}$ [-]")

    return fig, ax


windows = [
    ["0", "N_e"],
    ["0", "N_e/3"],
    ["N_e/3", "2N_e/3"],
    ["2N_e/3", "N_e"],
    ["N_e/2", "N_e"],
    ["3N_e/4", "N_e"],
    ["7N_e/8", "N_e"],
]
reordering = [3, 0, 1, 2, 4, 5, 6]

optimization_results = [
    {"window": 0, "function": "t", "aggregation": "max", "C_p": 0.80048190998512850},
    {"window": 2, "function": "t_+", "aggregation": "sum", "C_p": 0.73751093403808926},
    {"window": 2, "function": "t_-", "aggregation": "mean", "C_p": 0.63572257612309653},
    {"window": 3, "function": "c_-", "aggregation": "sum", "C_p": 0.60695910889314162},
    {"window": 5, "function": "c_+", "aggregation": "mean", "C_p": 0.57928593984953931},
    {"window": 0, "function": "p", "aggregation": "var", "C_p": 0.55207947660282985},
    {"window": 0, "function": "t", "aggregation": "var", "C_p": 0.50803210774350944},
    {"window": 4, "function": "c", "aggregation": "sum", "C_p": 0.49555546143018796},
    {"window": 2, "function": "t_+", "aggregation": "mean", "C_p": 0.49039721437555236},
    {"window": 6, "function": "c", "aggregation": "sum", "C_p": 0.48523002054096161},
    {"window": 1, "function": "t_+", "aggregation": "sum", "C_p": 0.47172468481132535},
    {"window": 1, "function": "t_-", "aggregation": "sum", "C_p": 0.46558974599468906},
]

for o in optimization_results:
    o["window"] = f"w_{reordering[o['window']]}"


labels = ["Voxel Grid", "MDES", "Time Surface", "2D Histogram", "TORE"]
C_p = [
    0.517637696535689,
    0.497969176297073,
    0.4725212186799115,
    0.7567818168662296,
    0.4842212186799116,
]

fig, ax = plot_cp_overtime(C_p, labels, optimization_results)
fig.savefig("/home/nzubic/works.pdf", bbox_inches="tight")

plt.show()
