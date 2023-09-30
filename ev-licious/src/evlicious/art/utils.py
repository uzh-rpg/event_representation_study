import numpy as np
try:
    import open3d as o3d
except ImportError:
    print("Cannot import Open3d")
import pandas as pd
import plotly.express as px


def array_to_o3d_pts(pts, xyt, factor, color, t0):
    xyt = xyt.astype("float32")
    if t0 is not None:
        xyt[:, -1] -= t0

    xyt[:, -1] = xyt[:, -1] * factor
    pts.points = o3d.utility.Vector3dVector(xyt)
    colors = np.zeros_like(xyt).astype("float64")
    colors[:, 0] = color[0]
    colors[:, 1] = color[1]
    colors[:, 2] = color[2]
    pts.colors = o3d.utility.Vector3dVector(colors)


def events_array_to_front_projection_to_o3d_pts(pts, events, factor, num_events=10000, t0=None):
    x, y, t, p = events[-num_events:].T

    if t0 is not None:
        t = t - t0

    t = t * factor

    h = int(np.max(y) + 1)
    w = int(np.max(x) + 1)
    img = np.zeros((h, w))
    np.add.at(img, (y, x), p)
    y_nz, x_nz = np.nonzero(img)
    p = (img[y_nz, x_nz] > 0).astype("int")
    t_nz = np.full_like(y_nz, fill_value=t[-1])

    points = np.stack([x_nz, y_nz, t_nz], axis=-1)
    pts.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros_like(points).astype("float64")
    colors[np.arange(len(colors)), -p] = 0.8
    pts.colors = o3d.utility.Vector3dVector(colors)

def blend_images(image0, image1, alpha):
    H, W = image0.shape[:2]
    XX, _ = np.meshgrid(np.arange(W), np.arange(H))
    alpha = 1 / (1 + np.exp((XX-W//2)*alpha))
    a = alpha[...,None]
    return (image0.astype("float32") * a + image1.astype("float32") * (1-a)).astype('uint8')

def visualize_events_3d_plotly(events):
    num_samples = 8000
    random_indices = np.random.choice(events.t.size, size=num_samples, replace=False)

    x_subset = events.x[random_indices]
    y_subset = events.y[random_indices]
    pol_subset = (events.p[random_indices].astype("int32")+1)//2
    time_subset = events.t[random_indices]

    time_subset = time_subset - time_subset.min()
    time_subset = time_subset/time_subset.max()
    df_dict = {
        'x': x_subset,
        'y': y_subset,
        'p': pol_subset.astype("uint8"),
        't': time_subset
        }

    df = pd.DataFrame.from_dict(df_dict)

    fig = px.scatter_3d(df, x='x', y='y', z='t', color='p',color_continuous_scale='Inferno')
    fig.update_traces(marker=dict(size=1))
    fig.update_scenes(yaxis_autorange='reversed')
    fig.update_scenes(zaxis_autorange='reversed')

    ht = np.max(y_subset)+1
    wd = np.max(x_subset)+1
    fig.update_layout(
        scene=dict(
        xaxis=dict(range=[0, wd], showgrid=False, showticklabels=False),
        yaxis=dict(range=[0, ht], showgrid=False, showticklabels=False),
        zaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
        bgcolor='white'),
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        ),
        )

    fig.show()




def plot_y_slice(ax, events, y):
    events_temp = events[events.y == y]
    ax.scatter(events_temp.t, events_temp.x, s=1)
    ax.set_ylabel("X coordinate")
    ax.set_xlabel("Time [us]")


def plot_x_slice(ax, events, x):
    events_temp = events[events.x == x]
    ax.scatter(events_temp.t, events_temp.y, s=1)
    ax.set_ylabel("Y coordinate")
    ax.set_xlabel("Time [us]")

