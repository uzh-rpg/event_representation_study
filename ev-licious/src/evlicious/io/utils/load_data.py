import numpy as np
import os
import glob


def load_images(path, downsample_factor):
    if path == "":
        return {}
    image_files = sorted(glob.glob(os.path.join(path, "*.jpg")))
    image_timestamps = np.genfromtxt(os.path.join(path, "timestamps.txt"))
    mean_dt = np.diff(image_timestamps).mean()
    if mean_dt < 1: # is s
        image_timestamps *= 1e6
    if mean_dt > 1e5: # is ns
        image_timestamps /= 1e3
    return dict(files=image_files[::downsample_factor],
                timestamps=image_timestamps[::downsample_factor])

def load_feature_tracks(path):
    if path == "":
        return {}
    tracks = np.genfromtxt(path, delimiter=",", dtype="float64")
    tracks_data = {i: tracks[tracks[:, 0] == i][:, [2, 3, 1]] for i in np.unique(tracks[:, 0])}
    for i in tracks_data:
        tracks_data[i][:, 2] *= 1e6
    return {i: track for i, track in tracks_data.items() if len(track) > 2}