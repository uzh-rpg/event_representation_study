try:
    import open3d as o3d
except ImportError:
    print("Cannot import Open3d")
import time
import numpy as np
from os.path import join
import os
import cv2
from matplotlib import cm
from functools import partial

import evlicious
from .utils import array_to_o3d_pts, events_array_to_front_projection_to_o3d_pts


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


def line_mesh(merged_mesh, points, lines, color, radius=0.15):
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    points = np.array(points)
    lines = np.array(
        lines) if lines is not None else lines_from_ordered_points(points)

    first_points = points[lines[:, 0], :]
    second_points = points[lines[:, 1], :]
    line_segments = second_points - first_points
    line_segments_unit, line_lengths = normalized(line_segments)

    cylinder_segments = []
    z_axis = np.array([0, 0, 1])
    # Create triangular mesh cylinder segments of line

    for i in range(line_segments_unit.shape[0]):
        line_segment = line_segments_unit[i, :]
        line_length = line_lengths[i]
        # get axis angle rotation to allign cylinder with line segment
        axis, angle = align_vector_to_another(z_axis, line_segment)
        # Get translation vector
        translation = first_points[i, :] + line_segment * line_length * 0.5
        # create cylinder and apply transformations
        cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
            radius, line_length)
        cylinder_segment = cylinder_segment.translate(
            translation, relative=False)
        if axis is not None:
            axis_a = axis * angle
            cylinder_segment = cylinder_segment.rotate(R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a),
                                                       center=cylinder_segment.get_center())
        # color cylinder
        cylinder_segment.paint_uniform_color(color)

        cylinder_segments.append(cylinder_segment)

    vertices_list = [np.asarray(mesh.vertices) for mesh in cylinder_segments]
    triangles_list = [np.asarray(mesh.triangles) for mesh in cylinder_segments]
    triangles_offset = np.cumsum([v.shape[0] for v in vertices_list])
    triangles_offset = np.insert(triangles_offset, 0, 0)[:-1]

    vertices = np.vstack(vertices_list)
    triangles = np.vstack([triangle + offset for triangle, offset in zip(triangles_list, triangles_offset)])

    merged_mesh.vertices = o3d.open3d.utility.Vector3dVector(vertices)
    merged_mesh.triangles = o3d.open3d.utility.Vector3iVector(triangles)

    merged_mesh.paint_uniform_color(color)

    return merged_mesh

def events_to_o3d_pts(pts, events, factor, color, t0=None):
    xyt = events.to_array(format="xyt").astype("int64")
    array_to_o3d_pts(pts, xyt, factor, color, t0)

def image_to_colored_o3d_pts(point_cloud, image, t, factor=1):
    t = t * factor
    image = np.ascontiguousarray(cv2.imread(image)[..., ::-1])
    image = np.clip(image.astype("int32")*2, 0, 255).astype("uint8")
    h, w = image.shape[:2]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(height=h, width=w, fx=t, fy=t, cx=0, cy=0)
    depth_map = np.full(fill_value=t, shape=(h, w), dtype="float32")
    z = o3d.geometry.Image(depth_map)
    image_o3d = o3d.geometry.Image(image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, z,
                                                              depth_scale=1,
                                                              depth_trunc=1e18,
                                                              convert_rgb_to_intensity=False)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    point_cloud.points = pc.points
    point_cloud.colors = pc.colors

def events_to_front_projection_to_o3d_pts(pts, events, factor, num_events=10000, t0=0):
    # project events onto an image
    xytp = events.to_array(format="xytp").astype("int64")
    events_array_to_front_projection_to_o3d_pts(pts, xytp, factor, num_events, t0=0)

def clear_point_cloud(point_cloud):
    point_cloud.points = o3d.utility.Vector3dVector(np.zeros((0,3)))
    point_cloud.colors = o3d.utility.Vector3dVector(np.zeros((0,3)))

def visualize_3d(events, images={}, tracks={}, factor=1,time_window_us=100000, time_step_us=1000, loop=False, output_path=""):
    def draw(vis, events, images, tracks, factor, time_window_us, colors, geometry_objects, t0, first):
        if len(images) > 0:
            for t_img, f in zip(images['timestamps'], images['files']):
                if t_img >= t0 and t_img <= t0 + time_window_us:
                    image_to_colored_o3d_pts(geometry_objects[f], f, t_img - t0, factor=factor)
                else:
                    clear_point_cloud(geometry_objects[f])

        # update events
        events_ = events[(events.t <= t0 + time_window_us) & (events.t >= t0)]
        positive_events, negative_events = evlicious.tools.split_into_positive_negative(events_)
        events_to_o3d_pts(geometry_objects['positive_event_pc'], positive_events, factor=factor, color=[0, 0, 1], t0=t0)
        events_to_o3d_pts(geometry_objects['negative_event_pc'], negative_events, factor=factor, color=[1, 0, 0], t0=t0)
        events_to_front_projection_to_o3d_pts(geometry_objects["front_projection"], events_, factor=factor, t0=t0)
        # add tracks
        for j, i in enumerate(sorted(tracks.keys())):
            tracks_ = tracks[i].copy()
            tracks_ = tracks_[(tracks_[:, -1] <= t0 + time_window_us) & (tracks_[:, -1] >= t0)]
            tracks_[:, -1] -= t0
            tracks_[:, -1] *= factor
            correspondences = [(i, i + 1) for i in range(len(tracks_) - 1)]
            line_mesh(geometry_objects[f'track_{i}'],
                      tracks_,
                      correspondences,
                      colors[j], radius=1)

        for v in geometry_objects.values():
            if first:
                vis.add_geometry(v, reset_bounding_box=True)
            else:
                vis.update_geometry(v)

    def run(vis, events, images, tracks, factor, time_window_us, time_step_us, output_path, colors, geometry_objects, t0):
        # if loop, slice objects and set the things
        counter = 0
        while t0 < events.t[-1]:
            draw(vis, events, images, tracks, factor, time_window_us, colors, geometry_objects, t0, False)

            vis.poll_events()
            vis.update_renderer()

            if loop:
                time.sleep(0.1)

                if output_path and counter > 0:
                    vis.capture_screen_image(join(output_path, "%05d.png" % counter))

                # add
                counter += 1

            t0 += time_step_us
            print(t0)

    colors = get_equally_spaced_colors(len(tracks))

    # setting all geometry objects
    geometry_objects = {}
    geometry_objects['positive_event_pc'] = o3d.geometry.PointCloud()
    geometry_objects['negative_event_pc'] = o3d.geometry.PointCloud()
    geometry_objects['front_projection'] = o3d.geometry.PointCloud()

    for i in tracks:
        geometry_objects[f"track_{i}"] = o3d.geometry.TriangleMesh()

    if len(images) > 0:
        mask = (images['timestamps'] < events.t[-1]) & (images['timestamps'] > events.t[0])
        images['timestamps'] = images['timestamps'][mask]
        images["files"] = [f for m, f in zip(mask, images['files']) if m]

        for f in images['files']:
            geometry_objects[f] = o3d.geometry.PointCloud()

    t0 = events.t[0]

    if not loop:
        time_window_us = events.t[-1] - events.t[0]
        time_step_us = 1

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    draw(vis, events=events, images=images, tracks=tracks, factor=factor,
         time_window_us=time_window_us, colors=colors, geometry_objects=geometry_objects,
         t0=t0, first=True)

    vis.register_key_callback(ord(" "), partial(run, events=events,
                                                images=images,
                                                tracks=tracks,
                                                factor=factor,
                                                time_window_us=time_window_us,
                                                time_step_us=time_step_us,
                                                output_path=output_path,
                                                colors=colors,
                                                geometry_objects=geometry_objects,
                                                t0=t0))
    vis.run()


def get_equally_spaced_colors(n_colors):
    return cm.rainbow(np.linspace(0, 1, n_colors))[:,:3]


class O3DVoxelGridVisualizer:
    def __init__(self, events: evlicious.io.H5EventHandle, factor: float):
        self.events = events
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.cube_vertices_and_lines = dict(points=np.array([[0, 0, 0],[1, 0, 0],[0, 1, 0],[1, 1, 0],
                                                             [0, 0, 1],[1, 0, 1],[0, 1, 1],[1, 1, 1]]),
                                            lines=np.array([[0, 1],[0, 2],[1, 3],[2, 3],[4, 5],[4, 6],
                                                            [5, 7],[6, 7],[0, 4],[1, 5],[2, 6],[3, 7]]))

        self.bins = self.events.height // 5

        self.subsample = 1

        print("Precomputing stuff...")
        voxel_points = np.stack(np.meshgrid(np.arange(self.bins), np.arange(events.height), np.arange(events.width), indexing="ij"), axis=-1)
        voxel_colors = np.ones(shape=(self.bins, self.events.height, self.events.width, 3))*.97
        self.voxel_points_colors = np.concatenate([voxel_points, voxel_colors], axis=-1)
        self.pts = o3d.geometry.PointCloud()

        b, h, w = self.bins, self.events.height, self.events.width
        b_c, h_c, w_c = voxel_points[...,0], voxel_points[...,1], voxel_points[...,2]
        inside = (0 < b_c) & (b_c < b-1) & (0 < h_c) & (h_c < h-1) & (0 < w_c) & (w_c < w-1)
        self.surface = ~inside
        print("Done!")

        self.vis.create_window()

    def empty_cube(self, xyz, hwd, color):
        points = self.cube_vertices_and_lines['points']
        lines = self.cube_vertices_and_lines['lines']

        points_shifted = np.array([xyz]) + np.array([hwd]) * points
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_shifted),
            lines=o3d.utility.Vector2iVector(lines),
        )
        colors = [color for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def full_cube(self, xyz, hwd, color):
        h, w, d = hwd
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)
        mesh_box.compute_vertex_normals()
        mesh_box.paint_uniform_color(color)
        mesh_box.translate(xyz, relative=True)
        return mesh_box

    def cube(self, xyz, hwd, face=False, color=[1,0,0]):
        if not face:
            return self.empty_cube(xyz, hwd, color)
        else:
            return self.full_cube(xyz, hwd, color)

    def draw_grid(self, height, width, bins):
        hwd = width+2, height+2, -bins-2

        points = self.cube_vertices_and_lines['points']
        lines = self.cube_vertices_and_lines['lines']

        points_shifted = (np.array([hwd])+2) * points
        points_shifted[:,:2] -= 1
        points_shifted[:,2] += 1

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_shifted),
            lines=o3d.utility.Vector2iVector(lines),

        )
        colors = [[0,0,0] for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(line_set, reset_bounding_box=False)

    def loop(self, output_path, n_events_window, n_events_step):

        vox = None
        counter = 0
        for events in self.events.iterator(window=n_events_window, step_size=n_events_step, window_unit="nr", step_size_unit="nr", pbar=True):
            if vox is not None:
                self.vis.remove_geometry(vox, reset_bounding_box=False)
            vox = self._update(events)
            reset_bounding_box = counter == 0
            self.vis.add_geometry(vox, reset_bounding_box=reset_bounding_box)
            self.draw_grid(self.events.height, self.events.width, self.bins)  # self.events.height)

            if counter == 0:
                ctr = self.vis.get_view_control()
                ctr.rotate(-1180.0, 1180.0)
                self.vis.poll_events()
                self.vis.update_renderer()
                self.vis.run()

            self.vis.poll_events()
            self.vis.update_renderer()

            if output_path and counter > 0:
                self.vis.capture_screen_image(str(output_path / ("%05d.png" % counter)))

            counter += 1

    def get_surface(self, grid):
        return grid[self.surface,:]

    def _update(self, events):
        voxel_grid = evlicious.tools.events_to_voxel_grid(events, num_bins=5, normalize=False)
        voxel_grid[-1] += voxel_grid.sum(0)
        voxel_grid[0] += voxel_grid.sum(0)
        num_bins = self.bins // 5
        voxel_grid = np.concatenate([np.stack([v]*num_bins) for v in voxel_grid],axis=0)
        b, y, x = np.nonzero(np.abs(voxel_grid))
        is_pos = voxel_grid[b, y, x] > 0

        self.voxel_points_colors[...,3:] = 1
        self.voxel_points_colors[b[is_pos], y[is_pos], x[is_pos], 3:] = [0,0,1]
        self.voxel_points_colors[b[~is_pos], y[~is_pos], x[~is_pos], 3:] = [1,0,0]

        only_surface = self.get_surface(self.voxel_points_colors)

        pts = only_surface[:,:3][:,::-1].copy()
        pts[:,-1] *=  -1
        self.pts.colors = o3d.utility.Vector3dVector(only_surface[:,3:])
        self.pts.points = o3d.utility.Vector3dVector(pts)

        return o3d.geometry.VoxelGrid.create_from_point_cloud(self.pts, voxel_size=1)



class O3DVisualizer:
    def __init__(self, events: evlicious.io.H5EventHandle, factor: float):
        self.paused = False
        self.events = events
        self.factor = factor

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

        self.vis.register_key_callback(32,  # space,
                                       lambda x,: self.loop(x))

        self.positive_events = o3d.geometry.PointCloud()
        self.negative_events = o3d.geometry.PointCloud()
        self.front_projection = o3d.geometry.PointCloud()

    def _update(self, vis, events, update_fn):
        t_curr = events.t[0]
        events.t -= t_curr

        positive_events, negative_events = evlicious.tools.split_into_positive_negative(events)
        events_to_o3d_pts(self.positive_events, positive_events, factor=self.factor, color=[0, 0, 1])
        events_to_o3d_pts(self.negative_events, negative_events, factor=self.factor, color=[1, 0, 0])

        events_to_front_projection_to_o3d_pts(self.front_projection, events, factor=self.factor)

        update_fn(self.positive_events)
        update_fn(self.negative_events)
        update_fn(self.front_projection)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.001)

    def loop(self, output_path="", t_window_us=10000, t_step_us=1000):
        counter = 0
        for events in self.events.iterator(t_step_us, t_window_us, "us", "us", pbar=True):
            update_fn = self.vis.add_geometry if counter == 0 else self.vis.update_geometry

            self._update(self.vis, events, update_fn)

            if counter == 0:
                ctr = self.vis.get_view_control()
                ctr.rotate(-1180.0, 1180.0)

            if output_path and counter > 0:
                self.vis.capture_screen_image(join(output_path, "%05d.png" % counter))

            counter += 1
