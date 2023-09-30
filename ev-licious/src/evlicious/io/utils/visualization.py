import cv2
import numpy as np
import os
import tqdm
import skvideo.io

from .render import RenderingType


class Visualizer:
    def __init__(self, event_handle,
                 step_size: int = 10000,
                 window: int = 20000,
                 step_size_unit: str = "us",
                 window_unit: str = "us",
                 rendering_type: RenderingType = RenderingType.RED_BLUE_NO_OVERLAP):
        self.event_handle = event_handle

        self.window = window
        self.step_size = step_size
        self.window_unit = window_unit
        self.step_size_unit = step_size_unit
        self.rendering_type = rendering_type
        self.time_between_screen_refresh_ms = 5

        self.is_paused = False
        self.is_looped = False

        self.step = 0
        self.cv2_window_name = 'Events'
        cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_NORMAL)

        self.time_windows = None
        self.index = None
        self.update_indices()

    def update_indices(self):
        t_current = None if self.time_windows is None else self.time_windows[self.step, 1]

        (t0, t1), (idx0, idx1) = self.event_handle.compute_time_and_index_windows(self.step_size, self.window, self.step_size_unit, self.window_unit)
        self.index = np.stack([idx0, idx1], axis=-1)
        self.time_windows = np.stack([t0, t1], axis=-1)

        if t_current is not None:
            step = np.searchsorted(self.time_windows[:,1], t_current)
            self.step = np.clip(step, 0, len(self.index)-1)

    def print_help(self):
        print("##################################")
        print("#     interactive visualizer     #")
        print("#                                #")
        print("#     a: backward                #")
        print("#     d: forward                 #")
        print("#     w: jump to end             #")
        print("#     s: jump to front           #")
        print("#     e: lengthen time window    #")
        print("#     q: shorten time window     #")
        print("#     c: lengthen step size      #")
        print("#     z: shorten step size       #")
        print("#     x: cycle color scheme      #")
        print("#   esc: quit                    #")
        print("# space: pause                   #")
        print("#     h: print help              #")
        print("##################################")

    def cycle_colors(self):
        self.rendering_type = (self.rendering_type % len(RenderingType)) + 1
        print("New Rendering Type: ", self.rendering_type)

    def pause(self):
        self.is_paused = True

    def unpause(self):
        self.is_paused= False

    def togglePause(self):
        self.is_paused = not self.is_paused

    def toggleLoop(self):
        self.is_looped = not self.is_looped

    def forward(self, num_timesteps = 1):
        if self.is_looped:
            self.step = (self.step + 1) % len(self.index)
        else:
            self.step = min(self.step + num_timesteps, len(self.index) - 1)

    def backward(self, num_timesteps = 1):
        self.step = max(self.step - num_timesteps, 0)

    def goToBegin(self):
        self.step = 0

    def goToEnd(self):
        self.step = len(self.index) - 1

    def visualizationLoop(self):
        while True:
            self.image = self.update(self.step)
            cv2.imshow(self.cv2_window_name, self.image)

            if not self.is_paused:
                self.forward(1)

            c = cv2.waitKey(self.time_between_screen_refresh_ms)
            key = chr(c & 255)

            if c == 27:             # 'q' or 'Esc': Quit
                break
            elif key == 'r':                      # 'r': Reset
                self.goToBegin()
                self.unpause()
            elif key == 'p' or c == 32:           # 'p' or 'Space': Toggle play/pause
                self.togglePause()
            elif key == "a":                         # 'Left arrow': Go backward
                self.backward(1)
                self.pause()
            elif key == "d":                         # 'Right arrow': Go forward
                self.forward(1)
                self.pause()
            elif key == "s":                         # 'Down arrow': Go to beginning
                self.goToBegin()
                self.pause()
            elif key == "v":
                output_path = input('enter output path: ')
                self.convert_to_video(output_path)
            elif key == "w":                         # 'Up arrow': Go to end
                self.goToEnd()
                self.pause()
            elif key == 'l':                      # 'l': Toggle looping
                self.toggleLoop()
            elif key == "e":
                self.update_window(1.2)
            elif key == "q":
                self.update_window(1/1.2)
            elif key == "z":
                self.update_step_size(1/1.2)
            elif key == "c":
                self.update_step_size(1.2)
            elif key == "x":
                self.cycle_colors()

        cv2.destroyAllWindows()

    def convert_to_video(self, output_path):
        writer = skvideo.io.FFmpegWriter(output_path)
        for step in tqdm.tqdm(range(len(self.index))):
            image = self.update(step, put_text=False)
            if image.dtype == np.dtype("float64"):
                image = np.clip(image[...,:3] * 255, 0, 255).astype("uint8")
            writer.writeFrame(image)
        writer.close()

    def update_step_size(self, factor):
        self.step_size = int(self.step_size*factor)
        self.update_indices()

    def update_window(self, factor):
        self.window = int(self.window * factor)
        self.update_indices()

    def update(self, index, put_text=True):
        i0, i1 = self.index[index]
        events = self.event_handle.get_between_idx(i0, i1)
        image = events.render(None, self.rendering_type)

        height, width = image.shape[:2]
        fixed_height = 1000
        new_size = (int(width * fixed_height / height), fixed_height)
        image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_NEAREST)

        if put_text:
            t = events.t[-1]

            scale = 1
            pos_1 = (int(scale*10),int(scale*30))
            pos_2 = (int(scale*10),int(scale*60))
            pos_3 = (int(scale*10),int(scale*90))
            th = int(np.ceil(scale*3))

            cv2.putText(image, f"t={t} us", pos_1, cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, thickness=th, color=(0,0,0))
            cv2.putText(image, f"step_size={self.step_size} {self.step_size_unit}", pos_2, cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, thickness=th, color=(0,0,0))
            cv2.putText(image, f"window={self.window} {self.window_unit}", pos_3, cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, thickness=th, color=(0,0,0))

        return image