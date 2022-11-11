''' 
A GUI to run the algorithm for unattended bags.
Uses PySimpleGUI.

USAGE
    python gui.py 
'''

import numpy as np
import argparse
import time
import cv2
import os
import PySimpleGUI as sg
from skytrain_algo import Algo
from pathlib import Path
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.plots import plot_one_box
sg.theme('LightGreen')

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

class GUI:
    ''' PySimpleGUI for Skytrain Unattended Bag Detection '''

    def __init__(self):
        print("[INFO] Initialising GUI")
        self.open_config_window()

        self.save_video = False # to refactor to checkbox
        self.show_video = True
        self.win_started = False # whether main video playback window has started
        
    def open_config_window(self):
        ''' Open window for user to set configurations '''
        print("[INFO] Opening config window")
        # Setup window
        layout = 	[
                    [sg.Text('YOLO Video Player', size=(18,1), font=('Any',18),text_color='#1c86ee' ,justification='left')],
                    [sg.Text('Path to input video'), sg.In("",size=(40,1), key='input'), sg.FileBrowse()],
                    [sg.Text('Optional Path to output video'), sg.In("output.mp4",size=(40,1), key='output'), sg.FileSaveAs()],
                    [sg.Text('YOLO weights'), sg.In("yolov7.pt",size=(40,1), key='yolo-weights')],
                    [sg.Text('StrongSORT weights'), sg.In("osnet_ain_x1_0_msmt17.pt",size=(40,1), key='strong-sort-weights')],
                    [sg.Text('Detection Confidence (YOLO)'), sg.Slider(range=(0,1),orientation='h', resolution=.1, default_value=.1, size=(15,15), key='detection-confidence')],
                    [sg.OK(), sg.Cancel()]
                        ]
        win = sg.Window('YOLO Video',
                        default_element_size=(21,1),
                        text_justification='right',
                        auto_size_text=False).Layout(layout)
        
        # Read values from user input
        event, self.args = win.read()
        if event is None or event =='Cancel':
            exit()
        else:
            self.yolo_conf = self.args["detection-confidence"]
            self.yolo_weights = self.args["yolo-weights"]
            self.strong_sort_weights = self.args["strong-sort-weights"]
        win.Close()

    def load_video_stream(self):
        ''' Use in-built YoloV7 loaders to load the source video into a stream of images '''
        # initialize the video stream, pointer to output video file, and frame dimensions
        # self.vs = cv2.VideoCapture(args["input"])
        # writer = None
        # (W, H) = (None, None)
        print("[INFO] Loading video stream")

        self.source = self.args["input"]
        is_file = Path(self.source).suffix[1:] in (VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            self.source = check_file(self.source)  # download

        stride = self.algo.model.stride.max()  # model stride
        imgsz = (640, 640)
        imgsz = check_img_size(imgsz[0], s=stride.cpu().numpy())  # check image size

        # Dataloader
        if self.webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.source, img_size=imgsz, stride=stride.cpu().numpy())
        else:
            self.dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
        self.vid_path, self.vid_writer, self.txt_path = [None], [None], [None]

    def draw_bboxes(self, im0):
        ''' Takes in list of detections and frame and draws bounding boxes ''' 
    
        # draw boxes for visualization
        if len(self.detections) > 0:
            for output, conf in zip(self.detections, self.detection_confs):
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                if self.save_video or self.show_video:  # Add bbox to image
                    c = int(cls)  # integer class
                    id = int(id)  # integer id
                    unattended_flag = output[7]

                    if unattended_flag and c in [24, 26, 28]: # backpack, suitcase, handbag
                        label = f"UNATTENDED BAG {id} {conf:.2f}"
                        color = [60,20,220]
                    else:
                        label = f'{id} {self.algo.model.names[c]} {conf:.2f}'
                        color = self.algo.COLORS[int(cls)]
                    
                    plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)

        self.imgbytes = cv2.imencode('.png', im0)[1].tobytes()  # ditto

    def update_playback_window(self, im0):
        ''' Update playback window showing detection'''

        self.draw_bboxes(im0)

        if not self.win_started:
            # Create new window
            self.win_started = True
            layout = [
                [sg.Text('Yolo Playback in PySimpleGUI Window', size=(30,1))],
                [sg.Image(data=self.imgbytes, key='_IMAGE_')],
                [sg.Text('Confidence')],
                [sg.Slider(range=(0, 1), orientation='h', resolution=.1, default_value=.1, size=(15, 15), key='detection-confidence')],
                [sg.Exit()]
            ]
            self.win = sg.Window('YOLO Output', layout,
                            default_element_size=(14, 1),
                            text_justification='right',
                            auto_size_text=False, finalize=True)
            self.image_elem = self.win['_IMAGE_']
        else:
            # Update existing window
            self.image_elem.Update(data=self.imgbytes)

        # self._save_video() # TO IMPLEMENT

        # Close window if frames are completed or if user exits
        event, values = self.win.read(timeout=0)
        if event is None or event == 'Exit':
            return False
        else:
            if self.yolo_conf != values["detection-confidence"]:
                self.yolo_conf = values["detection-confidence"]
                print(f"Yolo detection confidence updated to {self.yolo_conf}")
            self.algo.yolo_conf = self.yolo_conf
            return True

    def write_video(self, im0):
        ''' Save video to local file '''
        # Save results (image with detections) [TO UPDATE]
        # if self.save_video:
        #     if vid_path[i] != save_path:  # new video
        #         vid_path[i] = save_path
        #         if isinstance(vid_writer[i], cv2.VideoWriter):
        #             vid_writer[i].release()  # release previous video writer
        #         if vid_cap:  # video
        #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #         else:  # stream
        #             fps, w, h = 30, im0.shape[1], im0.shape[0]
        #         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        #         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #     vid_writer[i].write(im0)

        # Save results (image with detections)
        if vid_path[i] != save_path:  # new video
            vid_path[i] = save_path
            if isinstance(vid_writer[i], cv2.VideoWriter):
                vid_writer[i].release()  # release previous video writer
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, im0.shape[1], im0.shape[0]
            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer[i].write(im0)

    def run(self):
        ''' Loop over frames from video file stream and run algorithm '''
        self.algo = Algo(yolo_weights=self.yolo_weights, 
                    yolo_conf=self.yolo_conf, 
                    strong_sort_weights=self.strong_sort_weights)
        
        self.load_video_stream()

        for frame_idx, (path, im, im0, vid_cap) in enumerate(self.dataset): # loop over frames

            self.detections, self.detection_confs = self.algo.run(im, im0) # Main Algo function here

            has_next_frame = self.update_playback_window(im0)
            if not has_next_frame:
                break

        self.win.Close()

if __name__ == "__main__":
    gui = GUI()
    gui.run()