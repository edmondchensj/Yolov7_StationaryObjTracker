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
from datetime import datetime

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
SG_THEME_OK = "LightGreen"
SG_THEME_ALARM = "Reds"
BUTTON_CHECKING_TEXT = "I am checking on it"
BUTTON_FALSE_ALARM_TEXT = "False alarm"
BAG_UNATTENDED_STATUS = "Unattended"
BAG_CHECKING_STATUS = "Checking"
BAG_FALSE_ALARM_STATUS = "False Alarm"
STATUS_COLORS = {BAG_UNATTENDED_STATUS: [60,20,220], # bgr
                BAG_CHECKING_STATUS: [0,128,225],
                BAG_FALSE_ALARM_STATUS: [0,153,76],
                "Normal": [128,128,128]}

class GUI:
    ''' PySimpleGUI for Skytrain Unattended Bag Detection '''

    def __init__(self):
        print("[INFO] Initialising GUI")
        self.open_config_window()

        self.save_video = False # to refactor to checkbox
        self.win_started = False # whether main video playback window has started
        self.suppress_alarm = False # need to update logic for this. e.g. set timer or track the indexes of bags that are deemed false alarms. 
        self.unattended_bag_found = False
        self.unattended_bags = {} # store of all unattended bag statuses <id>:{"status": <status>, "first_trigger": <timestamp>}  pairs
        self.unattended_bags_on_screen = [] # store of bags that can be seen on screen

    def open_config_window(self):
        ''' Open window for user to set configurations '''
        print("[INFO] Opening config window")
        sg.theme(SG_THEME_OK)
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
        
        print(f"[INFO] Unattended bags state: {self.unattended_bags}")

        # Draw bbox on image
        if len(self.detections) > 0:
            for detection in self.detections:
                bboxes = [detection["bbox_x1"], detection["bbox_y1"], detection["bbox_x2"], detection["bbox_y2"]]
                id = detection["object_id"]
                cls = detection["object_class"]
                conf = detection["detection_conf"]
                unattended_flag = detection["unattended_bag"]

                # initialise data store
                if unattended_flag and id not in self.unattended_bags: 
                    print(f"[INFO] Saving new unattended bag {id} to data store")
                    self.unattended_bags[id] = {"status": BAG_UNATTENDED_STATUS, "first_trigger": datetime.fromtimestamp(detection["timestamp"])}                       

                # remove from store if somehow unattended flag is removed e.g. after some time, a new bag is tagged to the new index
                if not unattended_flag and id in self.unattended_bags:
                    del self.unattended_bags[id]

                # Add bbox to image
                if unattended_flag and cls in [24, 26, 28]: # backpack, suitcase, handbag
                    label = f"{self.unattended_bags[id]['status'].upper()} {id} {conf:.2f}"
                    color = STATUS_COLORS[self.unattended_bags[id]["status"]]
                else:
                    label = f'{id} {self.algo.model.names[cls]} {conf:.2f}'
                    color = STATUS_COLORS["Normal"]
                
                plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)

        self.imgbytes = cv2.imencode('.png', im0)[1].tobytes()
    
    def _create_playback_window(self, alarm=False):
        if alarm: 
            sg.ChangeLookAndFeel(SG_THEME_ALARM)
            self.alarm_state = True
        else:
            sg.ChangeLookAndFeel(SG_THEME_OK)
            self.alarm_state = False

        unattended_bag_alerts = [[
                    sg.Text(f"Bag ID: {id} | First detected {bag['first_trigger'].strftime('%d %b, %H:%M')}",
                        size=(40, 1)),
                    sg.Button(BUTTON_CHECKING_TEXT, key=f'_BUTTON_CHECKING_{id}', visible=alarm), 
                    sg.Button(BUTTON_FALSE_ALARM_TEXT, key=f'_FALSE_ALARM_{id}', visible=alarm)] 
                        for id, bag in self.unattended_bags.items() if \
                        id in self.detected_bags and bag["status"] in [BAG_UNATTENDED_STATUS, BAG_CHECKING_STATUS]]
        
        print("unattended bag alerts: ", unattended_bag_alerts)
        layout = [[sg.Text('Yolo Playback in PySimpleGUI Window', size=(30,1))],
                [sg.Image(data=self.imgbytes, key='_IMAGE_')]]

        for alert in unattended_bag_alerts:
            layout.append(alert)

        layout.append([sg.Exit()])

        self.win = sg.Window('YOLO Output', layout,
                        default_element_size=(14, 1),
                        element_justification="c",
                        text_justification='right',
                        auto_size_text=False, finalize=True)    
        self.win_image_elem = self.win['_IMAGE_']

    def update_playback_window(self, im0):
        ''' Update playback window showing detection'''

        self.detected_bags = [detection["object_id"] for detection in self.detections]
        self.draw_bboxes(im0)

        if not self.win_started:
            # Create new window
            self.win_started = True
            self._create_playback_window()

        else:
            # New uattended bag found 
            unattended_bags_on_screen = [bag_id for bag_id, bag in self.unattended_bags.items() if bag_id in self.detected_bags and bag["status"] in [BAG_UNATTENDED_STATUS, BAG_CHECKING_STATUS]]

            if sorted(unattended_bags_on_screen)==sorted(self.unattended_bags_on_screen) \
                and self.alarm_state: 
                # Already in alarm state and unattended bags are the same, so we just need to refresh the image
                self.win_image_elem.Update(data=self.imgbytes)

            elif len(unattended_bags_on_screen) > 0:
                # Previously in normal state, so we change to alarm layout
                print("[INFO] Unattended bag found and updating window layout ")
                self.win.Close()
                self._create_playback_window(alarm=True)
                self.unattended_bags_on_screen = unattended_bags_on_screen

            elif self.alarm_state: 
                # no unattended bags on screen, but previous frame was alarm state so we need to update the layout
                self.win.Close()
                self._create_playback_window()

            else: # no unattended bags on screen and previous frame is in normal state
                self.win_image_elem.Update(data=self.imgbytes)

        # self._save_video() # TO IMPLEMENT

        # Close window if frames are completed or if user exits
        event, values = self.win.read(timeout=0)
        print("[INFO] UI Click Event: ", event)
        if event is None or event == 'Exit':
            return False
        elif "_BUTTON_CHECKING_" in event:
            bag_id = event.split("_")[-1]
            self.unattended_bags[int(bag_id)]["status"] = BAG_CHECKING_STATUS
        elif "_FALSE_ALARM_" in event:
            bag_id = event.split("_")[-1]
            self.unattended_bags[int(bag_id)]["status"] = BAG_FALSE_ALARM_STATUS
        
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
        ''' Loop over frames from video file stream and run algorithm 
        
        The Algo class handles the logic for detection and tracking unattended bag. It must contain the following two methods:

        1. __init__
            Keyword arguments:
                yolo_weights 
                yolo_conf
                strong_sort_weights

        2. run
            Args:
                im: image tensor that has been resized/padded/transposed to RGB
                im0: original image read by cv2.imread()

            Ret:
                detections: List of dicts. Each dict represents a bag detection: 
                    [ 
                        {
                            "bbox_x1": int,
                            "bbox_y1": int,
                            "bbox_x2": int,
                            "bbox_y2": int,
                            "object_id": int,           # ReID identifier
                            "object_class": int,        # coco class,
                            "detection_conf": float,    # YOLO confidence,
                            "timestamp": string,        # UTC timestamp of detection
                            "unattended_bag": bool      # whether bag is unattended
                        }
                    ]

        TODOS:
            - (y) refactor detections from list to dict. in both algo class and GUI class. 

            - (y) global state in GUI class for bag status
                - global state in GUI class to store status of bag (unattended, checking, false alarm) | only for bags that have been flagged as unattended

            - update Algo class. 
                - pass in optional config value for unattended threshold (e.g. 5min)
                - (y) refactor algo logic to use ts for unattended logic (done)
                - (y) return ts in detection list (done)
                
            - (y) update UI to show list of unattended bags 

            - (y) add logic for handling user input
                - (y) when btn checking on it clicked: 
                    - update data store of bag to "checking", change bbox title to "CHECKING"
                - (y) when false alarm clicked: 
                    - update data store to false alarm, change bbox title to false alarm (this will lead to ever increasing data store)
                    - change theme back to normal
                - (y) new btn for actual unattended bag + cleared: 
                    - not needed -- bag will just not show up in self.detections 

            - add logging
                - write ts, image, action to log file 

            - update UI styling (make buttons larger, centered)

            - add feature for two videos streams in same window

            - discuss with andrej return values

        '''
        self.algo = Algo(yolo_weights=self.yolo_weights, 
                        yolo_conf=self.yolo_conf, 
                        strong_sort_weights=self.strong_sort_weights)
        
        self.load_video_stream()

        for frame_idx, (path, im, im0, vid_cap) in enumerate(self.dataset): # loop over frames

            self.detections = self.algo.run(im, im0) # Main Algo function here
            
            has_next_frame = self.update_playback_window(im0)
            if not has_next_frame:
                break

        self.win.Close()

if __name__ == "__main__":
    gui = GUI()
    gui.run()