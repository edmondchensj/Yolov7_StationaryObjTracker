''' 
A GUI to run the algorithm for CAG Skytrain unattended bags.
Uses PySimpleGUI.
'''
import logging
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
import sys

# To move these constants to separate file
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
SG_THEME_OK = "LightGreen"
SG_THEME_ALARM = "Reds"
COCO_BAG_CLASSES = [24, 26, 28] # backpack, suitcase, handbag
BUTTON_CHECKING_TEXT = "I am checking on it"
BUTTON_FALSE_ALARM_TEXT = "False alarm"
BAG_UNATTENDED_STATUS = "Unattended"
BAG_CHECKING_STATUS = "Checking"
BAG_FALSE_ALARM_STATUS = "False Alarm"
STATUS_COLORS = {BAG_UNATTENDED_STATUS: [60,20,220], # bgr
                BAG_CHECKING_STATUS: [0,128,225],
                BAG_FALSE_ALARM_STATUS: [0,153,76],
                "Normal": [128,128,128]}
GUI_TITLE = 'Skytrain Unattended Bag Detection System'
GUI_FONT = ("Noto Sans", 13)
GUI_FONT_LARGE = ("Helvetica", 15)

class GUI:
    ''' Graphical User Interface for CAG Skytrain Unattended Bag Detection System
    
    Dependencies: 
        1. Set up an Algo class in the same directory. The Algo class must have an __init__ method and a "run" method. 
            > You will need to replace the import statement above ('from skytrain_algo import Algo') to import your custom Algo class e.g. 'from my_custom_script import Algo'
        2. Install dependencies by running `pip install -r requirements.txt`
        3. Clone the YoloV7 repo to make use of the YoloV7 utils

    Usage:
        Run `python gui.py`

    Logging: 
        i. Logs are saved to ./logs/txt/ folder.
        ii. Annotated images with bounding boxes are saved to the /logs/images folder in three cases:
            - when a bag is first detected 
            - when operator clicks on "checking" button
            - when operator clicks on "false alarm" button
        
    '''

    def __init__(self):
        self.create_log_folders()
        self.open_config_window()
        self.win_started = False                # whether main video playback window has started 
        self.unattended_bags = {}               # store of all unattended bag statuses <id>:{"status": <status>, "first_trigger": <timestamp>}  pairs
        self.unattended_bags_on_screen = []     # store of unattended bags that are currently shown on screen

    def create_log_folders(self):
        ''' Create a log folders for text and annotated images '''
        current_directory = os.getcwd()
        self.log_directory = os.path.join(current_directory, 'logs', 'images')
        self.log_directory_txt = os.path.join(current_directory, 'logs', 'txt')
        os.makedirs(self.log_directory, exist_ok=True)
        os.makedirs(self.log_directory_txt, exist_ok=True)

        # Setup log statements        
        now = datetime.now().strftime("%Y-%m-%d_%H%M") 
        logging.basicConfig(filename=os.path.join(self.log_directory_txt, f'{now}.log'),
                    filemode='w',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')                            # save logs to file
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))   # print logs to console

    def open_config_window(self):
        ''' Open window for user to set configurations '''
        logging.info("Opening config window")
        sg.theme(SG_THEME_OK)
        layout = 	[
                    [sg.Text('Settings', size=(18,1), font=('Helvetica',18),text_color='#1c86ee' ,justification='left')],
                    [sg.Text('Path to input video / stream', size=(30,1)), sg.In("",size=(40,1), key='input'), sg.FileBrowse()],
                    #[sg.Text('Optional path to output video'), sg.In("output.mp4",size=(40,1), key='output'), sg.FileSaveAs()],
                    [sg.Text('YOLO weights', size=(30,1)), sg.In("yolov7.pt",size=(40,1), key='yolo-weights')],
                    [sg.Text('ReID weights', size=(30,1)), sg.In("osnet_ain_x1_0_msmt17.pt",size=(40,1), key='reid-weights')],
                    [sg.Text('YOLO Detection Confidence', size=(30,1)), sg.Slider(range=(0,1), orientation='h', resolution=.05, default_value=.1, size=(15,15), key='detection-confidence')],
                    [sg.Text('Unattended Bag Threshold (seconds)', size=(30,1)), sg.Slider(range=(0,1000), orientation='h', resolution=10, default_value=300, size=(15,15), key='unattended-threshold')],
                    [sg.Checkbox('Hide regular bag detections', size=(30,1), default=False, key='hide-regular-detections')],
                    [sg.OK(), sg.Cancel()]
                        ]
        win = sg.Window(GUI_TITLE,
                        default_element_size=(21,1),
                        text_justification='right',
                        auto_size_text=False,
                        font=GUI_FONT).Layout(layout)
        
        # Read values from user input
        event, self.args = win.read()
        if event is None or event =='Cancel':
            exit()
        else:
            self.yolo_conf = self.args["detection-confidence"]
            self.yolo_weights = self.args["yolo-weights"]
            self.reid_weights = self.args["reid-weights"]
            self.unattended_threshold_seconds = self.args['unattended-threshold']
            self.hide_regular_detections = self.args['hide-regular-detections']
        win.Close()

    def load_video_stream(self):
        ''' Use in-built YoloV7 loaders to load the source video into a stream of images '''
        logging.info("Loading video stream")

        # Get source
        self.source = self.args["input"]
        is_file = Path(self.source).suffix[1:] in (VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            self.source = check_file(self.source)  # download

        # Get model stride and img size
        stride = self.algo.model.stride.max() 
        imgsz = (640, 640)
        imgsz = check_img_size(imgsz[0], s=stride.cpu().numpy())  

        # Dataloader
        if self.webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.source, img_size=imgsz, stride=stride.cpu().numpy())
        else:
            self.dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
        self.vid_path, self.vid_writer, self.txt_path = [None], [None], [None]

    def save_image(self, image, object_id, action):
        ''' Save an annotated image to the logs folder.
        File name is saved as current timestamp '''
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{object_id}_{action}.jpg"
        cv2.imwrite(os.path.join(self.log_directory, filename), image)
        logging.info(f"Saved image ({filename}) to log folder")

    def draw_bboxes(self, im0):
        ''' Takes in list of detections and frame and draws bounding boxes ''' 
        
        #print(f"[INFO] Unattended bags state: {self.unattended_bags}")

        # Draw bbox on image
        if len(self.detections) > 0:
            for detection in self.detections:
                is_first_detection = False

                # Parse data
                bboxes = [detection["bbox_x1"], detection["bbox_y1"], detection["bbox_x2"], detection["bbox_y2"]]
                id = detection["object_id"]
                cls = detection["object_class"]
                conf = detection["detection_conf"]
                unattended_flag = detection["unattended_bag"]
                detected_ts = datetime.fromtimestamp(detection["timestamp"])

                # initialise data store
                if unattended_flag and id not in self.unattended_bags: 
                    logging.info(f"Bag {id}: First unattended bag detection at {detected_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                    self.unattended_bags[id] = {"status": BAG_UNATTENDED_STATUS, "first_trigger": detected_ts}    
                    is_first_detection = True                   

                # remove from store if somehow unattended flag is removed e.g. after some time, a new bag is tagged to the new index
                if not unattended_flag and id in self.unattended_bags:
                    del self.unattended_bags[id]

                # Add bbox for unattended bags
                if unattended_flag and cls in COCO_BAG_CLASSES:
                    label = f"{self.unattended_bags[id]['status'].upper()} {id} {conf:.2f}"
                    color = STATUS_COLORS[self.unattended_bags[id]["status"]]

                elif self.hide_regular_detections:
                    continue

                # Add bbox for regular bags if needed
                else:
                    label = f'{id} {self.algo.model.names[cls]} {conf:.2f}'
                    color = STATUS_COLORS["Normal"]
                
                plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)

                if is_first_detection:
                    self.save_image(im0, id, "first-detection")

        self.imgbytes = cv2.imencode('.png', im0)[1].tobytes()
        self.annotated_image = im0
    
    def _create_playback_window(self, alarm=False):
        # Update theme
        if alarm: 
            sg.ChangeLookAndFeel(SG_THEME_ALARM)
            self.alarm_state = True
        else:
            sg.ChangeLookAndFeel(SG_THEME_OK)
            self.alarm_state = False

        # Set layout
        unattended_bag_alerts = [[
                    sg.Text(f"Bag ID: {id} | First detected {bag['first_trigger'].strftime('%d %b, %H:%M')}",
                        size=(40, 1), font=GUI_FONT_LARGE),
                    sg.Button(BUTTON_CHECKING_TEXT, key=f'_BUTTON_CHECKING_{id}', visible=alarm, size=(15,1), font=GUI_FONT_LARGE), 
                    sg.Button(BUTTON_FALSE_ALARM_TEXT, key=f'_FALSE_ALARM_{id}', visible=alarm, size=(15,1), font=GUI_FONT_LARGE)] 
                        for id, bag in self.unattended_bags.items() if \
                        id in self.detected_bags and bag["status"] in [BAG_UNATTENDED_STATUS, BAG_CHECKING_STATUS]]
        
        layout = [[sg.Text(GUI_TITLE+" [LIVE]", size=(50,1))],
                [sg.Image(data=self.imgbytes, key='_IMAGE_')]]

        for alert in unattended_bag_alerts:
            layout.append(alert)

        layout.append([sg.Exit()])

        self.win = sg.Window(GUI_TITLE, layout,
                        default_element_size=(14, 1),
                        element_justification="c",
                        text_justification='right',
                        font=GUI_FONT,
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
            # Update window
            unattended_bags_on_screen = [bag_id for bag_id, bag in self.unattended_bags.items() if bag_id in self.detected_bags and bag["status"] in [BAG_UNATTENDED_STATUS, BAG_CHECKING_STATUS]]

            if sorted(unattended_bags_on_screen)==sorted(self.unattended_bags_on_screen) \
                and self.alarm_state: 
                # Already in alarm state and unattended bags are the same, so we just need to refresh the image
                self.win_image_elem.Update(data=self.imgbytes)

            elif len(unattended_bags_on_screen) > 0:
                # Previously in normal state, so we change to alarm layout
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

        # Handle window button clicks and events
        event, values = self.win.read(timeout=0)
        if event is None or event == 'Exit':
            return False
        elif "_BUTTON_CHECKING_" in event:
            bag_id = event.split("_")[-1]
            self.unattended_bags[int(bag_id)]["status"] = BAG_CHECKING_STATUS
            logging.info(f"Bag {bag_id} status changed to {BAG_CHECKING_STATUS}")
            self.save_image(self.annotated_image, bag_id, "checking")

        elif "_FALSE_ALARM_" in event:
            bag_id = event.split("_")[-1]
            self.unattended_bags[int(bag_id)]["status"] = BAG_FALSE_ALARM_STATUS
            logging.info(f"Bag {bag_id} status changed to {BAG_FALSE_ALARM_STATUS}")
            self.save_image(self.annotated_image, bag_id, "false-alarm")
        
        return True

    def write_video(self, im0):
        ''' Save video to local file 
        NOT UPDATED YET. DOES NOT WORK. 
        '''
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
        ''' Main function. Loop over frames from video file stream and run algorithm 
        
        The Algo class must contain the following two methods:

        1. __init__
            Keyword arguments:
                yolo_weights 
                yolo_conf
                reid_weights
                unattended_threshold_seconds
            Task:
                - initialise all global variables and models

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
        '''
        self.algo = Algo(yolo_weights=self.yolo_weights, 
                        yolo_conf=self.yolo_conf, 
                        reid_weights=self.reid_weights,
                        unattended_threshold_seconds=self.unattended_threshold_seconds)
        
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