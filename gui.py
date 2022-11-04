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
import Algo from skytrain_algo
from yolov7.utils.datasets import LoadImages, LoadStreams

class GUI:
    ''' PySimpleGUI for Skytrain Unattended Bag Detection '''

    def __init__(self):
        sg.theme('LightGreen')
        
        self.open_config_window()

        self.initialise_global_variables()

        self.load_weights()

    def initialise_global_variables(self):
        ''' Initialise global variables for detection algorithm '''
        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")
        
    def open_config_window(self):
        ''' Open window for user to set configurations '''
        # Setup window
        layout = 	[
                    [sg.Text('YOLO Video Player', size=(18,1), font=('Any',18),text_color='#1c86ee' ,justification='left')],
                    [sg.Text('Path to input video'), sg.In("",size=(40,1), key='input'), sg.FileBrowse()],
                    [sg.Text('Optional Path to output video'), sg.In("output.mp4",size=(40,1), key='output'), sg.FileSaveAs()],
                    [sg.Text('YOLO weights'), sg.In("yolov7.pt",size=(40,1), key='yolo-weights')],
                    [sg.Text('StrongSORT weights'), sg.In("osnet_ain_x1_0_msmt17.pt.pt",size=(40,1), key='strong-sort-weights')],
                    [sg.Text('Detection Confidence (YOLO)'), sg.Slider(range=(0,1),orientation='h', resolution=.1, default_value=.5, size=(15,15), key='detection-confidence')],
                    [sg.Text('Threshold'), sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.3, size=(15,15), key='threshold')],
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
            # more...
        win.Close()

    def load_weights(self):
        ''' Load weights into neural networks for YOLO and StrongSORT '''
        pass
        # # derive the paths to the YOLO weights and model configuration
        # weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
        # configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

        # # load our YOLO object detector trained on COCO dataset (80 classes)
        # # and determine only the *output* layer names that we need from YOLO
        # print("[INFO] loading YOLO from disk...")
        # net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        # ln = net.getLayerNames()
        # # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]        # old code changed on Feb-4-2022
        # ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    def load_video_stream(self):
        # initialize the video stream, pointer to output video file, and frame dimensions
        # self.vs = cv2.VideoCapture(args["input"])
        # writer = None
        # (W, H) = (None, None)

        source = args["input"]
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Dataloader
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=imgsz, stride=stride.cpu().numpy())
        else:
            self.dataset = LoadImages(source, img_size=imgsz, stride=stride)
        self.vid_path, self.vid_writer, self.txt_path = [None], [None], [None]

    def draw_bboxes(self):
        ''' Takes in list of detections and frame and draws bounding boxes ''' 
       
        # for detection in self.detections:
        #     # extract the bounding box coordinates
        #     (x, y) = (detection["bbox"][0], detection["bbox"][1])
        #     (w, h) = (detection["bbox"][2], detection["bbox"][3])

        #     # draw a bounding box rectangle and label on the frame
        #     color = [int(c) for c in COLORS[classIDs[i]]]
        #     cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
        #     text = f"UNATTENDED BAG {detection["bag_id"]} {detection["conf"]:.4f}"
        #     cv2.putText(self.frame, text, (x, y - 5),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # draw boxes for visualization
        if len(self.detections) > 0:
            for output in detections:
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                if save_txt:
                    # to MOT format
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_w = output[2] - output[0]
                    bbox_h = output[3] - output[1]
                    # Write MOT compliant results to file
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                        bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                if save_vid or save_crop or show_vid:  # Add bbox to image
                    c = int(cls)  # integer class
                    id = int(id)  # integer id
                    unattended_flag = output[7]

                    if unattended_flag and c in [24, 26, 28]: # backpack, suitcase, handbag
                        label = f"UNATTENDED BAG {id} {conf:.2f}"
                        color = [60,20,220]
                    else:
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                        color = colors[int(cls)]
                    
                    plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)
                    if save_crop:
                        txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                        save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
        # ...
	    self.imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto

    def update_playback_window(self):
        ''' Update playback window showing detection'''
        if not win_started:
            # Create new window
            win_started = True
            layout = [
                [sg.Text('Yolo Playback in PySimpleGUI Window', size=(30,1))],
                [sg.Image(data=self.imgbytes, key='_IMAGE_')],
                [sg.Text('Confidence'),
                sg.Slider(range=(0, 1), orientation='h', resolution=.1, default_value=.5, size=(15, 15), key='confidence'),
                sg.Text('Threshold'),
                sg.Slider(range=(0, 1), orientation='h', resolution=.1, default_value=.3, size=(15, 15), key='threshold')],
                [sg.Exit()]
            ]
            self.win = sg.Window('YOLO Output', layout,
                            default_element_size=(14, 1),
                            text_justification='right',
                            auto_size_text=False, finalize=True)
            image_elem = self.win['_IMAGE_']
        else:
            # Update existing window
            image_elem.Update(data=self.imgbytes)

        # Save results (image with detections) [TO UPDATE]
        if save_vid:
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

        # Close window if frames are completed or if user exits
        event, values = self.win.read(timeout=0)
        if event is None or event == 'Exit':
            break

    def run(self):
        ''' Loop over frames from video file stream and run algorithm '''
        # loop over frames from the video file stream
        win_started = False
        
        # initialise algo 
        algo = Algo(yolo_weights=self.yolo_weights, 
                    yolo_conf=self.yolo_conf, 
                    strong_sort_weights=self.strong_sort_weights)

        for frame_idx, (path, im, im0s, vid_cap) in enumerate(self.dataset): # loop over frames
            
            self.detections = algo.run(frame_idx, path, im, im0s, vid_cap)

            self.draw_bboxes()

            self.update_playback_window()
    
        self.win.Close()

        # release the file pointers
        print("[INFO] cleaning up...")
        writer.release() if writer is not None else None
        vs.release()

if __name__ == "__main__":
    gui = GUI()
    gui.run()