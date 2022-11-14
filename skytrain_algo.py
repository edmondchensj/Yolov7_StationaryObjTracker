import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov7 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort_gui import StrongSORT


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
CLASSES = [24, 26, 28]

class Algo:
    ''' Class for unattended bag detection algorithm '''
    def __init__(self, **kwargs):
        ''' Initialise global variables and load weights '''
        self.yolo_weights = kwargs.get("yolo_weights", "yolov7.pt") # WEIGHTS / 
        self.yolo_conf = kwargs.get("yolo_conf", 0.05)
        self.strong_sort_weights = kwargs.get("strong_sort_weights", "")
        self.config_strongsort = 'strong_sort/configs/strong_sort.yaml'
        self.iou_thres = 0.3 # NMS IOU threshold
        self.agnostic_nms=False,  # class-agnostic NMS
        self.device = '' # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.half = False # whether to use FP16 half-precision inference
        self._initialise_yolo()
        self._initialise_strongsort()

    def _initialise_yolo(self):
        ''' Make YOLO model and load weights '''
        self.device = select_device(self.device)      
        self.model = attempt_load(Path(self.yolo_weights), map_location=self.device)  # load FP32 model        
        self.COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in self.model.names]
    
    def _initialise_strongsort(self):
        ''' Make StrongSORT model '''
        # initialize StrongSORT
        cfg = get_config()
        print("strong sort config file: ", self.config_strongsort)
        cfg.merge_from_file(self.config_strongsort)
        print(f"\nStrongSORT Configs: {cfg}")

        # Create as many strong sort instances as there are video sources
        self.strongsort = StrongSORT(
                    self.strong_sort_weights,
                    self.device,
                    self.half,
                    max_dist=cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.STRONGSORT.MAX_AGE,
                    n_init=cfg.STRONGSORT.N_INIT,
                    nn_budget=cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
                    unattended_bag_min_hits=cfg.STRONGSORT.UNATTENDED_BAG_MIN_HITS
                )
        self.strongsort.model.warmup()
        
    @torch.no_grad()
    def run(self, im, im0):
        ''' Main tracking algorithm
        '''
        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()

        # Inference
        pred = self.model(im)
        t3 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred[0], self.yolo_conf, self.iou_thres, CLASSES, self.agnostic_nms)
        det = pred[0]
        
        # Pass YOLO detections to StrongSORT
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            # pass detections to strongsort
            t4 = time_synchronized()
            detections = self.strongsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0, t3)
            t5 = time_synchronized()

            print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')
            return detections

        else:
            self.strongsort.increment_ages()
            print('No detections')
            detections = []
        return detections
