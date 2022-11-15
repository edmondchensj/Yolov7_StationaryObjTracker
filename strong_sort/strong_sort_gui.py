import numpy as np
import torch
import sys
import cv2
import gdown
from os.path import exists as file_exists, join
import torchvision.transforms as transforms

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
from .deep.reid_model_factory import show_downloadeable_models, get_model_url, get_model_name

from torchreid.utils import FeatureExtractor
from torchreid.utils.tools import download_url
from .reid_multibackend import ReIDDetectMultiBackend

__all__ = ['StrongSORT']


class StrongSORT(object):
    '''
    
    Todo for Unattended Bags MVP

    0. Change fps to <2fps
    - Test videos: Speed up by 10x
    - Actual inference: 

    1. Improve tracking (DONE)
    - Increase max-age to very high number (original: 70) to help occlusion (2 fps -> 2 x 60 x 5 = 600)
    - Decrease OSNet conf threshold (cant find in repo)

    2. Restrict tracker to stationary objects only (NO NEED)
    - Decrease max dist (what is this?) original: 0.2
    - Decrease max iou distance to small number such that we only want stationary objects e.g. 0.1 (original: 0.9)
    - This might not be needed 
    --> tracker generally assigns same index to bag in stationary position, and those that move around will disappear eventually. 
    --> people are generally stationary inside train, so we dont have to differentiate. 

    3. Improve detection (reduce false negative) (YOLO) (OK)
    - Decrease confidence threshold from 0.25 to 0.05

    4. Set unattended threshold 

    5. Set logic to render unattended status when num occurrence exceed X within Y min of first timestamp

    6. Refactor params to config file
    '''
    def __init__(self, 
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.1, 
                 max_iou_distance=0.9, 
                 max_age=360, # 3fps x 2min x 60s/min = 360
                 n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9,
                 unattended_bag_min_hits=100, # MIN HITS TO FLAG BAG UNATTENDED: 3fps x 2min x 60s/min = 360
                 unattended_threshold_seconds=100
                ):
        self.UNATTENDED_BAG_THRESHOLD_SEC = unattended_threshold_seconds

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img, detection_ts):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i], detection_ts) for i, conf in enumerate(
            confidences)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: # time_since_update refers to number of frames
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf

            # Output data
            #data = np.array([x1, y1, x2, y2, track_id, class_id, conf, unattended_flag])
            data = {
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "object_id": int(track_id),
                "object_class": int(class_id),
                "detection_conf": conf,
                "unattended_bag": True if track.total_duration > self.UNATTENDED_BAG_THRESHOLD_SEC else False,
                "timestamp":  track.ts_latest
            }
            outputs.append(data)

        # if len(outputs) > 0:
        #     outputs = np.stack(outputs, axis=0)

        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
