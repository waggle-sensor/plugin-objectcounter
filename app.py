#!/usr/bin/env python3
import logging
import time
import argparse

import cv2
import torch
import torch.nn as nn
from models.experimental import Ensemble
from models.common import Conv, DWConv
from utils.general import non_max_suppression, apply_classifier

from waggle.plugin import Plugin
from waggle.data.vision import Camera

TOPIC_TEMPLATE = "env.count"


def load_class_names(namesfile):
    class_names = {}
    with open(namesfile, 'r') as fp:
        for index, class_name in enumerate(fp):
            class_names[index] = class_name.strip()
    return class_names


class YOLOv7_Main():
    def __init__(self, args, weightfile):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = Ensemble()
        ckpt = torch.load(weightfile, map_location=self.device)
        self.model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

        # Compatibility updates
        for m in self.model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        self.model = self.model.half()
        self.model.eval()

    def pre_processing(self, frame):
        sized = cv2.resize(frame, (640, 640))

        image = sized / 255.0
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(self.device).half()
        return image.unsqueeze(0)

    def inference(self, image):
        with torch.no_grad():
            return self.model(image)[0]


def run(args):
    with Plugin() as plugin, Camera(args.stream) as camera:
        classes_dict = load_class_names("coco.names")
        if args.all_objects:
            target_objects = classes_dict
        else:
            if args.object is None:
                logging.error('No object specified. Will use all registered objects')
                target_objects = classes_dict
            else:
                for target in args.object:
                    target_objects[target] = classes_dict[target]
        classes = [x for x, _ in target_objects.items()]
        logging.info(f'target objects:  {" ".join(target_objects)}')
        logging.debug(f'class numbers for target objects are {classes}')
        
        yolov7_main = YOLOv7_Main(args, args.model)
        logging.info(f'model {args.model} loaded')
        logging.info(f'cut-out confidence level is set to {args.confidence_level}')
        logging.info(f'IOU level is set to {args.iou_level}')
        sampling_countdown = -1
        if args.sampling_interval >= 0:
            logging.info(f'sampling enabled -- occurs every {args.sampling_interval}th inferencing')
            sampling_countdown = args.sampling_interval

        logging.info("object counter starts...")
        for sample in camera.stream():
            do_sampling = False
            if sampling_countdown > 0:
                sampling_countdown -= 1
            elif sampling_countdown == 0:
                do_sampling = True
                sampling_countdown = args.sampling_interval
            image = yolov7_main.pre_processing(sample.data)
            pred = yolov7_main.inference(image)
            results = non_max_suppression(
                pred,
                args.confidence_level,
                args.iou_level,
                classes,
                agnostic=True)[0]

            found = {}
            w = image.shape[1]
            h = image.shape[0]
            for x1, y1, x2, y2, conf, cls in results:
                object_label = classes_dict[cls]
                if object_label in target_objects:
                    l = x1 * w/640  ## x1
                    t = y1 * h/640  ## y1
                    r = x2 * w/640  ## x2
                    b = y2 * h/640  ## y2
                    rounded_conf = round(conf, 2)
                    if do_sampling:
                        frame = cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255,0,0), 2)
                        frame = cv2.putText(frame, f'{object_label}:{rounded_conf}', (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    if not object_label in found:
                        found[object_label] = 1
                    else:
                        found[object_label] += 1
            detection_stats = 'found objects: '
            for object_found, count in found.items():
                detection_stats += f'{object_found} [{count}] '
                plugin.publish(f'{TOPIC_TEMPLATE}.{object_found}', count, timestamp=sample.timestamp)
            logging.info(detection_stats)

            if do_sampling:
                sample.data = image
                sample.save(f'sample.jpg')
                plugin.upload_file(f'sample.jpg', timestamp=sample.timestamp)
                logging.info("uploaded sample")

            if args.continuous == False:
                exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='Debug flag')
    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="camera",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-object', dest='object',
        action='append',
        help='Object name to count')
    parser.add_argument(
        '-all-objects', dest='all_objects',
        action='store_true', default=False,
        help='Consider all registered objects to detect')
    parser.add_argument(
        '-model', dest='model',
        action='store', default='yolov7.pt', type=str,
        help='Path to model')
    parser.add_argument(
        '-image-size', dest='image_size',
        action='store', default=300, type=int,
        help='Input image size')
    parser.add_argument(
        '-confidence-level', dest='confidence_level',
        action='store', default=0.25, type=float,
        help='Confidence level [0. - 1.] to filter out result')
    parser.add_argument(
        '-iou-level', dest='iou_level',
        action='store', default=0.45, type=float,
        help='IOU threshold for NMS')
    parser.add_argument(
        '-continuous', dest='continuous',
        action='store_true', default=False,
        help='Flag to run this plugin forever')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    run(args)
