#!/usr/bin/env python3
import logging
import time
import argparse

import cv2
import torch
import numpy as np
from hubconf import nvidia_ssd_processing_utils

import waggle.plugin as plugin
from waggle.data import open_data_source

TOPIC_TEMPLATE = "env.count."

plugin.init()


def run(args):
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    # utils are sourced from
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/utils.py
    utils = nvidia_ssd_processing_utils()
    classes_to_labels = utils.get_coco_object_dictionary()
    if args.all_objects:
        target_objects = classes_to_labels
    else:
        target_objects = args.object
    logging.info("target objects: %s", ' '.join(target_objects))

    logging.info("loading model %s...", args.model)
    if torch.cuda.is_available():
        logging.info("CUDA is available")
        ssd_model = torch.load(args.model)
        ssd_model.to('cuda')
        flag_cuda = True
    else:
        logging.info("CUDA is not avilable; using CPU")
        ssd_model = torch.load(args.model, map_location=torch.device('cpu'))
        flag_cuda = False
    ssd_model.eval()

    logging.info("cut-out confidence level is set to %s", args.confidence_level)
    sampling_countdown = -1
    if args.sampling_interval >= 0:
        logging.info("sampling enabled -- occurs every %sth inferencing", args.sampling_interval)
        sampling_countdown = args.sampling_interval
    logging.info("object counter starts...")
    while True:
        with open_data_source(id=args.stream) as cap:
            timestamp, image = cap.get()

            inputs = [utils.prepare_input(None, image)]
            tensor = utils.prepare_tensor(inputs, cuda=flag_cuda)

            with torch.no_grad():
                detections_batch = ssd_model(tensor)

            results_per_input = utils.decode_results(detections_batch)
            best_results_per_input = [utils.pick_best(results, args.confidence_level) for results in results_per_input]

            bboxes, classes, confidences = best_results_per_input[0]
            classes -= 1

            found = {}
            for box, cls in zip(bboxes, classes):
                object_label = classes_to_labels[cls]
                if object_label in target_objects:
                    if not object_label in found:
                        found[object_label] = 1
                    else:
                        found[object_label] += found[object_label]

            detection_stats = 'found objects: '
            for object_found, count in found.items():
                detection_stats += f'{object_found} [{count}] '
                plugin.publish(f'{TOPIC_TEMPLATE}.{object_found}', count, timestamp=timestamp)
            logging.info(detection_stats)

            if sampling_countdown > 0:
                sampling_countdown -= 1
            elif sampling_countdown == 0:
                # NOTE: OpenCV assumes the image is BGR, but we have RGB (PyWaggle converts it into RGB)
                #       so we flip RGB to BGR to save the image in RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite('/tmp/sample.jpg', image)
                plugin.upload_file('/tmp/sample.jpg')
                logging.info("uploaded sample")
                # Reset the count
                sampling_countdown = args.sampling_interval

            if args.interval > 0:
                time.sleep(args.interval)


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
        action='store', default='coco_ssd_resnet50_300_fp32.pth',
        help='Path to model')
    parser.add_argument(
        '-image-size', dest='image_size',
        action='store', default=300, type=int,
        help='Input image size')
    parser.add_argument(
        '-confidence-level', dest='confidence_level',
        action='store', default=0.4,
        help='Confidence level [0. - 1.] to filter out result')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0, type=int,
        help='Inference interval in seconds')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    run(parser.parse_args())
