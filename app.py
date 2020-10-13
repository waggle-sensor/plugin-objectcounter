import time
import argparse

import torch
import numpy as np
from hubconf import nvidia_ssd_processing_utils

import waggle.plugin as plugin
from waggle.data import open_data_source

TOPIC_IMAGE = "bottom_image"
TOPIC_CAR = "env.count.car"
TOPIC_PEDESTRIAN = "env.count.pedestrian"

plugin.init()


def run(args):
    # utils are sourced from
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/utils.py
    utils = nvidia_ssd_processing_utils()
    classes_to_labels = utils.get_coco_object_dictionary()

    if torch.cuda.is_available():
        print("CUDA is available")
        ssd_model = torch.load(args.model)
        ssd_model.to('cuda')
    else:
        print("CUDA is not avilable; use CPU")
        ssd_model = torch.load(args.model, map_location=torch.device('cpu'))
    ssd_model.eval()

    print("Cut-out confidence level is {:.2f}".format(args.confidence_level))
    print("Car pedestrian counter starts...")
    while True:
        with open_data_source(id=TOPIC_IMAGE) as cap:
            timestamp, image = cap.get()

            inputs = [utils.prepare_input(None, image)]
            tensor = utils.prepare_tensor(inputs)

            #with torch.no_grad():
            detections_batch = ssd_model(tensor)

            results_per_input = utils.decode_results(detections_batch)
            best_results_per_input = [utils.pick_best(results, args.confidence_level) for results in results_per_input]

            bboxes, classes, confidences = best_results_per_input[0]
            classes -= 1

            cars = 0
            pedestrians = 0
            for box, cls in zip(bboxes, classes):
                if "car" in classes_to_labels[cls]:
                    cars += 1
                elif "person" in classes_to_labels[cls]:
                    pedestrians += 1

            print("cars {:d}, pedestrians{:d}".format(cars, pedestrians))
            if args.interval > 0:
                time.sleep(args.interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        help='Sampling interval in seconds')
    parser.add_argument(
        '-sampling-path', dest='sampling_path',
        action='store', default='./',
        help='Path to sample images')
    run(parser.parse_args())
