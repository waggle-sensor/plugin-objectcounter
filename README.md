# Plugin Object Counter

The plugin counts objects of interest from an image. Supported objects can be found in [category_names](category_names.txt). The plugin automatically uses Nvidia GPU for inferencing when available.

## How to use

```bash
# -stream option is required to indicate source of a streaming input
# for example, python3 app.py -stream bottom_image gets
#   images from the camera named "bottom_image"

# to count cars and pedestrian
$ python3 app.py -object car -object pedestrian

# to count cows with 70% confidence level
$ python3 app.py -object cow -confidence-level 0.7

# to count all registered objects and sample the image used in interencing
$ python3 app.py -all-objects -sampling-interval 0

# to count cars every 10 seconds
$ python3 app.py -object car -interval 10
```

## Developer Notes

- The SSD model and util functions in [Nvidia's DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) have an issue that when it does not detect any object from an image, it throws the [runtime exception](https://github.com/NVIDIA/DeepLearningExamples/issues/680). As a workaround, a code change is applied as [suggested](https://github.com/NVIDIA/DeepLearningExamples/issues/680#issuecomment-690337224)

## Acknowledgement

The model scheme and codes are from [Nvidia's DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples). The COCO-trained SSD model is also from Nvidia.
