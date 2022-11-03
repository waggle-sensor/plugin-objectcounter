# Object Counter

The plugin counts objects of interest from an image. Supported objects can be found in [coco.names](coco.names). The plugin automatically uses Nvidia GPU for inferencing when available.

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
$ python3 app.py -object car -continuous -interval 10
```

## funding
[NSF 1935984](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1935984)

## collaborators
Bhupendra Raut, Dario Dematties Reyes, Joseph Swantek, Neal Conrad, Nicola Ferrier, Pete Beckman, Raj Sankaran, Robert Jackson, Scott Collis, Sean Shahkarami, Seongha Park, Sergey Shemyakin, Wolfgang Gerlach, Yongho Kim
