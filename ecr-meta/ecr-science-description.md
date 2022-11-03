# Science
Counting objects tells many things about the environment. For example, cars passing through an intersection may indicate the level of traffic congestion. Counting birds flying in the sky may indicate the size of the birds flocking. It is important to understand the crowdedness in the current environment because this may interest other AI codes to be run to analyze the environment more.

# AI@Edge
The code runs a COCO-trained SSD model with a given time interval. In each run, it takes a still image from a given camera and outputs counts of any recognized objects listed in [category_names.txt](category_names.txt). The model resizes the input image to 300x300 (WxH) as the model was trained with the size. scheme and codes are from [Nvidia’s DeepLearningExamples](http://github.com/NVIDIA/DeepLearningExamples) [1]. The COCO-trained SSD model is also from Nvidia. The code is compatible to run the model using CUDA for faster inferencing.

# Using the code
Output: number of objects (car, pedestrian, bike, etc)  
Input: An image  
Image resolution: 300x300  
Inference (calculation) time:  
Model loading time:  

# Arguments
   `-debug`: Debug flag  
   `-stream`: ID or name of a stream, e.g. bottom-camera  
   `-object`: Object name to count  
   `-all-objects`: Consider all registered objects to detect (default = False)  
   `-model`: Path to model (default = `coco_ssd_resnet50_300_fp32.pth`)  
   `-image-size`: Input image size (default = 0.4)  
   `-confidence-level`: Confidence level [0. - 1.] to filter out result  
   `-interval`: Inference interval in seconds (default = 0, no interval)  
   `-sampling-interval`: Sampling interval between inferencing (default = -1, no sampling)  

# Ontology
The code publishes measurements with topic `env.count.OBJECT`, where `OBJECT` is the object recognized. Value for a topic indicates the count of the object.

# Inference from Sage codes
To query the output from the plugin, you can do with python library 'sage_data_client':
```
import sage_data_client

# query and load data into pandas data frame
df = sage_data_client.query(
    start="-1h",
    filter={
        "name": "env.count.*",
    }
)

# print results in data frame
print(df)
# print results by its name
print(df.name.value_counts())
# print filter names
print(df.name.unique())
```
For more information, please see [Access and use data documentation](https://docs.sagecontinuum.org/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).

# References
[1] Nvidia’s DeepLearningExamples, http://github.com/NVIDIA/DeepLearningExamples
