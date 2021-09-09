# Science
Counting objects tells many things about the environment. For example, cars passing through an intersection may indicate the level of traffic congestion. Counting birds flying in the sky may indicate the size of the birds flocking. It is important to understand the crowdedness in the current environment because this may interest other AI codes to be run to analyze the environment more.

# AI at Edge
The code runs a COCO-trained SSD model with a given time interval. In each run, it takes a still image from a given camera and outputs counts of any recognized objects listed in [category_names.txt](category_names.txt). The model resizes the input image to 300x300 (WxH) as the model was trained with the size. scheme and codes are from [Nvidia’s DeepLearningExamples](http://github.com/NVIDIA/DeepLearningExamples). The COCO-trained SSD model is also from Nvidia. The code is compatible to run the model using CUDA for faster inferencing.

# Ontology
The code publishes measurements with topic `env.count.OBJECT`, where `OBJECT` is the object recognized. Value for a topic indicates the count of the object.
