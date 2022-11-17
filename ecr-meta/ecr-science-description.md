# Science

Counting objects tells many things about the environment. For example, cars passing through an intersection may indicate the level of traffic congestion. It is important to understand crowdedness in the current environment because this may interest other AI codes to analysize environment more.

# AI at Edge

The code runs a YOLOv7 model which is trained with COCO dataset with a given time interval. In each run, it takes a still image from a given camera (bottom) and outputs counts of any recognized objects listed in `coco.names`.The model resizes the input image to 640x640 (WxH) as the model was trained with the size.

# Ontology

The code publishes measurement with topic `env.count.OBJECT`, where `OBJECT` is the object recognized. Value for a topic indicates the count of the object.

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
For more information, please see [Access and use data documentation](https://docs.waggle-edge.ai/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).
