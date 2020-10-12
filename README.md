# plugin-carped
Car and pedestrian counter plugin

### Developer Notes

- The SSD model and util functions in [Nvidia's DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) have an issue that when it does not detect any object from an image, it throws the [runtime exception](https://github.com/NVIDIA/DeepLearningExamples/issues/680). As a workaround, a code change is applied as [suggested](https://github.com/NVIDIA/DeepLearningExamples/issues/680#issuecomment-690337224)
