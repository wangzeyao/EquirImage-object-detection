# EquirImage object detection
## About
Its a tool for testing the performance of equirectangular image detection using IoU or grand circle distance.

## Requirements
- mmdetection (https://github.com/open-mmlab/mmdetection), I recommend you to use docker.
- tqdm
- glob2
- argparse

## Get Started
1. Use nfov.py to cut equir images into fov
2. Use IOU.py to do the detection and get result and the performance using IoU as the metrics.