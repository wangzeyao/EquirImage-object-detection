# EquirImage object detection
## About
Its a tool for testing the performance of equirectangular image detection using IoU or grand circle distance.

## Requirements
- mmdetection (https://github.com/open-mmlab/mmdetection), I recommend you to use docker.
- pysot (https://github.com/STVIR/pysot) docker recommended
- tqdm
- glob2
- argparse
- imageio

## Get Started
Equirectangular image object detection:
1. Use nfov.py to cut equir images into fov
2. Use IOU.py to do the detection and get result and the performance using IoU as the metrics.

VR Video object tracking

1. Download VR video and generate frames.
2. Use object_tracking.py to track object and get image with bbox as the outputs.


## Hint
To get VR video from youtube in the right format, you can follow the steps below
1. Download youtube-dl.exe
2.  In the same directory, open powershell (shift + right-click)
3.  Enter the following command:  
    .\youtube-dl.exe --user-agent "" URL video link here  -F
4.  Note down the code for the file with largest size (Most probably 266).
5.  Enter the following command:  
    .\youtube-dl.exe --user-agent https://www.youtube.com/watch?v=hR1mNJ11ycU  -f 266+251

Then you can generate the frames using youtube-8m-videos-frames(https://github.com/gsssrao/youtube-8m-videos-frames) with the script I modified generateframesfromvideos.sh