# Deep Racer Bull Fighting Gym
This repository implements the [OpenAI Gym](http://github.com/openai/gym "OpenAI Gym") interface to be compatible with[ AWS DeepRacer](https://aws.amazon.com/deepracer/ " AWS DeepRacer")

## Requirements
-  python 3.6
- tensorflow=1.15.x
## Setting up
### Object Detection
For object detection we used [this](https://github.com/qqwweee/keras-yolo3 "this")  implementation of YOLO V3 deep neural network. To use the model you will need to download weights and convert them by runnig the following:
```bash
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```
Visit qqwweee/keras-yolo3 for mor instructions.
### Obtaining cookie and x-csrf-token
As for now, we do not automatically retrieve the cookie and x-csrf-token. In order to do that, follow the instructions [here](https://www.youtube.com/watch?v=9eTNSt_zHeA "here") to connect your DeepRacer to the wifi.  After the car is connected, surf to its console, [capture the network traffic](https://developers.google.com/web/tools/chrome-devtools/network "capture the network traffic")  and extract the fields from one of the headers' request.

## Demo
<div align="center">
  <a href="https://www.youtube.com/watch?v=L2MsII6-kd8"><img src="https://img.youtube.com/vi/L2MsII6-kd8/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>

