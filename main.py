import sys

sys.path.append(".")
import logging
import os
import time
from multiprocessing import Event, Queue
from multiprocessing.sharedctypes import Value

import torch
import argparse

from src.clearBuffer.processClearBuffer import processClearBuffer

from src.directionControl.decisionMaking.processDecisionMaking import (
    processDecisionMaking,
)

# ===================================== PROCESS IMPORTS ==================================
from src.gateway.processGateway import processGateway
from src.hardware.camera.processCamera import processCamera
from src.imageProcessing.laneDetection.processSegmentation import processSegmentation
from src.server.processServer import processServer
from src.positionFusion.processPositionFusion import processPositionFusion
from src.control.manualControl.processManualControl import processManualControl
from src.imageProcessing.objectDetection.processObjectDetection import (
    processObjectDetection,
)

# ======================================== SETTING UP ====================================
parser = argparse.ArgumentParser(description='Process Input IP')
parser.add_argument('--jetsonip', type=str, help='Jetson IP', default="192.168.2.193")
parser.add_argument('--jetsonport', type=int, help='Jetson Port', default=12345)
args = parser.parse_args()

allProcesses = list()
queueList = {
    'Control': Queue(),
    "Critical": Queue(),
    "Warning": Queue(),
    "General": Queue(),
    "Config": Queue(),
    # After Processed
    "Segmentation": Queue(),
    "ObjectDetection": Queue(),
    "ObjectDetectionImage": Queue(),
    "Points": Queue(),

    # Camera
    "MainCamera": Queue(),
    "SegmentCamera": Queue(),
    "ObjectCamera": Queue(),
    "Key": Queue(),
}

blocksend_flag = Value('i', 0)
key = Value('i', 0)

logging = logging.getLogger()

# Camera
Camera = True
  
# Image Processing
Segmentation = False
ObjectDetection = True

# Server
Server = True

# CV control
DecisionMaking = False

# Clear Buffer (Optional Remove Further)
ClearBuffer = True

# Manual Control
ManualControl = False

# Position Fusion
PositionFusion = False

# =========================== CHECKING NECESSARY PROCESSES ===============================
# if not Camera:
#     raise Exception("Camera is not initialized!!!")

# if (ManualControl or DecisionMaking) and not SerialHandler:
#     raise Exception("Serial Handler is not initialized!!!")

# if not ClearBuffer:
#     raise Exception("Clear Buffer is not initialized!!!")

# ===================================== SETUP PROCESSES ==================================

# Initializing gateway
processGateway = processGateway(queueList, logging)
allProcesses.append(processGateway)

# Initializing camera
if Camera:
    width = 640.0
    height = 480.0
    processCamera = processCamera(queueList, logging, width, height, debugging=False)
    allProcesses.append(processCamera)

if Segmentation:
    processSegmentation = processSegmentation(queueList, logging, debugging=False)
    allProcesses.append(processSegmentation)

if ObjectDetection:
    library = "./models/libmyplugins.so"
    engine = "./models/yolov5n.engine"
    conf_thres = 0.6
    iou_thres = 0.4
    # classes = [
    #     "Car",
    #     "CrossWalk",
    #     "Greenlight",
    #     "HighwayEnd",
    #     "HighwayEntry",
    #     "NoEntry",
    #     "OneWay",
    #     "Parking",
    #     "Pedestrian",
    #     "PriorityRoad",
    #     "Redlight",
    #     "Roundabout",
    #     "Stop",
    #     "Yellowlight",
    # ]
    classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]
    processObjectDetection = processObjectDetection(
        queueList,
        logging,
        library,
        engine,
        conf_thres,
        iou_thres,
        classes,
        debugging=True,
    )
    allProcesses.append(processObjectDetection)

if Server:
    hostname = args.jetsonip # Jetson IP
    port = args.jetsonport
    receive_flag = 1
    kindofimages = ["MainCamera", "Segmentation", "Points", "ObjectDetectionImage"]
    kind = kindofimages[0]
    processServer1 = processServer(
        queueList, logging, hostname, port, kind, blocksend_flag, key, not receive_flag, debugging=False
    )
    allProcesses.append(processServer1)
    port += 1
    kind = kindofimages[3]
    processServer2 = processServer(
        queueList, logging, hostname, port, kind, blocksend_flag, key, not receive_flag, debugging=False
    )
    allProcesses.append(processServer2)


if DecisionMaking:
    processDecisionMaking = processDecisionMaking(
        queueList, logging, blocksend_flag, key, debugging=False
    )
    allProcesses.append(processDecisionMaking)


if ClearBuffer:
    processClearBuffer = processClearBuffer(queueList, logging, debugging=False)
    allProcesses.append(processClearBuffer)

if PositionFusion:
    processPositionFusion = processPositionFusion(queueList, logging, debugging=False)
    allProcesses.append(processPositionFusion)

if ManualControl:
    processManualControl = processManualControl(queueList, logging, blocksend_flag, debugging=False)
    allProcesses.append(processManualControl)

# ===================================== START PROCESSES ==================================
for process in allProcesses:
    process.daemon = True
    process.start()

# ===================================== STAYING ALIVE ====================================
blocker = Event()
try:
    blocker.wait()
except KeyboardInterrupt:
    print("\nCatching a Keyboard Interruption exception! Shutdown all processes.\n")
    for proc in allProcesses:
        print("Process stopped", proc)
        proc.stop()
        time.sleep(0.1)
        proc.join()
