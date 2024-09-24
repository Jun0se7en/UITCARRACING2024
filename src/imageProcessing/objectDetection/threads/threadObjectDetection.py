# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import sys
import imutils
import cv2
import threading
import base64
import time
import numpy as np
import os
import torch
import tensorrt as trt
import ctypes

from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    ObjectDetection,
    ObjectDetectionImage,
    Recording,
    Record,
    Config,
)
from src.templates.threadwithstop import ThreadWithStop
from src.imageProcessing.objectDetection.threads.yoloDet import YoloTRT

class threadObjectDetection(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, library, engine, conf_thres, iou_thres, classes, debugging):
        super(threadObjectDetection, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.library = library
        ctypes.CDLL(self.library)
        self.engine = engine
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.debugging = debugging
        self.model = YoloTRT(engine=self.engine, conf=self.conf_thres, categories=self.classes, iou=self.iou_thres, debugging=self.debugging)
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.video_writer = ""
        self.subscribe()
        self.Configs()

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadObjectDetection", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadObjectDetection", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        self.model.destroy()
        super(threadObjectDetection, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            print(message)
        threading.Timer(1, self.Configs).start()

    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        while self._running:
            if not self.queuesList["ObjectCamera"].empty():
                img = self.queuesList["ObjectCamera"].get()
                image_data = base64.b64decode(img["msgValue"])
                img = np.frombuffer(image_data, dtype=np.uint8)
                image = cv2.imdecode(img, cv2.IMREAD_COLOR)
                frame = imutils.resize(image, width=600)
                if (self.debugging):
                    start = time.time()
                    # print("Yolo Detecting!!!")
                    res, t = self.model.infer(frame)
                    print('YOLO FPS: ', round(1/(time.time()-start),1))
                else:
                    res, t = self.model.infer(frame)
                classes = []
                areas = []
                for i in res:
                    classes.append(i["class"])
                    x_min, y_min, x_max, y_max = i["box"]
                    # Calculate area of object
                    area = (x_max - x_min) * (y_max - y_min)
                    areas.append(area)
                if (self.debugging):
                    _, encoded_img = cv2.imencode(".jpg", frame)
                    image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                    msg = {"Image": image_data_encoded, "Class": classes, "Area": areas}
                else:
                    msg = {"Image": None, "Class": classes, "Area": areas}
                self.queuesList[ObjectDetection.Queue.value].put(
                        {
                            "Owner": ObjectDetection.Owner.value,
                            "msgID": ObjectDetection.msgID.value,
                            "msgType": ObjectDetection.msgType.value,
                            "msgValue": msg,
                        }
                    )
                self.queuesList[ObjectDetectionImage.Queue.value].put(
                    {
                        "Owner": ObjectDetectionImage.Owner.value,
                        "msgID": ObjectDetectionImage.msgID.value,
                        "msgType": ObjectDetectionImage.msgType.value,
                        "msgValue": msg,
                    }
                )

    # =============================== START ===============================================
    def start(self):
        super(threadObjectDetection, self).start()

        
