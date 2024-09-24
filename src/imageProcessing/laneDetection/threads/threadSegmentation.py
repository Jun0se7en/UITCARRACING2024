import cv2
import threading
import base64
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Segmentation,
    Record,
    Config,
)
from src.templates.threadwithstop import ThreadWithStop
from src.imageProcessing.laneDetection.threads.infer_trt import TRT
import torch
import math

# Use this thread for LaneLine Segmentation
class threadSegmentation(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, debugger):
        super(threadSegmentation, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.model = TRT('./models/best.engine')
        self.W_ = 640
        self.H_ = 384
        self.debugger = debugger
        self.subscribe()
        self.Configs()

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadSegmentation", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadSegmentation", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        # cv2.destroyAllWindows()
        super(threadSegmentation, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            print(message)
        threading.Timer(1, self.Configs).start()

    # =============================== Utilities Function ==============================================
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im
    
    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        var = True
        while self._running:
            if var:
                start = time.time()
                # Segmentation
                img = {"msgValue": 1}
                while type(img["msgValue"]) != type(":text"):
                    img = self.queuesList["SegmentCamera"].get()    # Get image from camera
                image_data = base64.b64decode(img["msgValue"])
                img = np.frombuffer(image_data, dtype=np.uint8)     
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (640, 360))
                img_rs= np.zeros((360,640,3))

                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img=torch.from_numpy(img)
                img = torch.unsqueeze(img, 0)  # add a batch dimension
                img=img.cuda().float() / 255.0
                img = img.cuda()
                
                with torch.no_grad():
                    img_out = self.model(img)

                x0=img_out[0]

                _,da_predict=torch.max(x0, 1)

                DA = da_predict.byte().cpu().data.numpy()[0]*255
                img_rs[DA>100]=[255,255,255]

                if img_rs.dtype == np.float64:  
                    img_rs = cv2.convertScaleAbs(img_rs) 
                img_rs = cv2.resize(img_rs, (320, 240))
                img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
                contours, _ = cv2.findContours(img_rs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                black_image=np.zeros((240,320,3),np.uint8)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(black_image, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((3, 3), np.uint8)
                black_image = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, kernel)
                black_image = cv2.erode(black_image, kernel, iterations = 3)
                black_image = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)
                _, encoded_img = cv2.imencode(".jpg", black_image)
                image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                self.queuesList[Segmentation.Queue.value].put(
                    {
                        "Owner": Segmentation.Owner.value,
                        "msgID": Segmentation.msgID.value,
                        "msgType": Segmentation.msgType.value,
                        "msgValue": image_data_encoded,
                    }
                )
                # print('Inference FPS: ', round(1/(time.time() - start),1))
            var = not var

    # =============================== START ===============================================
    def start(self):
        super(threadSegmentation, self).start()