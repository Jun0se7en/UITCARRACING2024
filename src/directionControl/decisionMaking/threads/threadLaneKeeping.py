import cv2
import threading
import base64
import time
import numpy as np
import os
import sys
import json
import random
import ctypes

from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Record,
    Config,
    Segmentation,
    Points,
)
from src.templates.threadwithstop import ThreadWithStop
import socket
import json
import math
import signal

class threadLaneKeeping(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, flag, key, debugger):
        super(threadLaneKeeping, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.speed = 4
        self.angle = 0
        self.debugger = debugger
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.CHECKPOINT = 50
        self.old_angle = 0
        self.Beta = 0.1
        self.smooth_angle = 0
        self.startflag = False
        self.flag = flag
        self.trigger_flag = 0
        self.key = 0
        self.manual_speed = 0
        self.manual_steer = 0
        self.old_manual_speed = 0
        self.old_manual_steer = 0
        # self.client.send(self.speed, self.smooth_angle)
        self.subscribe()
        self.Configs()
        self.message = {}
        self.message_type = ""
    
    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadManualControl", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadManualControl", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        self.socket.close()
        print('Socket Close')
        super(threadLaneKeeping, self).stop()

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
        """This function will run while the running flag is True. 
        It captures the image from camera and make the required modifies and then it send the data to process gateway."""   
        while self._running:
            ### YOUR CODE HERE ###
            print('Lane Keeping Running!!!')

    # =============================== START ===============================================
    def start(self):
        super(threadLaneKeeping, self).start()

        
