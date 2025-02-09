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

import cv2
import threading
import base64
import time
import numpy as np
import os

from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Recording,
    Record,
    Config,
)
from src.templates.threadwithstop import ThreadWithStop
import struct
import pickle


class threadServerSend(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, socket, address, server, kind, debugger):
        super(threadServerSend, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.debugger = debugger
        self.frame_rate = 5
        self.recording = False
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.video_writer = ""
        self.server = server
        self.socket = socket
        self.address = address
        if (kind in ['Segmentation', 'MainCamera', 'Points', 'ObjectDetectionImage']):
            self.kind = kind
        else:
            print('Wrong Kind of Image!!!')
            self.stop()
        self.subscribe()
        self.Queue_Sending()
        self.Configs()
        # print('Initialize camera thread!!!')

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadImageProcessing", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadImageProcessing", "pipe": self.pipeSendConfig},
            }
        )

    def Queue_Sending(self):
        """Callback function for recording flag."""
        self.queuesList[Recording.Queue.value].put(
            {
                "Owner": Recording.Owner.value,
                "msgID": Recording.msgID.value,
                "msgType": Recording.msgType.value,
                "msgValue": self.recording,
            }
        )
        threading.Timer(1, self.Queue_Sending).start()

    # =============================== STOP ================================================
    def stop(self):
        self.socket.close()
        # cv2.destroyAllWindows()
        super(threadServerSend, self).stop()

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
            start = time.time()
            msg = {"msgValue": None, 'msgOwner': None, 'msgType': None, 'Queue': None}
            # msg = {"msgValue": None, 'Image': None}
            if not self.queuesList[self.kind].empty():
                msg = self.queuesList[self.kind].get()
            if (msg['msgValue']!=None):
                msg = msg["msgValue"]
                # print(msg)
                img = None
                if self.kind == "ObjectDetectionImage":
                    img = msg["Image"]
                    print(msg['Class'])
                    print(msg['Area'])
                else:
                    img = msg
                if(img == None):
                    if self.debugger:
                        print("None Image!!!")
                    self.socket.close()
                else:
                    # image_data = base64.b64decode(img)
                    # img = np.frombuffer(image_data, dtype=np.uint8)
                    # image = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    # image = cv2.resize(image,(320,240))
                    # _, encoded_img = cv2.imencode(".jpg", image)
                    # image = base64.b64encode(encoded_img).decode("utf-8")
                    image = img
                    if (self.debugger):
                        print('FPS: ', 1/(time.time()-start))
                        print('Got', self.kind, ' image')
                    start = time.time()
                    image_bytes = pickle.dumps(image)
                    message = struct.pack("Q", len(image_bytes))+image_bytes
                    self.socket.sendall(message)
                    if (self.debugger):
                        print('Sending Time: ', time.time()-start)
        self.socket.close()

    # =============================== START ===============================================
    def start(self):
        super(threadServerSend, self).start()

        
