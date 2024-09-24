from src.templates.workerprocess import WorkerProcess
from src.directionControl.decisionMaking.threads.threadLaneKeeping import threadLaneKeeping
from multiprocessing import Pipe
from threading import Event
from multiprocessing.sharedctypes import Value
class processDecisionMaking(WorkerProcess):
    """This process decide car speed and angle\n
    Args:
        queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
        example (bool, optional): A flag for running the example. Defaults to False.
    """

    # ===================================== INIT =========================================
    def __init__(self, queueList, logging, flag, key, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        pipeRecv, pipeSend = Pipe(duplex=False)
        self.pipeRecv = pipeRecv
        self.pipeSend = pipeSend
        self.flag = flag
        self.key = key
        self.debugging = debugging
        super(processDecisionMaking, self).__init__(self.queuesList)

    # ===================================== STOP ==========================================
    def stop(self):
        """Function for stopping threads and the process."""
        for thread in self.threads:
            thread.stop()
            thread.join()
        super(processDecisionMaking, self).stop()

    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processDecisionMaking, self).run()

    # ===================================== INIT TH =================================
    def _init_threads(self):
        """Initializes the read and the write thread."""

        LaneKeepingTh = threadLaneKeeping(self.pipeRecv, self.pipeSend, self.queuesList, self.logging, self.flag, self.key, self.debugging)
        self.threads.append(LaneKeepingTh)
