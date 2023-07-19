from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtMultimedia import (QCamera, QImageCapture,
                                QMediaCaptureSession,
                                QMediaDevices, QVideoSink)
import numpy as np
import cv2
import time
from src.src.cv.cv import *
from araviq6 import (
    FrameToArrayConverter,
    ArrayWorker,
    ArrayProcessor,
    NDArrayLabel,
)
from src.src.cv.torch.worker import *
'''
PySide6 QCamera功能的封装，支持多个摄像头的热切换
'''
    
class QtDualCamManager(QObject):
    sigVideoInputsChanged = Signal(tuple)
    sigFingerTrackingPos = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.available_cameras = QMediaDevices.videoInputs()
        self.current_cam = None
        self.cams = [None, None]
        self.qmediadevices = QMediaDevices()
        self.qmediadevices.videoInputsChanged.connect(self.handle_videoinput_change)
        self.destines = [lambda e:e, lambda e:e]

    def refresh_cam_info(self) -> list:
        self.available_cameras = QMediaDevices.videoInputs()
        res = []
        for i in self.cams:
            if i is None:
                res.append(None)
            else:
                res.append(i._info)
        return self.available_cameras, res
                
    
    def set_destine(self, destine_func, destine_index):
        self.destines[destine_index] = destine_func
    
        
    def start_cam(self, cam, destine_index:int):
        if self.cams[destine_index] is not None:
            self.cams[destine_index].stop( )
        if cam is not None:
            print(f'Activating {cam.description()} for purpose {destine_index}')
            for i in self.available_cameras:
                if i.id() == cam.id():
                    self.cams[destine_index] = QtCam(i, destine_index)
                    self.cams[destine_index].sigFrameArray.connect(self.destines[destine_index])
                    self.cams[destine_index].start()
                    break
        else:
            self.cams[destine_index] = None
        self.sigVideoInputsChanged.emit(self.refresh_cam_info())
        
    
    def handle_videoinput_change(self):
        self.available_cameras = QMediaDevices.videoInputs()
        for destine_index, cam in enumerate(self.cams):
            if cam is None:
                continue
            if cam._camera.cameraDevice not in self.available_cameras:
                cam.stop()
                cam = None
        self.sigVideoInputsChanged.emit(self.refresh_cam_info())
        
    def dummy_output(self, _):
        pass
    
    def switch_worker(self, e, destine):
        if self.cams[destine] is not None:
            self.cams[destine].switch_worker(e, destine)
        
    
class QtCam(QObject):
    sigFrameArray = Signal(object)
    def __init__(self, camera_info, destine):
        super().__init__()
        self.destine = destine
        self._info = camera_info
        self._camera = QCamera(camera_info)
        self._captureSession = QMediaCaptureSession()
        self._cameraSink = QVideoSink()

        self._captureSession.setCamera(self._camera)
        self._captureSession.setVideoSink(self._cameraSink)

        self._arrayConverter = FrameToArrayConverter()
        self._arrayProcessor = ArrayProcessor()

        self._cameraSink.videoFrameChanged.connect(
            self._arrayConverter.convertVideoFrame
            )

        self._arrayConverter.arrayConverted.connect(
            self._arrayProcessor.processArray
            )
        self.empty_worker = EmptyWorker()
        self.destine_worker = None
        self._arrayProcessor.setWorker(self.empty_worker)
        self.sigFrameArray = self._arrayProcessor.arrayProcessed

    def start(self):
        self._camera.start()

    def stop(self):
        self.sigFrameArray.emit(np.zeros((480,853,3), np.uint8))
        self._arrayProcessor.stop()
        self._camera.stop()
        
    def pause_preview(self):
        self._arrayProcessor.worker().pausePreview = True
    def resume_preview(self):
        self._arrayProcessor.worker().pausePreview = False
        
    def switch_worker(self, e:int, destine:int):
        if e > 0:
            if self.destine_worker is None:
                if destine == 0:
                    self.destine_worker = GazeWorker()
                elif destine == 1:
                    self.destine_worker = FingerWorker()
            self._arrayProcessor.setWorker(self.destine_worker)
        else:
            self._arrayProcessor.setWorker(self.empty_worker)
            del self.destine_worker
            self.destine_worker = None
    
        

