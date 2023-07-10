from PySide6.QtCore import QThread
from qasync import QEventLoop


import asyncio


class QAsyncWorkerThread(QThread):
    """
    兼容Qt的异步WorkerThread
    线程具有独立的Event loop，可以使用asyncio语法
    并且支持signal/slot机制
    example：
        hub = SensorHub()
        thread = QAsyncWorkerThread()
        hub.moveToThread(thread)
        thread.start()
    """

    def run(self):
        loop = QEventLoop(self)
        asyncio.set_event_loop(loop)
        loop.run_forever()