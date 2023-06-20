import logging
import sys
import datetime
import asyncio
import qasync
import threading
from PySide6.QtCore import QObject, Slot, QThread, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout


class Logic(QObject):  # Logic to run in a separate thread
    enabled = False

    @Slot()
    def start(self):
        logger.debug('Start called')
        self.enabled = True
        asyncio.create_task(self.process())

    @Slot()
    def stop(self):
        logger.debug('Stop called')
        self.enabled = False

    async def process(self):
        while self.enabled:
            # demonstrate that we can emit a signal from a thread
            self.message.emit(f'Processing ({datetime.datetime.now()}), worker thread id: {threading.get_ident()}...')
            await asyncio.sleep(0.5)

    # signal declaration
    message = Signal(str)

# subclass QThread. override the run function to create an event loop and run forever
class WorkerThread(QThread):
    def run(self):
        loop = qasync.QEventLoop(self)
        asyncio.set_event_loop(loop)
        loop.run_forever()
        
    def cancel(self):
        loop = asyncio.get_event_loop()
        loop.stop()

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    app = QApplication(sys.argv)
    logger.debug(f'main thread id: {threading.get_ident()}')

    # Move logic to another thread
    logic = Logic()    
    thread = WorkerThread()
    logic.moveToThread(thread)
    thread.start()

    window = QMainWindow()
    window.setCentralWidget(QWidget())
    window.centralWidget().setLayout(QVBoxLayout())

    window.centralWidget().layout().addWidget(QPushButton(text='Start', clicked=logic.start)) # type: ignore
    window.centralWidget().layout().addWidget(QPushButton(text='Stop', clicked=logic.stop)) # type: ignore
    # connect logic in workerThread to lambda function in this thread
    logic.message.connect(lambda msg: logger.debug(f'current thread: {threading.get_ident()}, {msg}'))

    window.show()

    logger.debug('Launching the application...')
    exit(app.exec_())