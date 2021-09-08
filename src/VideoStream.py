import cv2
from threading import Thread

class VideoStream:
    def __init__(self, source, width=640, height=480, fps=30):
        self.stream = cv2.VideoCapture(source)
        self.stream.set(3, width)
        self.stream.set(4, height)
        self.stream.set(5, fps)
        self.width = self.stream.get(3)
        self.height = self.stream.get(4)
        self.fps = self.stream.get(5)
        self.ret, self.frame = self.stream.read()
        self.stop = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while self.stop == False:
            self.ret, self.frame = self.stream.read()
