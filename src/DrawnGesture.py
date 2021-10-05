import numpy
import time
import cv2

class GestureCatcher:
    def __init__(
        self,
        image_height = 640,
        image_width = 480
    ):
        self.speed_limit = 6
        self.speed_history_size = 5
        self.distance_min = 0.1

        self.tracked = numpy.full(
            [1, 2],
            numpy.zeros([2], dtype=float)
        )

        self.instant = numpy.full([2], time.monotonic_ns(), dtype = int)
        
        self.speed_history = numpy.zeros([self.speed_history_size], dtype=float)

        self.position = numpy.zeros([2], dtype=int)

        self.distance = 0.0

        self.speed_magnitude_amplified = 0.0

        self.is_drawing = False

        self.image_height = image_height
        self.image_width = image_width

        self.gesture_image = numpy.full(
            [self.image_height, self.image_width, 4],
            numpy.zeros([4], dtype=numpy.uint8)
        )

        self.drawn_gesture = DrawnGesture()

    def update(self, mp):
        self.track_velocity(mp)
        if not self.is_drawing: self.detect()

    def reset_gesture_image(self):
        self.gesture_image = numpy.full(
            [self.image_height, self.image_width, 4],
            numpy.zeros([4], dtype=numpy.uint8)
        )

    def track_velocity(self, mp):
        position = numpy.full(
            [2, 2],
            numpy.zeros([2], dtype=int)
        )

        position[0] = self.tracked[0]

        self.tracked[0] = mp.landmark[0][19]

        self.instant[0] = self.instant[1]
        self.instant[1] = time.monotonic_ns()

        position[1] = self.tracked[0]

        self.position = position
        position = position / mp.image_size

        self.distance = numpy.linalg.norm(position[1] - position[0])

        speed = abs(position[1] - position[0]) / (self.instant[1] - self.instant[0])
        self.speed_magnitude_amplified = numpy.linalg.norm(speed) * (10 ** 10)

        if numpy.mean(self.speed_history) < self.speed_limit:
            self.is_drawing = False
            print(self.drawn_gesture.points)

        if self.speed_magnitude_amplified > self.speed_limit and self.is_drawing == False and self.distance > self.distance_min:
            self.is_drawing = True
            self.drawn_gesture = DrawnGesture()
            self.speed_history = numpy.full([self.speed_history_size], self.speed_limit,dtype=float)
            self.reset_gesture_image()
        
        if self.is_drawing:
            self.speed_history[:-1] = self.speed_history[1:]
            self.speed_history[-1] = self.speed_magnitude_amplified
            self.speed_magnitude_amplified = numpy.mean(self.speed_history)
            self.drawn_gesture.points = numpy.append(self.drawn_gesture.points, self.position[1])

    def draw(self, image):
        if self.speed_magnitude_amplified <= self.speed_limit: return image

        position = self.position

        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        color_intensity = self.speed_magnitude_amplified - self.speed_limit
        color_intensity = color_intensity / (2 * self.speed_limit)
        color_intensity = int(color_intensity * 255)

        cv2.line(self.gesture_image, position[0], position[1], (255, 0, color_intensity), 4)

        added_image = cv2.addWeighted(image, 1, self.gesture_image, 1, 0)
        
        return added_image

    def detect(self):
        image = self.gesture_image
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        cv2.imwrite("gesture/gesture.png", image)

class DrawnGesture:
    def __init__(
        self
    ):
        self.points = numpy.array([], dtype=object)
