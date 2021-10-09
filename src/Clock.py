import time

import numpy
import cv2




class Clock:
    def __init__(
        self,
        name='',

        i_p_clock = (0,0,-1,-1),
        i_p_hand = (0,0,-1,-1),

        scale_mode = 0,
        i_p_ref_A = (0,0),
        i_p_ref_B = (0,0),
        i_p_ref_C = (0,0),
        i_p_ref_D = (0,0),

        is_gesture_catcher = True,
        is_gesture_classifier = False,

        is_clipped = True
    ):
        self.name = name

        self.i_p_clock = numpy.array(i_p_clock)
        self.i_p_hand = numpy.array(i_p_hand)

        self.i_p_ref_A = numpy.array(i_p_ref_A)
        self.i_p_ref_B = numpy.array(i_p_ref_B)
        self.i_p_ref_C = numpy.array(i_p_ref_C)
        self.i_p_ref_D = numpy.array(i_p_ref_D)

        self.p_clock = numpy.zeros([2], dtype=int)
        self.p_hand = numpy.zeros([2], dtype=int)
        self.p_ref_list = numpy.zeros(
            [4 if scale_mode == 2 else 8, 2],
            dtype=int
        )

        self.p_clock_norm = numpy.zeros([2], dtype=int)
        self.p_hand_norm = numpy.zeros([2], dtype=int)

        self.p_tran = numpy.zeros([2], dtype=int)
        self.p_norm = numpy.zeros([2], dtype=int)
        self.m_clock = 1.0
        self.m_ref_list = numpy.ones(
            [int(len(self.p_ref_list) / 2)],
            dtype=float
        )
        self.k_m_ref_list = numpy.ones(
            [int(len(self.p_ref_list) / 2)],
            dtype=float
        )

        self.k_r_clock = 1.0

        self.scale = 1.0
        self.r_clock = 1.0

        self.r_hand = 0.0
        self.phi_r_hand = 0.0
        self.x_r_hand = 0.0
        self.y_r_hand = 0.0

        self.enable_calibration = False
        self.is_calibrated = False

        self.scale_mode = scale_mode
        self.scale_history = numpy.ones([10], dtype=float)

        self.scale_ratio_A = 1.0
        self.scale_ratio_B = 1.0

        self.speed_tracked_point = numpy.zeros([2], dtype=float)
        self.position_history = numpy.full([2, 2], numpy.zeros([2], dtype=int))
        self.position_abs_history = numpy.full([2, 2], numpy.zeros([2], dtype=int))
        self.direction = numpy.zeros([2], dtype=float)
        self.instant = numpy.full([2], time.monotonic_ns(), dtype = int)
        self.speed = [0.0,0.0]
        self.speed_magnitude = 0.0
        self.speed_history = numpy.zeros(
            [5], dtype=float
        )
        self.speed_magnitude_average = 0.0

        self.is_gesture_catcher = is_gesture_catcher
        self.gesture_catcher = None

        self.is_clipped = is_clipped

    # Auxiliary methods.
    # ------------------

    def _midpoint(self, point_A, point_B):
        return numpy.array(
            [
                (
                    min(point_A[0], point_B[0])
                    + (abs(point_A[0] - point_B[0]) / 2)
                ),
                (
                    min(point_A[1], point_B[1])
                    + (abs(point_A[1] - point_B[1]) / 2)
                ),
            ],
            dtype=int
        )

    def _initialize_point(self, i_point, mp):
        i_point_A = i_point[:2]
        i_point_B = i_point[2:]

        if any(n < 0 for n in i_point_A):
            i_point_A = numpy.abs(i_point_A)
            return numpy.array(
                mp.landmark[i_point_A[0]][i_point_A[1]] * mp.image_size,
                dtype=int
            )
        if any(n < 0 for n in i_point_B):
            return mp.landmark[i_point_A[0]][i_point_A[1]]
        
        return self._midpoint(
            mp.landmark[i_point_A[0]][i_point_A[1]],
            mp.landmark[i_point_B[0]][i_point_B[1]]
        )

    # Main methods.
    # -------------

    def _setup(self, mp):
        self.p_clock = self._initialize_point(self.i_p_clock, mp)
        self.p_hand = self._initialize_point(self.i_p_hand, mp)

        if self.scale_mode == 1:
            self.p_ref_list[0] = (
                mp.landmark[self.i_p_ref_A[0]][self.i_p_ref_A[1]]
            )
            self.p_ref_list[1] = (
                mp.landmark[self.i_p_ref_B[0]][self.i_p_ref_B[1]]
            )

        elif self.scale_mode == 2:
            self.p_ref_list[0] = (
                mp.landmark[self.i_p_ref_A[0]][self.i_p_ref_A[1]]
            )
            self.p_ref_list[1] = (
                mp.landmark[self.i_p_ref_B[0]][self.i_p_ref_B[1]]
            )

            self.p_ref_list[2] = (
                mp.landmark[self.i_p_ref_C[0]][self.i_p_ref_C[1]]
            )
            self.p_ref_list[3] = (
                mp.landmark[self.i_p_ref_D[0]][self.i_p_ref_D[1]]
            )

        elif self.scale_mode == 3:
            self.p_ref_list[1] = (
                self._midpoint(mp.landmark[0][9], mp.landmark[0][10])
            )
            self.p_ref_list[0] = (
                self._midpoint(mp.landmark[0][3], mp.landmark[0][6])
            )

            self.p_ref_list[2] = mp.landmark[0][7]
            self.p_ref_list[3] = mp.landmark[0][8]

            self.p_ref_list[4] = self._midpoint(
                mp.landmark[0][11], mp.landmark[0][12]
            )
            self.p_ref_list[5] = self._midpoint(
                mp.landmark[0][23], mp.landmark[0][24]
            )

            self.p_ref_list[6] = mp.landmark[0][23]
            self.p_ref_list[7] = mp.landmark[0][24]

        self.p_clock_norm = self.p_clock / mp.image_size
        self.p_hand_norm = self.p_hand / mp.image_size

    def _translation(self):
        self.p_tran = self.p_hand - self.p_clock

    def _calibration(self):
        self.m_clock = numpy.linalg.norm(self.p_tran)

        if self.scale_mode == 1:
            m_ref = numpy.linalg.norm(self.p_ref_list[0] - self.p_ref_list[1])
            self.m_ref_list[0] = m_ref if m_ref > 0 else self.m_ref_list[0]

        elif self.scale_mode in {2, 3}:
            for i, j in zip(
                range(0, len(self.p_ref_list), 2),
                range(len(self.m_ref_list))
            ):
                m_ref = numpy.linalg.norm(
                    self.p_ref_list[i] - self.p_ref_list[i+1]
                )
                self.m_ref_list[j] = m_ref if m_ref > 0 else self.m_ref_list[j]

        if self.enable_calibration:
            self.k_r_clock = self.m_clock

            for i in range(len(self.m_ref_list)):
                self.k_m_ref_list[i] = self.m_ref_list[i]

            self.scale_ratio_A = self.k_m_ref_list[0] / self.k_m_ref_list[1]
            self.scale_ratio_B = self.k_m_ref_list[2] / self.k_m_ref_list[3]

            self.is_calibrated = True

    def _scaling(self):
        scale = 1.0

        if self.scale_mode == 1:
            scale = self.m_ref_list[0] / self.k_m_ref_list[0]

        elif self.scale_mode == 2:
            scale_AB = self.m_ref_list[0] / self.k_m_ref_list[0]
            scale_CD = self.m_ref_list[1] / self.k_m_ref_list[1]
            scale = max(scale_AB, scale_CD)

        elif self.scale_mode == 3:
            ratio_A = self.m_ref_list[0] / self.m_ref_list[1]

            if ratio_A >= self.scale_ratio_A:
                scale = self.m_ref_list[0] / self.k_m_ref_list[0]
            else:
                scale = self.m_ref_list[1] / self.k_m_ref_list[1]

        self.scale_history[:-1] = self.scale_history[1:]
        self.scale_history[-1] = scale
        self.scale = numpy.mean(self.scale_history)

        self.r_clock = self.k_r_clock * self.scale

    def _normalization(self):
        self.p_norm = self.p_tran / self.r_clock
        self.p_norm[1] = -self.p_norm[1]

        self.x_r_hand = self.p_norm[0]
        self.y_r_hand = self.p_norm[1]

        self.r_hand = numpy.linalg.norm(self.p_norm)

        self.phi_r_hand = numpy.degrees(
            numpy.arctan2(self.x_r_hand, self.y_r_hand)
        ) 
        self.phi_r_hand = -self.phi_r_hand + 90
        if self.phi_r_hand < 0: self.phi_r_hand = self.phi_r_hand + 360
        self.phi_r_hand = self.phi_r_hand / 360

        if self.is_clipped:
            self.r_hand =  numpy.clip(self.r_hand, 0, 1)
            self.phi_r_hand =  numpy.clip(self.phi_r_hand, 0, 1)
            self.x_r_hand =  numpy.clip(self.x_r_hand, 0, 1)
            self.y_r_hand =  numpy.clip(self.y_r_hand, 0, 1)

    def _speed_tracking(self, mp):
        self.position_history[0] = self.speed_tracked_point
        self.speed_tracked_point = numpy.clip(
            self.p_hand, [0,0], mp.image_size
        )
        self.position_history[1] = self.speed_tracked_point
        self.position_abs_history = self.position_history.astype(int)
        self.position_history = self.position_history / mp.image_size

        self.instant[0] = self.instant[1]
        self.instant[1] = time.monotonic_ns()

        distance = self.position_history[1] - self.position_history[0]

        if numpy.linalg.norm(distance) == 0:
            self.direction = [0.0,0.0]
        else:
            self.direction = distance / numpy.linalg.norm(distance)

        self.speed = (
            abs(distance)
            / (self.instant[1] - self.instant[0])
        )
        self.speed_magnitude = numpy.linalg.norm(self.speed) * (10 ** 10)
        self.speed_magnitude = self.speed_magnitude / self.scale

        self.speed_history[:-1] = self.speed_history[1:]
        self.speed_history[-1] = self.speed_magnitude
        self.speed_magnitude_average = numpy.mean(self.speed_history)

    def update(self, mp):
        self._setup(mp)
        self._translation()
        self._calibration()
        if self.is_calibrated:
            self._scaling()
            self._normalization()
        self._speed_tracking(mp)
    
        if self.is_gesture_catcher:
            if self.gesture_catcher == None:
                self.gesture_catcher = GestureCatcher(self, mp)
            self.gesture_catcher.update(self)




class GestureCatcher:
    def __init__(
        self,
        clock,
        mp
    ):
        self.clock = clock
        self.image_size = mp.image_size

        self.speed_limit = 3
        self.speed_history_size = 5
        self.line_width_default = 40
        self.line_width = 4

        self.speed_history = numpy.zeros(
            [self.speed_history_size], dtype=float
        )

        self.is_catching = False

        self.gesture_image = numpy.full(
            [self.image_size[1], self.image_size[0], 4],
            numpy.zeros([4], dtype=numpy.uint8)
        )

        self.gesture_points = numpy.array([])

    def _scale(self):
        self.line_width = (
            int(
                numpy.clip(
                    self.line_width_default * self.clock.scale,
                    4,
                    2 * self.line_width_default
                )
            )
        )

    def _catch(self):
        if (
            self.clock.speed_magnitude > self.speed_limit
            and self.is_catching == False
        ):
            self.is_catching = True

            self.speed_history = numpy.full(
                [self.speed_history_size], self.speed_limit,dtype=float
            )

            self.gesture_points = numpy.array([])

            self._gesture_image_reset()
        
        if self.is_catching:
            self.speed_history[:-1] = self.speed_history[1:]
            self.speed_history[-1] = self.clock.speed_magnitude
            self.clock.speed_magnitude_average = numpy.mean(self.speed_history)

            if len(self.gesture_points) < 1:
                self.gesture_points = numpy.array(
                    [
                        self.clock.position_abs_history[0],
                        self.clock.position_abs_history[1]
                    ]
                )
            else:
                self.gesture_points = numpy.concatenate(
                    (
                        self.gesture_points,
                        [self.clock.position_abs_history[1]]
                    ),
                    axis=0
                )

            self._gesture_image_update()

        if numpy.mean(self.speed_history) < self.speed_limit:
            self.is_catching = False
    
    def _gesture_image_reset(self):
        self.gesture_image = numpy.full(
            [self.image_size[1], self.image_size[0], 4],
            numpy.zeros([4], dtype=numpy.uint8)
        )
        
    def _gesture_image_update(self):
        color_intensity = self.clock.speed_magnitude_average - self.speed_limit
        color_intensity = color_intensity / (3 * self.speed_limit)
        color_intensity = int(color_intensity * 255)

        # line_width = self.clock.speed_magnitude_average - self.speed_limit
        # line_width = line_width / (3 * self.speed_limit)
        # line_width = int(line_width * 40)
        # line_width = numpy.clip(line_width, 4, 40)

        cv2.line(
            self.gesture_image,
            self.clock.position_abs_history[0],
            self.clock.position_abs_history[1],
            (255, 0, color_intensity),
            self.line_width
        )

    def update(self, clock):
        self.clock = clock
        self._scale()
        self._catch()
