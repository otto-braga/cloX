import time
from threading import Thread

import numpy
import cv2
from pythonosc import udp_client, dispatcher, osc_server




class Clock:
    def __init__(
        self,
        name='',

        i_p_clock = (0,0,-1,-1),
        i_p_hand = (0,0,-1,-1),

        scale_mode = 3,
        i_p_ref_A = (0,0),
        i_p_ref_B = (0,0),
        i_p_ref_C = (0,0),
        i_p_ref_D = (0,0),
        depth_clock_name = '',
        depth_flip = False,
        is_depth_clock = False,

        drawn_gesture_catcher = None,

        is_clipped = True,

        osc_send_ip = '127.0.0.1',
        osc_send_port = 4747,
        osc_rcv_ip = '127.0.0.1',
        osc_rcv_port = 4748
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

        self.depth_clock_name = depth_clock_name
        self.depth_flip = depth_flip
        self.depth_clock_r = 1.0
        self.is_depth_clock = is_depth_clock

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

        self.speed_tracked_point_relative = numpy.zeros([2], dtype=float)
        self.position_history_relative = numpy.full([2, 2], numpy.zeros([2], dtype=int))
        self.position_abs_history_relative = numpy.full([2, 2], numpy.zeros([2], dtype=int))
        self.direction_relative = numpy.zeros([2], dtype=float)
        self.instant_relative = numpy.full([2], time.monotonic_ns(), dtype = int)
        self.speed_relative = [0.0,0.0]
        self.speed_magnitude_relative = 0.0
        self.speed_history_relative = numpy.zeros(
            [5], dtype=float
        )
        self.speed_magnitude_average_relative = 0.0

        self.drawn_gesture_catcher = drawn_gesture_catcher

        self.is_clipped = is_clipped

        self.osc_client = udp_client.SimpleUDPClient(
            osc_send_ip, osc_send_port
        )

        self.osc_dispatcher = dispatcher.Dispatcher()
        self._osc_receive()
        self.osc_server = osc_server.ThreadingOSCUDPServer(
            (osc_rcv_ip, osc_rcv_port), self.osc_dispatcher
        )
        self.osc_server_thread = Thread(
            target = self.osc_server.serve_forever,
            args = ()
        ).start()

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
                i_point_A * mp.image_size,
                dtype=int
            )
        if any(n < 0 for n in i_point_B):
            return mp.landmark[i_point_A[0]][i_point_A[1]]
        
        return self._midpoint(
            mp.landmark[i_point_A[0]][i_point_A[1]],
            mp.landmark[i_point_B[0]][i_point_B[1]]
        )

    def _set_depth(self, address, args):
        self.depth_clock_r = -args if self.depth_flip else args

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

        elif self.scale_mode == 4:
            scale = ((self.depth_clock_r + 1) / 2) + 1

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
            self.x_r_hand =  numpy.clip(self.x_r_hand, -1, 1)
            self.y_r_hand =  numpy.clip(self.y_r_hand, -1, 1)

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

    def _speed_tracking_relative(self, mp):
        self.position_history_relative[0] = self.speed_tracked_point_relative
        self.speed_tracked_point_relative = numpy.clip(
            self.p_norm, [0,0], mp.image_size
        )
        self.position_history_relative[1] = self.speed_tracked_point_relative
        self.position_abs_history_relative = self.position_history_relative.astype(int)
        self.position_history_relative = self.position_history_relative / [1,1]

        self.instant_relative[0] = self.instant_relative[1]
        self.instant_relative[1] = time.monotonic_ns()

        distance = self.position_history_relative[1] - self.position_history_relative[0]

        if numpy.linalg.norm(distance) == 0:
            self.direction_relative = [0.0,0.0]
        else:
            self.direction_relative = distance / numpy.linalg.norm(distance)

        self.speed_relative = (
            abs(distance)
            / (self.instant_relative[1] - self.instant_relative[0])
        )
        self.speed_magnitude_relative = numpy.linalg.norm(self.speed_relative) * (10 ** 10)
        self.speed_magnitude_relative = self.speed_magnitude_relative / self.scale

        self.speed_history_relative[:-1] = self.speed_history_relative[1:]
        self.speed_history_relative[-1] = self.speed_magnitude_relative
        self.speed_magnitude_average_relative = numpy.mean(self.speed_history_relative)

    def _osc_send(self):
        msg = {}
        address = '/cloX/' + self.name + '/'
        
        if self.is_depth_clock:
            msg[address + 'x_r_hand'] = self.x_r_hand
        else:
            msg[address + 'r_hand'] = self.r_hand
            msg[address + 'phi_r_hand'] = self.phi_r_hand
            msg[address + 'x_r_hand'] = self.x_r_hand
            msg[address + 'y_r_hand'] = self.y_r_hand
            msg[address + 'scale'] = self.scale
            msg[address + 'speed'] = self.speed_magnitude
            msg[address + 'speed_average'] = self.speed_magnitude_average
            msg[address + 'direction_x'] = self.direction[0]
            msg[address + 'direction_y'] = self.direction[1]
            msg[address + 'speed_relative'] = self.speed_magnitude_relative
            msg[address + 'speed_average_relative'] = self.speed_magnitude_average_relative
            msg[address + 'direction_x_relative'] = self.direction_relative[0]
            msg[address + 'direction_y_relative'] = self.direction_relative[1]
            msg[address + 'p_clock_norm_x'] = self.p_clock_norm[0]
            msg[address + 'p_clock_norm_y'] = self.p_clock_norm[1]
            msg[address + 'p_hand_norm_x'] = self.p_hand_norm[0]
            msg[address + 'p_hand_norm_y'] = self.p_hand_norm[1]

            if self.drawn_gesture_catcher:
                for drawn_gesture_catcher in self.drawn_gesture_catcher:
                    address_g = (
                        address
                        + 'drawn_gesture/'
                        + drawn_gesture_catcher.name
                        + '/'
                    )

                    msg[address_g + 'is_catching'] = (
                        int(drawn_gesture_catcher.is_catching)
                    )

                    msg[address_g + 'threshold_catch'] = (
                        drawn_gesture_catcher.threshold_catch
                    )

                    msg[address_g + 'threshold_release'] = (
                        drawn_gesture_catcher.threshold_release
                    )

                    msg[address_g + 'speed_history_size'] = (
                        drawn_gesture_catcher.speed_history_size
                    )

                    if (
                        not drawn_gesture_catcher.is_catching
                        and len(drawn_gesture_catcher.gesture_points)
                    ):
                        length = len(drawn_gesture_catcher.gesture_points)

                        msg[address_g + 'gesture_points_length'] = length

                        for i in range(length):
                            msg[address_g + 'gesture_point_x_' + str(i)] = (
                                int(drawn_gesture_catcher.gesture_points[i,0])
                            )
                            msg[address_g + 'gesture_point_y_' + str(i)] = (
                                int(drawn_gesture_catcher.gesture_points[i,1])
                            )

        for address, value in msg.items():
            # print(address, value)
            self.osc_client.send_message(address, value)

    def _osc_receive(self):
        self.osc_dispatcher.map('/cloX/' + self.depth_clock_name + '/x_r_hand', self._set_depth)

    def print_clock(self):
        with numpy.printoptions(precision=3, suppress=True):
            print(
                self.name, '\n',
                "\t| r_hand polar", [self.r_hand, self.phi_r_hand], '\n',
                "\t| r_hand cartesian", [self.x_r_hand, self.y_r_hand], '\n',
                "\t| direction", self.direction, '\n',
                "\t| speed", self.speed_magnitude, '\n',
                "\t| relative speed", self.speed_magnitude_relative, '\n',
                "\t| p_clock", self.p_clock, " | p_hand", self.p_hand, '\n',
                "\t| scale", self.scale
            )

    def draw_clock(self, image):
        color = (0,0,128)
        cv2.circle(image, self.p_clock, int(self.r_clock), color, 1)
        cv2.circle(image, self.p_hand, 4, color, -1)
        cv2.line(image, self.p_clock, self.p_hand, color, 2)
        cv2.circle(image, self.p_clock, 4, color, -1)

        if self.drawn_gesture_catcher:
            for drawn_gesture_catcher in self.drawn_gesture_catcher:
                added_image = cv2.addWeighted(
                    image, 1, drawn_gesture_catcher.gesture_image, 0.5, 0
                )
                image = added_image
        
        return image

    def update(self, mp):
        self._setup(mp)
        self._translation()
        self._calibration()
        if self.is_calibrated:
            self._scaling()
            self._normalization()
            self._speed_tracking_relative(mp)
        self._speed_tracking(mp)
    
        if self.drawn_gesture_catcher:
            for drawn_gesture_catcher in self.drawn_gesture_catcher:
                drawn_gesture_catcher.update(self, mp.image_size)

        self._osc_send()




class DrawnGestureCatcher:
    def __init__(
        self,
        clock,
        name = "",
        threshold_catch = 4,
        threshold_release = 4,
        speed_history_size = 5
    ):
        self.clock = clock
        self.name = name
        self.threshold_catch = threshold_catch
        self.threshold_release = threshold_release
        self.speed_history_size = speed_history_size

        self.speed_history = numpy.zeros(
            [self.speed_history_size], dtype=float
        )

        self.image_size = None

        self.color = (
            numpy.random.randint(0, 127),
            numpy.random.randint(0, 127),
            numpy.random.randint(0, 127)
        )
        self.line_width_default = 40
        self.line_width = 4

        self.is_catching = False
        self.gesture_image = None
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
            self.clock.speed_magnitude > self.threshold_catch
            and self.is_catching == False
        ):
            self.is_catching = True

            self.speed_history = numpy.full(
                [self.speed_history_size], self.threshold_catch,dtype=float
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

        if numpy.mean(self.speed_history) < self.threshold_release:
            self.is_catching = False
    
    def _gesture_image_reset(self):
        self.gesture_image = numpy.full(
            [self.image_size[1], self.image_size[0], 4],
            numpy.zeros([4], dtype=numpy.uint8)
        )
        
    def _gesture_image_update(self):
        color_intensity = self.clock.speed_magnitude_average - self.threshold_catch
        color_intensity = color_intensity / (2 * self.threshold_catch)
        color = (
            int(numpy.clip(color_intensity * self.color[0], 48, 255)),
            int(numpy.clip(color_intensity * self.color[1], 48, 255)),
            int(numpy.clip(color_intensity * self.color[2], 48, 255))
        )

        index = self.clock.drawn_gesture_catcher.index(self)
        line_width = int(
            self.line_width_default / len(self.clock.drawn_gesture_catcher)
        )

        cv2.line(
            self.gesture_image,
            self.clock.position_abs_history[0] + (0, index * line_width),
            self.clock.position_abs_history[1] + (0, index * line_width),
            color,
            int(line_width / 2)
        )

    def update(self, clock, image_size):
        self.clock = clock
        if self.image_size == None:
            self.image_size = image_size
            self._gesture_image_reset()

        self._scale()
        self._catch()
