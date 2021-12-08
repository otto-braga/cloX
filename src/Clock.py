import time
from threading import Thread
import socketserver

import numpy
import cv2
from pythonosc import udp_client, dispatcher, osc_server

from MediapipeParsed import MediapipeParsed




class Clock:
    def __init__(
        self,
        name: str ='',

        i_p_clock: tuple = (0,0,-1,-1),
        i_p_hand: tuple = (0,0,-1,-1),

        scaling_mode: float = 3,
        i_p_ref: list = [(0,0), (0,0), (0,0), (0,0)],

        depth_clock_name: str = '',
        depth_flip: bool = False,
        is_depth_clock: bool = False,

        drawn_gestures: list = [],

        osc_send: list = None,
        osc_rcv: list = None,

        is_clipped: bool = True
    ):
        self.name = name

        self.i_p_clock = numpy.array(i_p_clock)
        self.i_p_hand = numpy.array(i_p_hand)
        
        self.i_p_ref = i_p_ref

        self.p_clock_abs = numpy.zeros([2], dtype=int)
        self.p_hand_abs = numpy.zeros([2], dtype=int)

        self.p_ref = numpy.zeros(
            [4 if scaling_mode == 2 else 8, 2],
            dtype=int
        )

        self.p_hand_translated = numpy.zeros([2], dtype=int)

        self.r_clock_abs = 1.0
        self.r_clock_abs_k = 1.0
        self.r_clock = 1.0
        
        self.m_ref = numpy.ones(
            [int(len(self.p_ref) / 2)],
            dtype=float
        )
        self.m_ref_k = numpy.ones(
            [int(len(self.p_ref) / 2)],
            dtype=float
        )

        self.p_hand = ClockPHand()

        self.enable_calibration = False
        self.is_calibrated = False

        self.scaling_mode = scaling_mode
        self.scale_history = numpy.ones([10], dtype=float)
        self.scale_ratio = 1.0
        self.scale = 1.0

        self.depth_clock_name = depth_clock_name
        self.depth_flip = depth_flip
        self.depth_clock_r = 0.0
        self.is_depth_clock = is_depth_clock

        self.tracker = ClockTracker(self)

        self.drawn_gestures = drawn_gestures

        self.osc_handler = ClockOSCHandler(
            self,
            osc_send,
            osc_rcv
        )

        self.is_clipped = is_clipped

    def _midpoint(self, point_A: tuple, point_B: tuple) -> numpy.ndarray:
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

    def _initialize_point(
        self,
        i_point: tuple,
        mp: MediapipeParsed
    ) -> numpy.ndarray:
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

    def _setup(self) -> None:
        mp = self.mediapipe_results
        self.p_clock_abs = self._initialize_point(self.i_p_clock, mp)
        self.p_hand_abs = self._initialize_point(self.i_p_hand, mp)

        if self.scaling_mode in {1, 2}:
            n = 2 if self.scaling_mode == 1 else 4
            for i in range(n):
                self.p_ref[i] = (
                    mp.landmark[self.i_p_ref[i][0]][self.i_p_ref[i][1]]
                )

        elif self.scaling_mode == 3:
            self.p_ref[1] = (
                self._midpoint(mp.landmark[0][9], mp.landmark[0][10])
            )
            self.p_ref[0] = (
                self._midpoint(mp.landmark[0][3], mp.landmark[0][6])
            )

            self.p_ref[2] = mp.landmark[0][7]
            self.p_ref[3] = mp.landmark[0][8]

            self.p_ref[4] = self._midpoint(
                mp.landmark[0][11], mp.landmark[0][12]
            )
            self.p_ref[5] = self._midpoint(
                mp.landmark[0][23], mp.landmark[0][24]
            )

            self.p_ref[6] = mp.landmark[0][23]
            self.p_ref[7] = mp.landmark[0][24]

    def _translation(self) -> None:
        self.p_hand_translated = self.p_hand_abs - self.p_clock_abs

    def _calibration(self) -> None:
        self.r_clock_abs = numpy.linalg.norm(self.p_hand_translated)

        if self.scaling_mode == 1:
            m_ref = numpy.linalg.norm(self.p_ref[0] - self.p_ref[1])
            self.m_ref[0] = m_ref if m_ref > 0 else self.m_ref[0]

        elif self.scaling_mode in {2, 3}:
            for i, j in zip(
                range(0, len(self.p_ref), 2),
                range(len(self.m_ref))
            ):
                m_ref = numpy.linalg.norm(
                    self.p_ref[i] - self.p_ref[i+1]
                )
                self.m_ref[j] = m_ref if m_ref > 0 else self.m_ref[j]

        if self.enable_calibration:
            self.r_clock_abs_k = self.r_clock_abs

            for i in range(len(self.m_ref)):
                self.m_ref_k[i] = self.m_ref[i]

            self.scale_ratio = self.m_ref_k[0] / self.m_ref_k[1]

            self.enable_calibration = False
            self.is_calibrated = True
        
        if not self.is_calibrated:
            self.p_hand.r = self.r_clock_abs

            self.p_hand.angle = numpy.degrees(
                numpy.arctan2(self.p_hand_abs[0], self.p_hand_abs[1])
            ) 
            self.p_hand.angle = -self.p_hand.angle + 90
            if self.p_hand.angle < 0: self.p_hand.angle = self.p_hand.angle + 360
            self.p_hand.angle = self.p_hand.angle / 360

            self.p_hand.x = self.p_hand_abs[0]
            self.p_hand.y = self.p_hand_abs[1]


    def _scaling(self) -> None:
        scale = 1.0

        if self.scaling_mode == 1:
            scale = self.m_ref[0] / self.m_ref_k[0]

        elif self.scaling_mode == 2:
            scale_AB = self.m_ref[0] / self.m_ref_k[0]
            scale_CD = self.m_ref[1] / self.m_ref_k[1]
            scale = max(scale_AB, scale_CD)

        elif self.scaling_mode == 3:
            ratio = self.m_ref[0] / self.m_ref[1]

            if ratio >= self.scale_ratio:
                scale = self.m_ref[0] / self.m_ref_k[0]
            else:
                scale = self.m_ref[1] / self.m_ref_k[1]

        elif self.scaling_mode == 4:
            scale = ((self.depth_clock_r + 1) / 2) + 1

        self.scale_history[:-1] = self.scale_history[1:]
        self.scale_history[-1] = scale
        self.scale = numpy.mean(self.scale_history)

        self.r_clock = self.r_clock_abs_k * self.scale

    def _normalization(self) -> None:
        p_hand_norm = self.p_hand_translated / self.r_clock

        self.p_hand.x = p_hand_norm[0]
        self.p_hand.y = p_hand_norm[1]

        self.p_hand.r = numpy.linalg.norm(p_hand_norm)

        self.p_hand.angle = numpy.degrees(
            numpy.arctan2(self.p_hand.x, self.p_hand.y)
        ) 
        self.p_hand.angle = -self.p_hand.angle + 90
        if self.p_hand.angle < 0: self.p_hand.angle = self.p_hand.angle + 360
        self.p_hand.angle = self.p_hand.angle / 360

        if self.is_clipped:
            self.p_hand.r =  numpy.clip(self.p_hand.r, 0, 1)
            self.p_hand.angle =  numpy.clip(self.p_hand.angle, 0, 1)
            self.p_hand.x =  numpy.clip(self.p_hand.x, -1, 1)
            self.p_hand.y =  numpy.clip(self.p_hand.y, -1, 1)

    def print_clock(self) -> None:
        with numpy.printoptions(precision=3, suppress=True):
            print(
                self.name, '\n',
                "\t| p_hand", [
                    self.p_hand.r, self.p_hand.angle, self.p_hand.x, self.p_hand.y 
                ], '\n',
                "\t| direction", self.tracker.direction, '\n',
                "\t| speed", self.tracker.speed, '\n',
                "\t| p_clock_abs", self.p_clock_abs, " | p_hand_abs", self.p_hand_abs, '\n',
                "\t| scale", self.scale
            )

    def draw_clock(self, image: numpy.ndarray) -> None:
        color = (0,0,128)
        cv2.circle(image, self.p_clock_abs.astype(int), int(self.r_clock), color, 1)
        cv2.line(image, self.p_clock_abs.astype(int), self.p_hand_abs.astype(int), color, 2)

        if self.drawn_gestures:
            for drawn_gesture in self.drawn_gestures:
                added_image = cv2.addWeighted(
                    image, 1, drawn_gesture.image, 0.5, 0
                )
                image = added_image
        
        return image

    def update(self, mediapipe_results_parsed: MediapipeParsed) -> None:
        self.mediapipe_results = mediapipe_results_parsed

        self.osc_handler.update(self)

        self._setup()
        self._translation()
        self._calibration()
        if self.is_calibrated:
            self._scaling()
            self._normalization()
        
        self.tracker.update(self)
    
        if self.drawn_gestures:
            for drawn_gesture in self.drawn_gestures:
                drawn_gesture.update(self)




class ClockPHand:
    def __init__(self):
        self.r = 0.0
        self.angle = 0.0
        self.x = 0.0
        self.y = 0.0
        self.direction = numpy.zeros([2], dtype=float)
        self.velocity = numpy.zeros([2], dtype=float)
        self.speed = 0.0
        self.speed_average = 0.0




class ClockTracker:
    def __init__(
        self,
        clock: Clock = None
    ):
        self.clock = clock

        # self.tracked_point = numpy.zeros([2], dtype=float)
        self.position = numpy.full([2, 2], numpy.zeros([2], dtype=float))
        self.position_abs = numpy.full([2, 2], numpy.zeros([2], dtype=int))
        self.direction = numpy.zeros([2], dtype=float)
        self.instant = numpy.full([2], time.monotonic_ns(), dtype = int)
        self.velocity = numpy.zeros([2], dtype=float)
        self.speed = 0.0

    def _speed_tracking(self):
        self.position[0] = self.position[1]
        self.position[1] = [self.clock.p_hand.x, self.clock.p_hand.y]

        self.position_abs[0] = self.position_abs[1]
        self.position_abs[1] = self.clock.p_hand_abs

        self.instant[0] = self.instant[1]
        self.instant[1] = time.monotonic_ns()

        distance = self.position[1] - self.position[0]

        if numpy.linalg.norm(distance) == 0:
            self.direction = [0.0,0.0]
        else:
            self.direction = distance / numpy.linalg.norm(distance)

        self.velocity = (
            abs(distance)
            / (self.instant[1] - self.instant[0])
        )
        self.speed = numpy.linalg.norm(self.velocity) * (10 ** 10)
        self.speed = self.speed / self.clock.scale

        self.clock.p_hand.direction = self.direction
        self.clock.p_hand.velocity = self.velocity
        self.clock.p_hand.speed = self.speed
    
    def update(self, clock):
        self.clock = clock
        self._speed_tracking()




class ClockOSCHandler:
    def __init__(
        self,
        clock: Clock = None,
        osc_clients: list = None,
        osc_receivers: list = None
    ):
        self.clock = clock

        self.osc_clients = osc_clients
        if osc_clients != None:
            self.osc_clients = [
                udp_client.SimpleUDPClient(
                    address["ip"],
                    address["port"]
                ) for address in osc_clients
            ]

        self.osc_receivers = osc_receivers
        if osc_receivers != None:
            self.osc_receivers = [
                (
                    address["ip"],
                    address["port"]
                ) for address in osc_receivers
            ]
        
        self.osc_dispatcher = None
        self.osc_servers = None
        self.osc_servers_threads = None

    def _osc_send(self) -> None:
        msg = {}
        address = '/cloX/' + self.clock.name + '/'
        
        if self.clock.is_depth_clock:
            msg[address + 'x'] = self.clock.p_hand.x
        else:
            msg[address + 'r'] = self.clock.p_hand.r
            msg[address + 'angle'] = self.clock.p_hand.angle
            msg[address + 'x'] = self.clock.p_hand.x
            msg[address + 'y'] = self.clock.p_hand.y
            msg[address + 'scale'] = self.clock.scale
            msg[address + 'speed'] = self.clock.p_hand.speed
            msg[address + 'speed_average'] = self.clock.p_hand.speed_average
            msg[address + 'direction_x'] = self.clock.p_hand.direction[0]
            msg[address + 'direction_y'] = self.clock.p_hand.direction[1]

            if self.clock.drawn_gestures:
                for drawn_gesture in self.clock.drawn_gestures:
                    address_g = (
                        address
                        + 'drawn_gesture/'
                        + drawn_gesture.name
                        + '/'
                    )

                    msg[address_g + 'is_catching'] = (
                        int(drawn_gesture.is_catching)
                    )

                    msg[address_g + 'threshold_catch'] = (
                        drawn_gesture.threshold_catch
                    )

                    msg[address_g + 'threshold_release'] = (
                        drawn_gesture.threshold_release
                    )

                    msg[address_g + 'speed_history_size'] = (
                        drawn_gesture.speed_history_size
                    )

                    if (
                        not drawn_gesture.is_catching
                        and len(drawn_gesture.points)
                    ):
                        length = len(drawn_gesture.points)

                        msg[address_g + 'points_length'] = length

                        for i in range(length):
                            msg[address_g + 'point_x_' + str(i)] = (
                                int(drawn_gesture.points[i,0])
                            )
                            msg[address_g + 'point_y_' + str(i)] = (
                                int(drawn_gesture.points[i,1])
                            )

        for address, value in msg.items():
            # print(address, value)
            for client in self.osc_clients:
                client.send_message(address, value)
            
    def _osc_map(self) -> None:
        self.osc_dispatcher.map(
            '/cloX/' + self.clock.depth_clock_name + '/x',
            self._osc_receive_depth
        )
    
    def _osc_receive_depth(self, address: str, args: float) -> None:
        self.depth_clock_r = -args if self.clock.depth_flip else args

    def _osc_server_init(self) -> None:
        if self.clock.scaling_mode == 4 and self.osc_receivers != None:
            if self.osc_dispatcher == None:
                self.osc_dispatcher = dispatcher.Dispatcher()
                self._osc_map()
            if self.osc_servers == None:
                socketserver.UDPServer.allow_reuse_address = True
                self.osc_servers = [
                    osc_server.ThreadingOSCUDPServer(
                        (receiver[0], receiver[1]), self.osc_dispatcher
                    ) for receiver in self.osc_receivers
                ]
            if self.osc_servers_threads == None:
                self.osc_servers_threads = [
                    Thread(
                        target = server.serve_forever,
                        args = ()
                    ).start() for server in self.osc.servers
                ]

    def update(self, clock: Clock) -> None:
        self.clock = clock
        self._osc_server_init()
        self._osc_send()




class ClockDrawnGesture:
    def __init__(
        self,
        clock: Clock = None,
        name: str = "",
        threshold_catch: float = 4,
        threshold_release: float = 4,
        speed_history_size: int = 5
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
        self.image = None
        self.points = numpy.array([])

    def _scale(self) -> None:
        self.line_width = (
            int(
                numpy.clip(
                    self.line_width_default * self.clock.scale,
                    4,
                    2 * self.line_width_default
                )
            )
        )

    def _catch(self) -> None:
        if (
            self.clock.p_hand.speed > self.threshold_catch
            and self.is_catching == False
        ):
            self.is_catching = True

            self.speed_history = numpy.full(
                [self.speed_history_size], self.threshold_catch,dtype=float
            )

            self.points = numpy.array([])

            self._gesture_image_reset()
        
        if self.is_catching:
            self.speed_history[:-1] = self.speed_history[1:]
            self.speed_history[-1] = self.clock.p_hand.speed
            self.clock.p_hand.speed_average = numpy.mean(self.speed_history)

            if len(self.points) < 1:
                self.points = numpy.array(
                    [
                        self.clock.tracker.position_abs[0],
                        self.clock.tracker.position_abs[1]
                    ]
                )
            else:
                self.points = numpy.concatenate(
                    (
                        self.points,
                        [self.clock.tracker.position_abs[1]]
                    ),
                    axis=0
                )

            self._gesture_image_update()

        if numpy.mean(self.speed_history) < self.threshold_release:
            self.is_catching = False
    
    def _gesture_image_reset(self) -> None:
        self.image = numpy.full(
            [self.image_size[1], self.image_size[0], 4],
            numpy.zeros([4], dtype=numpy.uint8)
        )
        
    def _gesture_image_update(self) -> None:
        color_intensity = self.clock.p_hand.speed_average - self.threshold_catch
        color_intensity = color_intensity / (2 * self.threshold_catch)
        color = (
            int(numpy.clip(color_intensity * self.color[0], 48, 255)),
            int(numpy.clip(color_intensity * self.color[1], 48, 255)),
            int(numpy.clip(color_intensity * self.color[2], 48, 255))
        )

        index = self.clock.drawn_gestures.index(self)
        line_width = int(
            self.line_width_default / len(self.clock.drawn_gestures)
        )

        cv2.line(
            self.image,
            self.clock.tracker.position_abs[0] + (0, index * line_width),
            self.clock.tracker.position_abs[1] + (0, index * line_width),
            color,
            int(line_width / 2)
        )

    def update(self, clock: Clock) -> None:
        self.clock = clock
        if self.image_size == None:
            self.image_size = self.clock.mediapipe_results.image_size
            self._gesture_image_reset()

        self._scale()
        self._catch()




def load_from_project(project: dict) -> list:
    clocks = []

    for clock in project['clocks']:
        clock_new = Clock(
            name = clock["name"],
            i_p_clock = clock["i_p_clock"],
            i_p_hand = clock["i_p_hand"],
            osc_send = project['setup']['osc_send'] if "osc_send" in project['setup'] else None,
            osc_rcv = project['setup']['osc_rcv'] if "osc_rcv" in project['setup'] else None
        )

        if "scale_mode" in clock:
            clock_new.scaling_mode = clock["scale_mode"]

            if 0 < clock["scale_mode"] < 3:
                clock_new.i_p_ref[0] = clock["i_p_ref_A"]
                clock_new.i_p_ref[1] = clock["i_p_ref_B"]
                if clock["scale_mode"] == 2:
                    clock_new.i_p_ref[2] = clock["i_p_ref_C"]
                    clock_new.i_p_ref[3] = clock["i_p_ref_D"]
            elif clock['scale_mode'] == 4:
                clock_new.depth_clock_name = clock['depth_clock_name']
                if 'depth_flip' in clock:
                    clock_new.depth_flip = bool(clock['depth_flip'])
        else:
            clock_new.scaling_mode = 3

        if "drawn_gesture_catcher" in clock:
            clock_new.drawn_gestures = []

            for drawn_gesture in clock["drawn_gesture_catcher"]:
                drawn_gesture_new = ClockDrawnGesture(
                    clock = clock_new,
                    name = drawn_gesture['name'],
                    threshold_catch = drawn_gesture['threshold_catch'],
                    threshold_release = (
                        drawn_gesture['threshold_release']
                    ),
                    speed_history_size = (
                        drawn_gesture['speed_history_size']
                    )
                )

                clock_new.drawn_gestures.append(
                    drawn_gesture_new
                )

        if "is_clipped" in clock:
            clock_new.is_clipped = bool(clock["is_clipped"])
        else:
            clock_new.is_clipped = True

        if "is_depth_clock" in clock:
            clock_new.is_depth_clock = bool(clock['is_depth_clock'])

        clocks.append(clock_new)

    return clocks
