import time
from threading import Thread
import socketserver

import numpy
import cv2
from pythonosc import udp_client, dispatcher, osc_server

from Clock import Clock




class ClockPHand:
    def __init__(
        self,
        clock: Clock = None
    ):
        self.clock = None
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

        self.tracked_point = numpy.zeros([2], dtype=float)
        self.position = numpy.full([2, 2], numpy.zeros([2], dtype=int))
        self.position_abs = numpy.full([2, 2], numpy.zeros([2], dtype=int))
        self.direction = numpy.zeros([2], dtype=float)
        self.instant = numpy.full([2], time.monotonic_ns(), dtype = int)
        self.velocity = numpy.zeros([2], dtype=float)
        self.speed = 0.0

    def _speed_tracking(self):
        self.position[0] = self.tracked_point

        self.tracked_point = numpy.clip(
            [self.clock.p_hand.x, self.clock.p_hand.y],
            [-1,-1] if self.clock.is_calibrated else [0,0],
            [1,1] if self.clock.is_calibrated else self.clock.mediapipe_results.image_size
        )

        self.position[1] = self.tracked_point
        self.position_abs = self.position.astype(int)
        self.position = (
            self.position / self.clock.mediapipe_results.image_size
        )

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
        self.speed = self.speed / self.scale

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
            self.osc_client = [
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
        
        if self.is_depth_clock:
            msg[address + 'x'] = self.clock.x_r_hand
        else:
            msg[address + 'r'] = self.clock.r_hand
            msg[address + 'angle'] = self.clock.phi_r_hand
            msg[address + 'x'] = self.clock.x_r_hand
            msg[address + 'y'] = self.clock.y_r_hand
            msg[address + 'scale'] = self.clock.scale
            msg[address + 'speed'] = self.clock.tracker.speed
            msg[address + 'speed_average'] = self.clock.tracker.speed_average
            msg[address + 'direction_x'] = self.clock.tracker.direction[0]
            msg[address + 'direction_y'] = self.clock.tracker.direction[1]

            if self.clock.drawn_gestures:
                for drawn_gesture_catcher in self.clock.drawn_gestures:
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
            for client in self.osc_clients:
                client.send_message(address, value)
            
    def _osc_map(self) -> None:
        self.osc_dispatcher.map(
            '/cloX/' + self.clock.depth_clock_name + '/x',
            self._osc_receive_depth
        )
    
    def _osc_receive_depth(self, address: str, args: float) -> None:
        self.depth_clock_r = -args if self.depth_flip else args

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
                        self.clock.position_abs_history[0],
                        self.clock.position_abs_history[1]
                    ]
                )
            else:
                self.points = numpy.concatenate(
                    (
                        self.points,
                        [self.clock.position_abs_history[1]]
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
            self.clock.position_abs_history[0] + (0, index * line_width),
            self.clock.position_abs_history[1] + (0, index * line_width),
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




class ClockList:
    def __init___(self):
        self.clocks = []
    
    def load_from_project(self, project: dict) -> list:
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

            self.clocks.append(clock_new)

        return self.clocks
