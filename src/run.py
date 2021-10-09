import json
import sys
from time import time

import numpy
import cv2
import mediapipe
import keras

from VideoStream import VideoStream
from MediapipeParsed import MediapipeParsed
from Clock import Clock

import osc

def main():
    # project load
    # ------------

    project_path = sys.argv[1]

    project = None
    clocks = []

    with open(project_path) as project_file:
        project = json.load(project_file)

    camera_id = project['setup']['camera_id']
    camera_width = project['setup']['camera_width']
    camera_height = project['setup']['camera_height']
    camera_fps = project['setup']['camera_fps']

    min_detect_conf = project['setup']['mp_min_detect_conf']
    min_track_conf = project['setup']['mp_min_track_conf']

    osc_ip = project['setup']['osc_ip']
    osc_port = project['setup']['osc_port']

    for clock in project['clocks']:
        scale_mode = clock["scale_mode"]
        clock_new = Clock(
            name = clock["name"],
            i_p_clock = clock["i_p_clock"],
            i_p_hand = clock["i_p_hand"],
            scale_mode = clock["scale_mode"],
            i_p_ref_A = clock["i_p_ref_A"] if 0 < scale_mode < 3 else [0,0],
            i_p_ref_B = clock["i_p_ref_B"] if 0 < scale_mode < 3 else [0,0],
            i_p_ref_C = clock["i_p_ref_C"] if scale_mode == 2 else [0,0],
            i_p_ref_D = clock["i_p_ref_D"] if scale_mode == 2 else [0,0],
            is_gesture_catcher = bool(clock["is_gesture_catcher"]),
            is_gesture_classifier = bool(clock["is_gesture_classifier"]),
            is_clipped = bool(clock["is_clipped"])
        )
        if clock_new.is_gesture_catcher:
            clock_new.gesture_catcher_model = keras.models.load_model(
                'gesture/model.h5'
            )
        clocks.append(clock_new)

    # initialization
    # --------------

    video = VideoStream(
        source = camera_id,
        width = camera_width,
        height = camera_height,
        fps = camera_fps
    ).start()
    print(video.width, video.height)
    print(video.fps)

    mp_holistic =  mediapipe.solutions.holistic.Holistic(
        min_detection_confidence = min_detect_conf,
        min_tracking_confidence = min_track_conf
    )

    mp_parsed = MediapipeParsed([camera_width, camera_height])

    osc_client = osc.client_setup(osc_ip, osc_port)

    # OpenCV window
    # -------------

    window_title = (
        'cloX ['
        + str(camera_width)+ 'x' + str(camera_height) + ', '
        + str(camera_fps) + ' FPS]'
    )
    cv2.namedWindow(window_title)
    cv2.createTrackbar('camera gain', window_title, 0, 8, (lambda a: None))
    cv2.createTrackbar('calibrate', window_title, 0, 1, (lambda a: None))
    cv2.createTrackbar(
        'calibration delay (s)', window_title, 0, 10, (lambda a: None)
    )
    cv2.createTrackbar('quit', window_title, 0, 1, (lambda a: None))

    calibration_start_time = None
    calibration_delay = None

    # start video capture
    # -------------------

    while video.stream.isOpened():
        # get video frame
        # ---------------

        video.stream.set(14, cv2.getTrackbarPos('camera gain', window_title))

        image = video.frame
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # mediapipe
        # ---------

        image.flags.writeable = False
        mp_results = mp_holistic.process(image)
        image.flags.writeable = True

        mp_parsed.update(mp_results, True, False)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # OpencCV window calibration manager
        # -------------------

        if (
            calibration_start_time == None
            and (
                cv2.getTrackbarPos('calibrate', window_title) == 1
                or cv2.waitKey(5) == ord('r')
            )
        ):
            calibration_start_time = time()
            calibration_delay = cv2.getTrackbarPos(
                'calibration delay (s)', window_title
            )
            cv2.setTrackbarPos('calibrate', window_title, 1)
        
        if  (
            calibration_start_time != None 
            and calibration_delay != None
            and (time() - calibration_start_time) > calibration_delay
        ):
            calibration_start_time = None
            calibration_delay = None
            calibrate = True
            cv2.setTrackbarPos('calibrate', window_title, 0)
        else:
            calibrate = False

        # update clocks
        # -------------

        for clock in clocks:
            clock.enable_calibration = calibrate
            clock.update(mp_parsed)

        # draw image
        # ----------

        image = numpy.zeros(image.shape, dtype=numpy.uint8)

        draw_mediapipe_results(mp_parsed.solution, image)
        # draw_mediapipe_parsed_landmarks(mp_parsed, image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)

        for clock in clocks:
            print_clock(clock)
            image = draw_clock(clock, image)
        print('\n')

        cv2.imshow(window_title, image)

        # send OSC messages
        # -------------------

        osc_messages = osc.make_messages(clocks)
        osc.send(osc_client, osc_messages)

        # OpenCV window quitting manager
        # ------------------------------

        if (
            cv2.waitKey(5) == 27
            or cv2.getTrackbarPos('quit', window_title) == 1
        ):
            break

    # quit
    # ----
    
    mp_holistic.close()
    video.stop = True
    cv2.destroyAllWindows()

def print_clock(clock):
    with numpy.printoptions(precision=3, suppress=True):
        print(
            clock.name, '\n',
            "\t| r_hand polar", [clock.r_hand, clock.phi_r_hand], '\n',
            "\t| r_hand cartesian", [clock.x_r_hand, clock.y_r_hand], '\n',
            "\t| direction", clock.direction, '\n',
            "\t| speed", clock.speed_magnitude, '\n',
            "\t| p_clock", clock.p_clock, " | p_hand", clock.p_hand, '\n',
            "\t| scale", clock.scale
        )
        if clock.is_gesture_classifier:
            print(
                "\t| gesture_classification [class, acc]",
                [
                    clock.gesture_classification,
                    clock.gesture_classification_accuracy
                ], '\n',
                "\t| classification_array",
                clock.gesture_classification_array
            )
    

def draw_clock(clock, image):
    color = (0,0,128)
    cv2.circle(image, clock.p_clock, int(clock.r_clock), color, 1)
    cv2.circle(image, clock.p_hand, 4, color, -1)
    cv2.line(image, clock.p_clock, clock.p_hand, color, 2)
    cv2.circle(image, clock.p_clock, 4, color, -1)

    if clock.is_gesture_catcher:
        added_image = cv2.addWeighted(
            image, 1, clock.gesture_catcher.gesture_image, 0.5, 0
        )
        image = added_image
    
    return image

def draw_mediapipe_results(mp_results, image):
    mp_solutions = mediapipe.solutions
    mp_solutions.drawing_utils.draw_landmarks(
        image,
        mp_results[0],
        mp_solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec = (
            mp_solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    )
    mp_solutions.drawing_utils.draw_landmarks(
        image,
        mp_results[1],
        mp_solutions.hands.HAND_CONNECTIONS,
        landmark_drawing_spec = (
            mp_solutions.drawing_styles.get_default_hand_landmarks_style()
        )
    )
    mp_solutions.drawing_utils.draw_landmarks(
        image,
        mp_results[2],
        mp_solutions.hands.HAND_CONNECTIONS,
        landmark_drawing_spec = (
            mp_solutions.drawing_styles.get_default_hand_landmarks_style()
        )
    )

def draw_mediapipe_parsed_landmarks(mp_parsed, image):
    for i in range(len(mp_parsed.landmark)):
        for j in range(len(mp_parsed.landmark[i])):
            cv2.circle(image, mp_parsed.landmark[i][j], 4, (255,255,255))

if __name__ == '__main__':
    main()

