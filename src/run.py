import json
import sys
from time import time

import numpy
import cv2
import mediapipe

from VideoStream import VideoStream
from MediapipeParsed import MediapipeParsed
import Clock

import osc

def main():
    # project load
    # ------------

    project_path = sys.argv[1]
    project = None
    with open(project_path) as project_file:
        project = json.load(project_file)

    camera_id = project['setup']['camera_id']
    camera_width = project['setup']['camera_width']
    camera_height = project['setup']['camera_height']
    camera_fps = project['setup']['camera_fps']

    min_detect_conf = project['setup']['mp_min_detect_conf']
    min_track_conf = project['setup']['mp_min_track_conf']

    clocks = Clock.load_from_project(project)

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
            clock.print_clock()
            image = clock.draw_clock(image)

        cv2.imshow(window_title, image)

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
    for clock in clocks:
        if clock.osc_server_thread:
            clock.osc_server_thread.join()
    cv2.destroyAllWindows()

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
