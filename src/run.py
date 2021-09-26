import json
import sys
import cv2
import mediapipe
from time import time

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
        clocks.append(
            Clock(
                clock["name"],
                clock["i_p_clock"],
                clock["i_p_hand"],
                clock["scale_mode"],
                clock["i_p_ref_A"],
                clock["i_p_ref_B"],
                clock["i_p_ref_C"],
                clock["i_p_ref_D"]
            )
        )

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
    cv2.createTrackbar('calibration delay (s)', window_title, 0, 10, (lambda a: None))
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

        mp_parsed = MediapipeParsed(mp_results, (image.shape[1], image.shape[0]))

        mediapipe.solutions.drawing_utils.draw_landmarks(
            image,
            mp_results.pose_landmarks,
            mediapipe.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mediapipe.solutions.drawing_styles.get_default_pose_landmarks_style()
        )

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
            clock.en_c = calibrate
            clock.update(mp_parsed)
            draw_clock(clock, image)

        cv2.imshow(window_title, image)

        # send OSC messages
        # -------------------

        osc_messages = osc.make_messages(clocks)
        osc.send(osc_client, osc_messages)

        # OpenCV window quitting manager
        # ------------------------------

        if cv2.waitKey(5) == 27 or cv2.getTrackbarPos('quit', window_title) == 1:
            break

    # quit
    # ----
    
    mp_holistic.close()
    video.stop = True
    cv2.destroyAllWindows()

def draw_clock(clock, image):
    print(
        clock.name, 
        " | p_hand polar (", clock.r_hand, ", ", clock.phi_r_hand, ")", 
        " | p_clock ", clock.p_clock,
        " | p_hand ", clock.p_hand,
        " | scale ", clock.scale
    )
    color = (0,0,128)
    cv2.circle(image, clock.p_clock, int(clock.r_clock), color, 1)
    cv2.circle(image, clock.p_hand, 4, color, -1)
    cv2.line(image, clock.p_clock, clock.p_hand, color, 2)
    cv2.circle(image, clock.p_clock, 4, color, -1)

if __name__ == '__main__':
    main()

