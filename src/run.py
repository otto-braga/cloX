import json
import cv2
import mediapipe
from time import time

from VideoStream import VideoStream
from MediapipeParsed import MediapipeParsed
from Clock import Clock

def main():
    # project load
    # ------------

    project = None
    clocks = []

    with open('examples/TEST-ip_camera.json') as project_file:
        project = json.load(project_file)

    camera_id = project['setup']['camera_id']
    camera_width = project['setup']['camera_width']
    camera_height = project['setup']['camera_height']
    camera_fps = project['setup']['camera_fps']

    min_detect_conf = project['setup']['mp_min_detect_conf']
    min_track_conf = project['setup']['mp_min_track_conf']

    for clock in project['clocks']:
        clocks.append(
            Clock(
                clock['name'],
                clock['i_p_clock'],
                clock['i_p_hand'],
                clock['i_p_ref_A'],
                clock['i_p_ref_B']
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

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # calibration handler
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

        # quitting handler
        # ----------------

        if cv2.waitKey(5) == 27 or cv2.getTrackbarPos('quit', window_title) == 1:
            break

    # close
    # -----
    
    mp_holistic.close()
    video.stop = True
    cv2.destroyAllWindows()

def draw_clock(clock, image):
    print(
        clock.name, 
        " (", clock.r_hand, ", ", clock.phi_r_hand, ")", 
        " (", clock.p_clock, ", ", clock.p_hand, ")"
    )
    color = (0,0,128)
    cv2.circle(image, clock.p_clock, int(clock.r_clock), color, 1)
    cv2.circle(image, clock.p_hand, 4, color, -1)
    cv2.line(image, clock.p_clock, clock.p_hand, color, 2)
    cv2.circle(image, clock.p_clock, 4, color, -1)

if __name__ == '__main__':
    main()

