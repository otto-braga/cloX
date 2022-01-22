import numpy

class MediapipeParsed:
    def __init__(self, image_size):
        self.image_size = image_size
        self.initial_value = -1.0

        self.landmark = numpy.array(
            [
                numpy.full([33,2], self.initial_value, dtype=float),
                numpy.full([21,2], self.initial_value, dtype=float),
                numpy.full([21,2], self.initial_value, dtype=float),
                #numpy.full([468,2], self.initial_value, dtype=float)
            ],
            dtype=object
        )

        # self.solution = [
        #     None,
        #     None,
        #     None
        # ]

        self.solution = [
            None,
            None
        ]

    def update(
        self,
        solution,
        handedness,
        pose_always_in_frame=False,
        skip_when_none=False
    ):
        # solution = [
        #     results.pose_landmarks,
        #     results.right_hand_landmarks,
        #     results.left_hand_landmarks,
        #     #results.face_landmarks
        # ]

        # if len(self.solution) < 1: self.solution = solution

        for i in range(len(solution)):
            if i == 1 and solution[i]:
                self.solution[i] = solution[i]
                if handedness:
                    for hand in range(len(solution[i])):
                        if handedness[hand].classification[0].label == "Left": h = 1
                        elif handedness[hand].classification[0].label == "Right": h = 2
                        for j in range(len(self.landmark[h])):
                            self.landmark[h][j][0] = (
                                solution[i][hand].landmark[j].x * self.image_size[0]
                            )
                            self.landmark[h][j][1] = (
                                solution[i][hand].landmark[j].y * self.image_size[1]
                            )
                        # if len(solution[i]) < 2:
                        #     if h == 1:
                        #         self.landmark[2] = numpy.full([21,2], self.initial_value, dtype=float)
                        #     else:
                        #         self.landmark[1] = numpy.full([21,2], self.initial_value, dtype=float)
                else:
                    self.landmark = numpy.array(
                        [
                            self.landmark[0],
                            numpy.full([21,2], self.initial_value, dtype=float),
                            numpy.full([21,2], self.initial_value, dtype=float),
                            #numpy.full([468,2], self.initial_value, dtype=float)
                        ],
                        dtype=object
                    )

            # elif i == 1 and type(solution[i]) == type(None):
            #     self.landmark = numpy.array(
            #         [
            #             self.landmark[0],
            #             numpy.full([21,2], self.initial_value, dtype=float),
            #             numpy.full([21,2], self.initial_value, dtype=float),
            #             #numpy.full([468,2], self.initial_value, dtype=float)
            #         ],
            #         dtype=object
            #     )

            elif solution[i]:
                self.solution[i] = solution[i]
                for j in range(len(self.landmark[i])):
                    # if skip_when_none and i > 0 and self.landmark[i] == None:
                    #     continue

                    self.landmark[i][j][0] = (
                        solution[i].landmark[j].x * self.image_size[0]
                    )
                    self.landmark[i][j][1] = (
                        solution[i].landmark[j].y * self.image_size[1]
                    )
                    # self.landmark[i][j][2] = (
                    #     solution[i].landmark[j].z * self.image_size[1]
                    # )

                    if pose_always_in_frame and i == 0:
                        self.landmark[i][j] = numpy.clip(
                            self.landmark[i][j], [0,0], self.image_size
                        )

            # elif not skip_when_none:
            #     self.solution[i] = solution[i]
            #     for j in range(len(self.landmark[i])):
            #         self.landmark[i][j] = numpy.full(
            #             [2], self.initial_value, dtype=float
            #         )
