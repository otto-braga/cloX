import numpy

class MediapipeParsed:
    def __init__(self, results, image_size):
        self.image_size = image_size
        self.initial_value = 0

        self.landmark = numpy.array(
            [
                numpy.full([33,3], self.initial_value, dtype=float),
                numpy.full([21,3], self.initial_value, dtype=float),
                numpy.full([21,3], self.initial_value, dtype=float),
                #numpy.full([468,3], self.initial_value, dtype=int)
            ],
            dtype=object
        )

        solution = [
            results.pose_world_landmarks,
            results.right_hand_landmarks,
            results.left_hand_landmarks,
            #results.face_landmarks
        ]

        for i in range(len(solution)):
            if solution[i] != None:
                for j in range(len(self.landmark[i])):
                    self.landmark[i][j][0] = solution[i].landmark[j].x
                    self.landmark[i][j][1] = solution[i].landmark[j].y
                    self.landmark[i][j][2] = solution[i].landmark[j].z
            else:
                for j in range(len(self.landmark[i])):
                    self.landmark[i][j] = numpy.full(
                        [3],
                        self.initial_value,
                        dtype=float
                    )
