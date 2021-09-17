import numpy

class MediapipeParsed:
    def __init__(self, results, image_size):
        self.image_size = image_size
        self.initial_value = -1

        self.landmark = numpy.array(
            [
                numpy.full([33,2], self.initial_value, dtype=int),
                numpy.full([21,2], self.initial_value, dtype=int),
                numpy.full([21,2], self.initial_value, dtype=int),
                #numpy.full([468,2], self.initial_value, dtype=int)
            ],
            dtype=object
        )

        solution = [
            results.pose_landmarks,
            results.right_hand_landmarks,
            results.left_hand_landmarks,
            #results.face_landmarks
        ]

        for i in range(len(solution)):
            if solution[i] != None:
                for j in range(len(self.landmark[i])):
                    self.landmark[i][j][0] = solution[i].landmark[j].x * image_size[0]
                    self.landmark[i][j][1] = solution[i].landmark[j].y * image_size[1]
                    #self.landmark[i][j][2] = solution[i].landmark[j].z * image_size[1]
            else:
                for j in range(len(self.landmark[i])):
                    self.landmark[i][j] = numpy.full([2], self.initial_value, dtype=int)
