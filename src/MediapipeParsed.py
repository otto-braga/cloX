import numpy

class MediapipeParsed:
    def __init__(self, image_size):
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

        self.solution = []

    def update(self, results, skip_none=False):
        solution = [
            results.pose_landmarks,
            results.right_hand_landmarks,
            results.left_hand_landmarks,
            #results.face_landmarks
        ]

        if len(self.solution) < 1: self.solution = solution

        for i in range(len(solution)):
            if solution[i] != None:
                self.solution[i] = solution[i]
                for j in range(len(self.landmark[i])):
                    if solution[i].landmark[j].visibility > 0.5:
                        self.landmark[i][j][0] = solution[i].landmark[j].x * self.image_size[0]
                        self.landmark[i][j][1] = solution[i].landmark[j].y * self.image_size[1]
                        #self.landmark[i][j][2] = solution[i].landmark[j].z * image_size[1]
            elif skip_none == False:
                self.solution[i] = solution[i]
                for j in range(len(self.landmark[i])):
                    self.landmark[i][j] = numpy.full([2], self.initial_value, dtype=int)
