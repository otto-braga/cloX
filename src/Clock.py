import numpy

class Clock:
    def __init__(
        self,
        name='',

        p_clock_mode = 0,
        index_p_clock_A = (0,0),
        index_p_clock_B = (0,0),

        p_hand_mode = 0,
        index_p_hand_A = (0,0),
        index_p_hand_B = (0,0),

        scale_mode = 0,
        index_p_ref_A = (0,0),
        index_p_ref_B = (0,0),
        index_p_ref_C = (0,0),
        index_p_ref_D = (0,0)
    ):
        self.name = name

        self.i_p_clock_A = numpy.array(index_p_clock_A)
        self.i_p_clock_B = numpy.array(index_p_clock_B)
        self.i_p_hand_A = numpy.array(index_p_hand_A)
        self.i_p_hand_B = numpy.array(index_p_hand_B)
        self.i_p_ref_A = numpy.array(index_p_ref_A)
        self.i_p_ref_B = numpy.array(index_p_ref_B)
        self.i_p_ref_C = numpy.array(index_p_ref_C)
        self.i_p_ref_D = numpy.array(index_p_ref_D)

        self.p_clock = numpy.zeros([2], dtype=int)
        self.p_hand = numpy.zeros([2], dtype=int)
        self.p_ref_A = numpy.zeros([2], dtype=int)
        self.p_ref_B = numpy.zeros([2], dtype=int)
        self.p_ref_C = numpy.zeros([2], dtype=int)
        self.p_ref_D = numpy.zeros([2], dtype=int)

        self.p_clock_norm = numpy.zeros([2], dtype=int)
        self.p_hand_norm = numpy.zeros([2], dtype=int)

        self.p_tran = numpy.zeros([2], dtype=int)
        self.p_norm = numpy.zeros([2], dtype=int)
        self.m_clock = 1.0
        self.m_ref_AB = 1.0
        self.m_ref_CD = 1.0
        self.k_m_ref_AB = 1.0
        self.k_m_ref_CD = 1.0
        self.k_r_clock = 1.0

        self.scale = 1.0
        self.r_clock = 1.0

        self.r_hand = 0.0
        self.phi_r_hand = 0.0
        self.x_r_hand = 0.0
        self.y_r_hand = 0.0

        self.p_clock_mode = p_clock_mode
        self.p_hand_mode = p_hand_mode
        self.scale_mode = scale_mode
        self.en_c = False
        self.flag_c = False

    def _setup(self, mp):
        if self.p_clock_mode == 0:
            p_clock = mp.landmark[self.i_p_clock_A[0]][self.i_p_clock_A[1]]
        elif self.p_clock_mode == 1:
            p_clock_A = mp.landmark[self.i_p_clock_A[0]][self.i_p_clock_A[1]]
            p_clock_B = mp.landmark[self.i_p_clock_B[0]][self.i_p_clock_B[1]]
            p_clock = numpy.array(
                [
                    min(p_clock_A[0], p_clock_B[0])
                    + (abs(p_clock_A[0] - p_clock_B[0]) / 2),

                    min(p_clock_A[1], p_clock_B[1])
                    + (abs(p_clock_A[1] - p_clock_B[1]) / 2),
                ],
                dtype=int
            )
        else:
            p_clock = numpy.array(
                numpy.abs(self.i_p_clock_A) * mp.image_size,
                dtype=int
            )
        
        if self.p_hand_mode == 0:
            p_hand = mp.landmark[self.i_p_hand_A[0]][self.i_p_hand_A[1]]
        elif self.p_hand_mode == 1:
            p_hand_A = mp.landmark[self.i_p_hand_A[0]][self.i_p_hand_A[1]]
            p_hand_B = mp.landmark[self.i_p_hand_B[0]][self.i_p_hand_B[1]]
            p_hand = numpy.array(
                [
                    min(p_hand_A[0], p_hand_B[0])
                    + (abs(p_hand_A[0] - p_hand_B[0]) / 2),

                    min(p_hand_A[1], p_hand_B[1])
                    + (abs(p_hand_A[1] - p_hand_B[1]) / 2),
                ],
                dtype=int
            )
        else:
            p_hand = numpy.array(
                numpy.abs(self.i_p_hand_A) * mp.image_size,
                dtype=int
            )

        if self.scale_mode == 0:
            p_ref_A = numpy.zeros([2], dtype=int)
            p_ref_B = numpy.zeros([2], dtype=int)
            p_ref_C = numpy.zeros([2], dtype=int)
            p_ref_D = numpy.zeros([2], dtype=int)
        elif self.scale_mode == 1:
            p_ref_A = mp.landmark[self.i_p_ref_A[0]][self.i_p_ref_A[1]]
            p_ref_B = mp.landmark[self.i_p_ref_B[0]][self.i_p_ref_B[1]]
            p_ref_C = numpy.zeros([2], dtype=int)
            p_ref_D = numpy.zeros([2], dtype=int)
        else:
            p_ref_A = mp.landmark[self.i_p_ref_A[0]][self.i_p_ref_A[1]]
            p_ref_B = mp.landmark[self.i_p_ref_B[0]][self.i_p_ref_B[1]]
            p_ref_C = mp.landmark[self.i_p_ref_C[0]][self.i_p_ref_C[1]]
            p_ref_D = mp.landmark[self.i_p_ref_D[0]][self.i_p_ref_D[1]]

        if (
            numpy.all(p_clock > 0)
            and numpy.all(p_hand > 0)
            and numpy.all(p_ref_A > 0)
            and numpy.all(p_ref_B > 0)
            and numpy.all(p_ref_C > 0)
            and numpy.all(p_ref_D > 0)
        ):
            self.p_clock = p_clock
            self.p_hand = p_hand

            self.p_ref_A = p_ref_A
            self.p_ref_B = p_ref_B
            self.p_ref_C = p_ref_C
            self.p_ref_D = p_ref_D

            self.p_clock_norm = p_clock / mp.image_size
            self.p_hand_norm = p_hand / mp.image_size

    def _translation(self):
        self.p_tran = self.p_hand - self.p_clock

    def _calibration(self):
        self.m_clock = numpy.linalg.norm(self.p_tran)

        if self.scale_mode == 0:
            self.m_ref_AB = 1
            self.m_ref_CD = 1
        elif self.scale_mode == 1:
            m_ref = numpy.linalg.norm(self.p_ref_A - self.p_ref_B)
            self.m_ref_AB = m_ref if m_ref > 0 else self.m_ref_AB
            self.m_ref_CD = 1.0
        else:
            m_ref = numpy.linalg.norm(self.p_ref_A - self.p_ref_B)
            self.m_ref_AB = m_ref if m_ref > 0 else self.m_ref_AB
            m_ref = numpy.linalg.norm(self.p_ref_C - self.p_ref_D)
            self.m_ref_CD = m_ref if m_ref > 0 else self.m_ref_CD

        if self.en_c:
            self.k_r_clock = self.m_clock
            self.k_m_ref_AB = self.m_ref_AB
            self.k_m_ref_CD = self.m_ref_CD
            self.flag_c = True

    def _scaling(self):
        if self.scale_mode == 0:
            self.r_clock = self.k_r_clock
        elif self.scale_mode == 1:
            self.scale = self.m_ref_AB / self.k_m_ref_AB
            self.r_clock = self.k_r_clock * self.scale
        else:
            scale_AB = self.m_ref_AB / self.k_m_ref_AB
            scale_CD = self.m_ref_CD / self.k_m_ref_CD
            self.scale = max(scale_AB, scale_CD)
            self.r_clock = self.k_r_clock * self.scale

    def _normalization(self):
        self.p_norm = self.p_tran / self.r_clock
        self.p_norm[1] = -self.p_norm[1]

        self.x_r_hand = self.p_norm[0]
        self.y_r_hand = self.p_norm[1]

        self.r_hand = numpy.linalg.norm(self.p_norm)

        self.phi_r_hand = numpy.degrees(
            numpy.arctan2(self.x_r_hand, self.y_r_hand)
        ) 
        self.phi_r_hand = -self.phi_r_hand + 90
        if self.phi_r_hand < 0: self.phi_r_hand = self.phi_r_hand + 360
        self.phi_r_hand = self.phi_r_hand / 360

    def update(self, mp):
        self._setup(mp)
        self._translation()
        self._calibration()
        if self.flag_c:
            self._scaling()
            self._normalization()
