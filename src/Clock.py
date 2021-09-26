import numpy

class Clock:
    def __init__(
        self,
        name='',

        p_clock_mode = 0,
        i_p_clock = (0,0,-1,-1),

        p_hand_mode = 0,
        i_p_hand = (0,0,-1,-1),

        scale_mode = 0,
        i_p_ref_A = (0,0),
        i_p_ref_B = (0,0),
        i_p_ref_C = (0,0),
        i_p_ref_D = (0,0)
    ):
        self.name = name

        self.i_p_clock = numpy.array(i_p_clock)
        self.i_p_hand = numpy.array(i_p_hand)

        self.i_p_ref_A = numpy.array(i_p_ref_A)
        self.i_p_ref_B = numpy.array(i_p_ref_B)
        self.i_p_ref_C = numpy.array(i_p_ref_C)
        self.i_p_ref_D = numpy.array(i_p_ref_D)

        self.p_clock = numpy.zeros([2], dtype=int)
        self.p_hand = numpy.zeros([2], dtype=int)
        self.p_ref_A = numpy.zeros([2], dtype=int)
        self.p_ref_B = numpy.zeros([2], dtype=int)
        self.p_ref_C = numpy.zeros([2], dtype=int)
        self.p_ref_D = numpy.zeros([2], dtype=int)
        
        self.p_ref_E = numpy.zeros([2], dtype=int)
        self.p_ref_F = numpy.zeros([2], dtype=int)
        self.p_ref_G = numpy.zeros([2], dtype=int)
        self.p_ref_H = numpy.zeros([2], dtype=int)

        self.p_clock_norm = numpy.zeros([2], dtype=int)
        self.p_hand_norm = numpy.zeros([2], dtype=int)

        self.p_tran = numpy.zeros([2], dtype=int)
        self.p_norm = numpy.zeros([2], dtype=int)
        self.m_clock = 1.0
        self.m_ref_AB = 1.0
        self.m_ref_CD = 1.0
        self.m_ref_EF = 1.0
        self.m_ref_GH = 1.0
        self.k_m_ref_AB = 1.0
        self.k_m_ref_CD = 1.0
        self.k_m_ref_EF = 1.0
        self.k_m_ref_GH = 1.0
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

    # Auxiliary methods.
    # ------------------

    def _midpoint(self, point_A, point_B):
        return numpy.array(
            [
                min(point_A[0], point_B[0]) + (abs(point_A[0] - point_B[0]) / 2),
                min(point_A[1], point_B[1]) + (abs(point_A[1] - point_B[1]) / 2),
            ],
            dtype=int
        )

    def _initialize_point(self, i_point, mp):
        i_point_A = i_point[:2]
        i_point_B = i_point[2:4]

        if numpy.any(i_point_A) < 0:
            i_point_A = numpy.abs(i_point_A)
            return numpy.array(
                mp.landmark[i_point_A[0]][i_point_A[1]] * mp.image_size,
                dtype=int
            )
        elif numpy.any(i_point_B) < 0:
            return mp.landmark[i_point_A[0]][i_point_A[1]]
        else:
            return self._midpoint(
                mp.landmark[i_point_A[0]][i_point_A[1]],
                mp.landmark[i_point_B[0]][i_point_B[1]]
            )

    # Main methods.
    # -------------

    def _setup(self, mp):
        p_clock = self._initialize_point(self.i_p_clock, mp)
        p_hand = self._initialize_point(self.i_p_hand, mp)

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

            self.p_ref_E = self._midpoint(mp.landmark[0][2], mp.landmark[0][5])
            self.p_ref_F = self._midpoint(mp.landmark[0][9], mp.landmark[0][10])
            self.p_ref_G = mp.landmark[0][8]
            self.p_ref_H = mp.landmark[0][9]

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
            m_ref_AB = numpy.linalg.norm(self.p_ref_A - self.p_ref_B)
            m_ref_CD = numpy.linalg.norm(self.p_ref_C - self.p_ref_D)
            m_ref_EF = numpy.linalg.norm(self.p_ref_E - self.p_ref_F)
            m_ref_GH = numpy.linalg.norm(self.p_ref_G - self.p_ref_H)

            self.m_ref_AB = m_ref_AB if m_ref_AB > 0 else self.m_ref_AB
            self.m_ref_CD = m_ref_CD if m_ref_CD > 0 else self.m_ref_CD
            self.m_ref_EF = m_ref_EF if m_ref_EF > 0 else self.m_ref_EF
            self.m_ref_GH = m_ref_GH if m_ref_GH > 0 else self.m_ref_GH

        if self.en_c:
            self.k_r_clock = self.m_clock
            self.k_m_ref_AB = self.m_ref_AB
            self.k_m_ref_CD = self.m_ref_CD
            self.k_m_ref_EF = self.m_ref_EF
            self.k_m_ref_GH = self.m_ref_GH
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
            scale_EF = self.m_ref_EF / self.k_m_ref_EF
            scale_GH = self.m_ref_GH / self.k_m_ref_GH

            self.scale = max(scale_AB, scale_CD, scale_EF, scale_GH)
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
