import numpy

class Clock:
    def __init__(
        self,
        name='',
        index_p_clock = (0,0),
        index_hand_tip_point = (0,0),
        index_reference_point_A = (0,0),
        index_reference_point_B = (0,0),
        en_s = True
    ):
        self.name = name

        self.i_p_clock = numpy.array(index_p_clock)
        self.i_p_hand = numpy.array(index_hand_tip_point)
        self.i_p_ref_A = numpy.array(index_reference_point_A)
        self.i_p_ref_B = numpy.array(index_reference_point_B)

        self.p_clock = numpy.zeros([2], dtype=int)
        self.p_hand = numpy.zeros([2], dtype=int)
        self.p_ref_A = numpy.zeros([2], dtype=int)
        self.p_ref_B = numpy.zeros([2], dtype=int)

        self.p_trans = numpy.zeros([2], dtype=int)
        self.p_norm = numpy.zeros([2], dtype=int)
        self.m_clock = 1.0
        self.m_ref = 1.0
        self.k_m_ref = 1.0
        self.k_r_clock = 1.0

        self.scale = 1.0
        self.r_clock = 1.0

        self.r_hand = 0.0
        self.phi_r_hand = 0.0
        self.x_r_hand = 0.0
        self.y_r_hand = 0.0

        self.en_c = False
        self.en_s = en_s
        self.flag_c = False

    def _setup(self, mp):
        if numpy.any(self.i_p_clock < 0):
            p_clock = numpy.array(
                numpy.abs(self.i_p_clock) * mp.image_size,
                dtype=int
            )
        else:
            p_clock = mp.landmark[self.i_p_clock[0]][self.i_p_clock[1]]

        if numpy.any(self.i_p_hand < 0):
            p_hand = numpy.array(
                numpy.abs(self.i_p_hand) * mp.image_size,
                dtype=int
            )
        else:
            p_hand = mp.landmark[self.i_p_hand[0]][self.i_p_hand[1]]

        p_ref_A = mp.landmark[self.i_p_ref_A[0]][self.i_p_ref_A[1]]
        p_ref_B = mp.landmark[self.i_p_ref_B[0]][self.i_p_ref_B[1]]

        if (
            numpy.all(p_clock > 0) and
            numpy.all(p_hand > 0) and
            numpy.all(p_ref_A > 0) and
            numpy.all(p_ref_B > 0)
        ):
            self.p_clock = p_clock
            self.p_hand = p_hand
            self.p_ref_A = p_ref_A
            self.p_ref_B = p_ref_B

    def _translation(self):
        self.p_trans = self.p_hand - self.p_clock

    def _calibration(self):
        self.m_clock = numpy.linalg.norm(self.p_trans)
        self.m_ref = numpy.linalg.norm(self.p_ref_A - self.p_ref_B)

        if self.en_c:
            self.k_r_clock = self.m_clock
            self.k_m_ref = self.m_ref
            self.flag_c = True

    def _scaling(self):
        if self.en_s:
            self.scale = self.m_ref / self.k_m_ref
            self.r_clock = self.k_r_clock * self.scale
        else:
            self.r_clock = self.k_r_clock

    def _normalization(self):
        self.p_norm = self.p_trans / self.r_clock
        self.p_norm[1] = -self.p_norm[1]

        self.x_r_hand = self.p_norm[0]
        self.y_r_hand = self.p_norm[1]
        self.r_hand = numpy.linalg.norm(self.p_norm)
        self.phi_r_hand = numpy.degrees(
            numpy.arctan2(self.x_r_hand, self.y_r_hand)
        )

    def update(self, mp):
        self._setup(mp)
        self._translation()
        self._calibration()
        if self.flag_c:
            self._scaling()
            self._normalization()
