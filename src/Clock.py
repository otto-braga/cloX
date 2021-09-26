import numpy

from scipy import signal
from scipy.signal import fir_filter_design

class Clock:
    def __init__(
        self,
        name='',

        i_p_clock = (0,0,-1,-1),
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
        self.p_ref_auto = numpy.zeros([8,2], dtype=int)

        self.p_clock_norm = numpy.zeros([2], dtype=int)
        self.p_hand_norm = numpy.zeros([2], dtype=int)

        self.p_tran = numpy.zeros([2], dtype=int)
        self.p_norm = numpy.zeros([2], dtype=int)
        self.m_clock = 1.0
        self.m_ref_AB = 1.0
        self.m_ref_CD = 1.0
        self.m_ref_auto = numpy.ones(
            [int(len(self.p_ref_auto) / 2)],
            dtype=float
        )
        self.k_m_ref_AB = 1.0
        self.k_m_ref_CD = 1.0
        self.k_m_ref_auto = numpy.ones(
            [int(len(self.p_ref_auto) / 2)],
            dtype=float
        )

        self.k_r_clock = 1.0

        self.scale = 1.0
        self.r_clock = 1.0

        self.r_hand = 0.0
        self.phi_r_hand = 0.0
        self.x_r_hand = 0.0
        self.y_r_hand = 0.0

        self.scale_mode = scale_mode
        self.en_c = False
        self.flag_c = False

        self.scale_history = numpy.ones([10], dtype=float)

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
        self.p_clock = self._initialize_point(self.i_p_clock, mp)
        self.p_hand = self._initialize_point(self.i_p_hand, mp)

        if self.scale_mode == 1:
            self.p_ref_A = mp.landmark[self.i_p_ref_A[0]][self.i_p_ref_A[1]]
            self.p_ref_B = mp.landmark[self.i_p_ref_B[0]][self.i_p_ref_B[1]]
            self.p_ref_C = numpy.zeros([2], dtype=int)
            self.p_ref_D = numpy.zeros([2], dtype=int)
        elif self.scale_mode == 2:
            self.p_ref_A = mp.landmark[self.i_p_ref_A[0]][self.i_p_ref_A[1]]
            self.p_ref_B = mp.landmark[self.i_p_ref_B[0]][self.i_p_ref_B[1]]
            self.p_ref_C = mp.landmark[self.i_p_ref_C[0]][self.i_p_ref_C[1]]
            self.p_ref_D = mp.landmark[self.i_p_ref_D[0]][self.i_p_ref_D[1]]
        elif self.scale_mode == 3:
            self.p_ref_auto[0] = self._midpoint(mp.landmark[0][3], mp.landmark[0][6])
            self.p_ref_auto[1] = self._midpoint(mp.landmark[0][9], mp.landmark[0][10])

            self.p_ref_auto[2] = mp.landmark[0][7]
            self.p_ref_auto[3] = mp.landmark[0][8]

            self.p_ref_auto[4] = mp.landmark[0][11]
            self.p_ref_auto[5] = mp.landmark[0][12]

            self.p_ref_auto[6] = mp.landmark[0][23]
            self.p_ref_auto[7] = mp.landmark[0][24]

        self.p_clock_norm = self.p_clock / mp.image_size
        self.p_hand_norm = self.p_hand / mp.image_size

    def _translation(self):
        self.p_tran = self.p_hand - self.p_clock

    def _calibration(self):
        self.m_clock = numpy.linalg.norm(self.p_tran)

        if self.scale_mode == 1:
            m_ref = numpy.linalg.norm(self.p_ref_A - self.p_ref_B)
            self.m_ref_AB = m_ref if m_ref > 0 else self.m_ref_AB
            self.m_ref_CD = 1.0
        elif self.scale_mode == 2:
            m_ref_AB = numpy.linalg.norm(self.p_ref_A - self.p_ref_B)
            m_ref_CD = numpy.linalg.norm(self.p_ref_C - self.p_ref_D)

            self.m_ref_AB = m_ref_AB if m_ref_AB > 0 else self.m_ref_AB
            self.m_ref_CD = m_ref_CD if m_ref_CD > 0 else self.m_ref_CD
        elif self.scale_mode == 3:
            for i, j in zip(range(0, len(self.p_ref_auto), 2), range(len(self.m_ref_auto))):
                m_ref = numpy.linalg.norm(
                    self.p_ref_auto[i] - self.p_ref_auto[i+1]
                )
                self.m_ref_auto[j] = m_ref if m_ref > 0 else self.m_ref_auto[j]

        if self.en_c:
            self.k_r_clock = self.m_clock

            self.k_m_ref_AB = self.m_ref_AB
            self.k_m_ref_CD = self.m_ref_CD

            for i in range(len(self.m_ref_auto)):
                self.k_m_ref_auto[i] = self.m_ref_auto[i]

            self.flag_c = True

    def _scaling(self):
        scale = 1.0

        if self.scale_mode == 0:
            self.r_clock = self.k_r_clock
        elif self.scale_mode == 1:
            self.scale = self.m_ref_AB / self.k_m_ref_AB
            self.r_clock = self.k_r_clock * self.scale
        elif self.scale_mode == 2:
            scale_AB = self.m_ref_AB / self.k_m_ref_AB
            scale_CD = self.m_ref_CD / self.k_m_ref_CD

            self.scale = max(scale_AB, scale_CD)
            self.r_clock = self.k_r_clock * self.scale
        elif self.scale_mode == 3:
            scale_auto = numpy.zeros(self.m_ref_auto.shape, dtype=float)
            for i in range(len(self.m_ref_auto)):
                scale_auto[i] = self.m_ref_auto[i] / self.k_m_ref_auto[i]

            scale = max(scale_auto)
            self.r_clock = self.k_r_clock * self.scale
        
        self.scale_history[:-1] = self.scale_history[1:]
        self.scale_history[-1] = scale
        self.scale = numpy.mean(self.scale_history)

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
