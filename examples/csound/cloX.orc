#define OSC_R_PORT # 4747 #

#include "cloX.udo"

gk_clock_A_r_hand init 0

instr cloX
    i_OSC_handle OSCinit $OSC_R_PORT

    i_offset init .2
    i_scale init 1
    i_limit_L init 0
    i_limit_H init 1
    i_LP_cutoff init 5000

    gk_clock_A_r_hand cloX_OSC "/cloX/clock_A/r_hand",
        i_offset, i_scale, i_limit_L, i_limit_H, i_LP_cutoff,
        i_OSC_handle
    
endin
