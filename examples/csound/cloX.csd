<CsoundSynthesizer>

<CsOptions>
</CsOptions>

<CsInstruments>

sr = 44100
kr = 32
0dbfs = 1

#include "cloX.orc"

instr example
    k_example = gk_clock_A_r_hand
    printk2 k_example
endin

</CsInstruments>

<CsScore>

i "cloX" 0 3600
i "example" 0 3600

</CsScore>

</CsoundSynthesizer>