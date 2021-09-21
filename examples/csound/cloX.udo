opcode cloX_OSC_filter, k, kiiiii
    k_in, 
    i_offset, 
    i_scale, 
    i_limit_low, 
    i_limit_high, 
    i_LP_cutoff_freq  xin

    k_out = (k_in - i_offset) * i_scale

    a_out interp k_out
    a_out butlp a_out, i_LP_cutoff_freq

    k_out downsamp a_out

    if k_out < i_limit_low then
        k_out = i_limit_low
    elseif k_out > i_limit_high then
        k_out = i_limit_high
    endif

    xout k_out
endop

opcode cloX_OSC2MIDI, kk, k
    k_in xin

    k_out_value = int(k_in * 127)
    k_out_trigger changed2 k_out_value

    xout k_out_value, k_out_trigger
endop

opcode cloX_MIDI_CC, k, Siiiiiiiii
    S_OSC_address,
    i_MIDI_CC_number,
    i_filter_offset, 
    i_filter_scale, 
    i_filter_limit_low, 
    i_filter_limit_high, 
    i_filter_LP_cutoff_freq,
    i_OSC_handle,
    i_MIDI_channel,
    i_MIDI_status xin

    k_OSC_argument init 0
    i_MIDI_CC_number = 8 + i_MIDI_CC_number

    k_OSC_action OSClisten i_OSC_handle,
        S_OSC_address, "f", k_OSC_argument
    
    k_OSC_value cloX_OSC_filter k_OSC_argument,
        i_filter_offset, 
        i_filter_scale, 
        i_filter_limit_low, 
        i_filter_limit_high, 
        i_filter_LP_cutoff_freq
    
    k_MIDI_CC_value, k_trigger cloX_OSC2MIDI k_OSC_value

    k_trigger changed2 k_MIDI_CC_value

    if k_trigger == 1 then
        midiout i_MIDI_status, i_MIDI_channel, 
            i_MIDI_CC_number, k_MIDI_CC_value
    endif

    xout k_MIDI_CC_value
endop

opcode cloX_OSC, k, Siiiiii
    S_OSC_address,
    i_filter_offset, 
    i_filter_scale, 
    i_filter_limit_low, 
    i_filter_limit_high, 
    i_filter_LP_cutoff_freq,
    i_OSC_handle  xin

    k_OSC_argument init 0

    k_OSC_action OSClisten i_OSC_handle,
        S_OSC_address, "f", k_OSC_argument
    
    k_OSC_value cloX_OSC_filter k_OSC_argument,
        i_filter_offset, 
        i_filter_scale, 
        i_filter_limit_low, 
        i_filter_limit_high, 
        i_filter_LP_cutoff_freq

    xout k_OSC_value
endop
