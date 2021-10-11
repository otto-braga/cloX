from pythonosc import udp_client

def client_setup(ip, port):
    return udp_client.SimpleUDPClient(ip, port)

def make_messages(clocks):
    messages = {}
    for clock in clocks:
        address = '/cloX/' + clock.name + '/'
        
        messages[address + 'r_hand'] = clock.r_hand
        messages[address + 'phi_r_hand'] = clock.phi_r_hand
        messages[address + 'x_r_hand'] = clock.x_r_hand
        messages[address + 'y_r_hand'] = clock.y_r_hand
        messages[address + 'scale'] = clock.scale
        messages[address + 'speed'] = clock.speed_magnitude
        messages[address + 'speed_average'] = clock.speed_magnitude_average
        messages[address + 'direction_x'] = clock.direction[0]
        messages[address + 'direction_y'] = clock.direction[1]
        messages[address + 'speed_relative'] = clock.speed_magnitude_relative
        messages[address + 'speed_average_relative'] = clock.speed_magnitude_average_relative
        messages[address + 'direction_x_relative'] = clock.direction_relative[0]
        messages[address + 'direction_y_relative'] = clock.direction_relative[1]
        messages[address + 'p_clock_norm_x'] = clock.p_clock_norm[0]
        messages[address + 'p_clock_norm_y'] = clock.p_clock_norm[1]
        messages[address + 'p_hand_norm_x'] = clock.p_hand_norm[0]
        messages[address + 'p_hand_norm_y'] = clock.p_hand_norm[1]

        if clock.drawn_gesture_catcher:
            for drawn_gesture_catcher in clock.drawn_gesture_catcher:
                address_g = (
                    address
                    + 'drawn_gesture/'
                    + drawn_gesture_catcher.name
                    + '/'
                )

                messages[address_g + 'is_catching'] = (
                    int(drawn_gesture_catcher.is_catching)
                )

                messages[address_g + 'threshold_catch'] = (
                    drawn_gesture_catcher.threshold_catch
                )

                messages[address_g + 'threshold_release'] = (
                    drawn_gesture_catcher.threshold_release
                )

                messages[address_g + 'speed_history_size'] = (
                    drawn_gesture_catcher.speed_history_size
                )

                if (
                    not drawn_gesture_catcher.is_catching
                    and len(drawn_gesture_catcher.gesture_points)
                ):
                    length = len(drawn_gesture_catcher.gesture_points)

                    messages[address_g + 'gesture_points_length'] = length

                    for i in range(length):
                        messages[address_g + 'gesture_point_x_' + str(i)] = (
                            int(drawn_gesture_catcher.gesture_points[i,0])
                        )
                        messages[address_g + 'gesture_point_y_' + str(i)] = (
                            int(drawn_gesture_catcher.gesture_points[i,1])
                        )

    return messages

def send(client, messages):
    for address, value in messages.items():
        print(address, value)
        client.send_message(address, value)
