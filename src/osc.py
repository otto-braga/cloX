from pythonosc import udp_client

def client_setup(ip, port):
    return udp_client.SimpleUDPClient(ip, port)

def make_messages(clocks):
    messages = []
    for clock in clocks:
        address = '/cloX/' + clock.name + '/'
        sep = ' '
        messages.append(address + 'r_hand' + sep + str(clock.r_hand))
        messages.append(address + 'phi_r_hand' + sep + str(clock.phi_r_hand))
        messages.append(address + 'x_r_hand' + sep + str(clock.x_r_hand))
        messages.append(address + 'y_r_hand' + sep + str(clock.y_r_hand))
        messages.append(address + 'scale' + sep + str(clock.scale))
        messages.append(address + 'speed' + sep + str(clock.speed_magnitude))
        messages.append(address + 'p_clock_norm_x' + sep + str(clock.p_clock_norm[0]))
        messages.append(address + 'p_clock_norm_y' + sep + str(clock.p_clock_norm[1]))
        messages.append(address + 'p_hand_norm_x' + sep + str(clock.p_hand_norm[0]))
        messages.append(address + 'p_hand_norm_y' + sep + str(clock.p_hand_norm[1]))
    return messages

def send(client, messages):
    for message in messages:
        message_split = message.split(sep=' ', maxsplit=1)
        address = message_split[0]
        argument = float(message_split[1])
        client.send_message(address, argument)
