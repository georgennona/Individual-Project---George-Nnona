# https://github.com/TapWithUs/tap-python-sdk

from tapsdk import TapSDK, TapInputMode
import time
import json
import socket

LEFT_ID = "BluetoothLE#BluetoothLEc0:35:32:95:ce:2c-fd:16:05:14:57:27"
RIGHT_ID = "BluetoothLE#BluetoothLEc0:35:32:95:ce:2c-fe:87:98:e2:2f:97"

tap = TapSDK()

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_to_unity(hand, fingers):
    msg = {
        "hand": hand,
        "fingers": list(fingers),
        "timestamp": time.time()
    }
    data = json.dumps(msg).encode("utf-8")
    sock.sendto(data, (UDP_IP, UDP_PORT))

def get_hand(identifier: str) -> str:
    if identifier == LEFT_ID:
        return "left"
    if identifier == RIGHT_ID:
        return "right"
    return "unknown"

def decode_tapcode(tapcode: int):
    finger_flags = {
        "thumb":  bool(tapcode & 0b00001),
        "index":  bool(tapcode & 0b00010),
        "middle": bool(tapcode & 0b00100),
        "ring":   bool(tapcode & 0b01000),
        "pinky":  bool(tapcode & 0b10000),
    }
    return tuple(name for name, active in finger_flags.items() if active)

def on_connect(identifier, name, fw):
    hand = get_hand(identifier)
    print(f"[CONNECT] {hand} {name}")
    tap.set_input_mode(TapInputMode("controller"), identifier)

def on_disconnect(identifier):
    hand = get_hand(identifier)
    print(f"[DISCONNECT] {hand}")

def on_tap(identifier, tapcode):
    hand = get_hand(identifier)
    fingers = decode_tapcode(tapcode)
    print(f"[TAP] hand={hand} fingers={fingers}")
    if hand != "unknown":
        send_to_unity(hand, fingers)

tap.register_connection_events(on_connect)
tap.register_disconnection_events(on_disconnect)
tap.register_tap_events(on_tap)

print("Starting SDK...")
tap.run()
print("SDK started. Waiting for taps...")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped.")