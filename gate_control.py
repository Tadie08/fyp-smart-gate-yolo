import serial
import time
import glob

def get_arduino_port():
    """Auto detect Arduino port"""
    ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
    if ports:
        return ports[0]
    raise Exception("Arduino not found! Check USB connection.")

# Connect to Arduino
try:
    arduino_port = get_arduino_port()
    arduino = serial.Serial(arduino_port, 9600, timeout=1)
    time.sleep(2)  # Wait for Arduino to reset
    print(f"✅ Arduino connected on {arduino_port}")
except Exception as e:
    print(f"❌ Arduino connection failed: {e}")
    arduino = None

def read_arduino():
    """Read any pending messages from Arduino"""
    messages = []
    while arduino and arduino.in_waiting:
        line = arduino.readline().decode('utf-8').strip()
        if line:
            messages.append(line)
    return messages

def get_distance():
    """Get latest distance reading from Arduino"""
    messages = read_arduino()
    for msg in messages:
        if msg.startswith("DIST:"):
            try:
                return int(msg.split(":")[1])
            except:
                pass
    return None

def car_detected():
    """Check if Arduino detected a car"""
    messages = read_arduino()
    return "CAR_DETECTED" in messages

def send_command(command):
    """Send command to Arduino"""
    if arduino:
        arduino.write(f"{command}\n".encode())
        time.sleep(0.1)

def open_gate():
    print("🔓 Opening gate...")
    send_command("OPEN")
    # Wait for gate to finish
    time.sleep(8)  # 1.8 open + 5 hold + 1.8 close
    print("🔒 Gate closed!")

def slow_open_gate():
    print("⚠️ Slow opening - medium risk")
    send_command("OPEN")
    time.sleep(8)
    print("🔒 Gate closed!")

def execute_decision(decision):
    if decision == "AUTO_OPEN":
        open_gate()
    elif decision == "SLOW_OPEN":
        slow_open_gate()
    elif decision == "VISITOR_OPEN":
        open_gate()
    elif decision in ["DENY", "LOG_ONLY"]:
        send_command("DENY")
        print("🚫 Access DENIED — gate remains closed")

def cleanup():
    if arduino:
        arduino.close()
        print("Arduino disconnected")
