from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory
from time import sleep
import warnings
warnings.filterwarnings("ignore")

factory = LGPIOFactory()
servo = Servo(14,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
    pin_factory=factory)

def boom_open():
    print("🔓 Boom gate opening...")
    for val in [i/200 for i in range(-200, 201, 1)]:
        servo.value = val
        sleep(0.080)
    print("🔓 Gate fully open!")

def boom_close():
    print("🔒 Boom gate closing...")
    for val in [i/200 for i in range(200, -201, -1)]:
        servo.value = val
        sleep(0.012)
    print("🔒 Gate fully closed!")

print("Get ready - 5 seconds...")
for i in range(5, 0, -1):
    print(f"  {i}...")
    sleep(1)

boom_open()
sleep(3)
boom_close()

servo.detach()
print("Done!")
