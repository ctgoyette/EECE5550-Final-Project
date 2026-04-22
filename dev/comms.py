import serial
import time

ser = serial.Serial(
    port='/dev/ttyAMA3',
    baudrate=115200,
    timeout=1
)

num = 1
try:
    while True:
        ser.write(f"{num}".encode())
        print("Data sent!")
        time.sleep(2)
        num += 1
except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()