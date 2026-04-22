import serial
import time

class SerialDriver:
    def __init__(self, port, baudrate=115200, timeout=1):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout
        )

    def send(self, data):
        """Send string or bytes."""
        if isinstance(data, str):
            data = data.encode()
        self.ser.write(data)

    def receive(self, num_bytes=64):
        """Read up to num_bytes, returns decoded string."""
        return self.ser.read(num_bytes).decode(errors='ignore')

    def receive_line(self):
        """Read until newline, returns decoded string."""
        return self.ser.readline().decode(errors='ignore').strip()

    def available(self):
        """Bytes waiting in the buffer."""
        return self.ser.in_waiting

    def flush(self):
        self.ser.flush()

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()