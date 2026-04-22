import pigpio
from rplidar import RPLidar


class RPLidarSensor:
    def __init__(self, pwm_pin, uart_port):
        self._baudrate = 115200
        self._pwm_freq = 25000
        self._pwm_duty = 660
        self._pwm_duty_max = 1023

        self._pwm_pin = pwm_pin
        self._port = uart_port
        self._lidar = RPLidar(self._port, self._baudrate)
        self._gpio = pigpio.pi()
        self._scan_iter = None

        self.init_motor()

    def init_motor(self):
        self._gpio.set_mode(self._pwm_pin, pigpio.OUTPUT)
        self._gpio.hardware_PWM(self._pwm_pin, self._pwm_freq, int(
            self._pwm_duty / self._pwm_duty_max * 1_000_000))
    
    def start_motor(self):
        self._gpio.hardware_PWM(self._pwm_pin, self._pwm_freq, int(
            self._pwm_duty / self._pwm_duty_max * 1_000_000))
        
    def stop_motor(self):
        self._gpio.hardware_PWM(self._pwm_pin, self._pwm_freq, 0)
        
    def read_scan_frame(self):
        if self._scan_iter is None:
            self._scan_iter = self._lidar.iter_scans(scan_type='express')
        return next(self._scan_iter)

    def read_measurement(self):
        return next(self._lidar.iter_measures())
    
    def stop(self):
        self._lidar.stop()

    def disconnect(self):
        self._lidar.disconnect()

    def flush(self):
        self._lidar._serial.reset_input_buffer()

