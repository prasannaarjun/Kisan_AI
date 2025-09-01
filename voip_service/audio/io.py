import sounddevice as sd
import logging

class AudioIO:
    def __init__(self):
        self.logger = logging.getLogger("AudioIO")

    def list_devices(self):
        devices = sd.query_devices()
        self.logger.info("Available audio devices:")
        for i, device in enumerate(devices):
            device = dict(device)  # Explicitly cast to dictionary
            device_name = device.get('name', 'Unknown Device')  # Safely access 'name'
            self.logger.info(f"{i}: {device_name}")
        return devices

    def start_stream(self, input_device: int, output_device: int):
        self.logger.info(f"Starting audio stream with input device {input_device} and output device {output_device}.")
        # Implement audio streaming logic here

    def stop_stream(self):
        self.logger.info("Stopping audio stream.")
        # Implement logic to stop audio streaming
