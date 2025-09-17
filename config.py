
# config.py
# Configuration settings for the VoIP server
import os
from dotenv import load_dotenv

load_dotenv()

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8765))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Twilio Media Streams inactivity timeout (seconds)
# If no events/media are received for this duration, the server will close the WS.
# Increase if calls are being closed too quickly.
TWILIO_INACTIVITY_TIMEOUT_SECS = float(os.getenv("TWILIO_INACTIVITY_TIMEOUT_SECS", 60))

# SIP credentials from .env
SIP_USERNAME = os.getenv("SIP_USERNAME")
SIP_PASSWORD = os.getenv("SIP_PASSWORD")
SIP_SERVER = os.getenv("SIP_SERVER")
