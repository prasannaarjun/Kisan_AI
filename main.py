# main.py

# FastAPI app supporting Twilio VoIP calls
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Request
from fastapi.responses import PlainTextResponse
import json
import base64
from typing import Optional
import logging
from config import LOG_LEVEL
from ai.ai_agent import AIAgent

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

def setup_logging():
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

@app.on_event("startup")
def _on_startup():
    setup_logging()

ai_agent = AIAgent()

# --- Twilio Media Streams integration ---

def mulaw_byte_to_linear(sample_byte: int) -> int:
    """
    Decode one 8-bit mu-law sample to signed 16-bit PCM.
    Returns an int in [-32768, 32767].
    """
    MU_LAW_BIAS = 0x84
    sample = ~sample_byte & 0xFF
    segment = (sample & 0x70) >> 4
    mantissa = sample & 0x0F
    magnitude = ((mantissa << 3) + MU_LAW_BIAS) << segment
    pcm = -(magnitude) if (sample & 0x80) else (magnitude)
    # Clamp to int16
    if pcm > 32767:
        pcm = 32767
    if pcm < -32768:
        pcm = -32768
    return pcm

def decode_mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """
    Convert a bytes object of 8kHz mono 8-bit mu-law to 16-bit PCM (little-endian).
    """
    out = bytearray()
    for b in mulaw_bytes:
        s16 = mulaw_byte_to_linear(b)
        out += int.to_bytes(s16 & 0xFFFF if s16 >= 0 else (s16 + 65536), 2, 'little', signed=False)
    return bytes(out)

def _build_twiml(stream_url: Optional[str]):
    """
    Compose TwiML to auto-answer and connect a Media Stream to our WebSocket.
    """
    target_url = stream_url or "wss://YOUR_PUBLIC_DOMAIN/twilio/stream"
    twiml = (
        f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{target_url}"/>
  </Connect>
</Response>"""
    )
    return Response(content=twiml, media_type="application/xml")

@app.get("/twiml/voice", response_class=PlainTextResponse)
def twiml_voice_get(request: Request, stream_url: Optional[str] = None):
    """Twilio may validate with GET; support both GET and POST."""
    return _build_twiml(stream_url)

@app.post("/twiml/voice", response_class=PlainTextResponse)
def twiml_voice_post(request: Request, stream_url: Optional[str] = None):
    return _build_twiml(stream_url)

@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """
    Receive Twilio Media Streams JSON messages and forward decoded PCM to AI agent.
    """
    await websocket.accept()
    logging.info("Twilio WebSocket connected")
    call_sid = None
    
    try:
        while True:
            message_text = await websocket.receive_text()
            event = json.loads(message_text)
            event_type = event.get("event")
            
            if event_type == "start":
                start = event.get("start", {})
                call_sid = start.get("callSid")
                logging.info(f"Twilio stream started: {call_sid}")
                
            elif event_type == "media":
                media = event.get("media", {})
                payload_b64 = media.get("payload")
                if not payload_b64:
                    continue
                    
                # Decode mu-law audio to PCM
                mulaw_bytes = base64.b64decode(payload_b64)
                pcm16_le = decode_mulaw_to_pcm16(mulaw_bytes)
                
                # Forward audio to AI agent
                ai_agent.process_audio_frame(pcm16_le)
                logging.debug(f"Processed {len(pcm16_le)} bytes of audio")
                
            elif event_type == "stop":
                logging.info(f"Twilio stream stopped: {call_sid}")
                break
                
    except WebSocketDisconnect:
        logging.info("Twilio WebSocket disconnected")
    except Exception as e:
        logging.error(f"Twilio stream error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass