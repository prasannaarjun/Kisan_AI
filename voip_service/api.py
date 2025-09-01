from fastapi import FastAPI
from voip_service.sip_handler import SIPHandler
from voip_service.audio_streamer import AudioStreamer
from voip_service.config import Config

app = FastAPI()

config = Config()
sip_handler = SIPHandler(config)
audio_streamer = AudioStreamer(config)

@app.post("/start_call")
def start_call(target: str):
    sip_handler.initiate_call(target)
    return {"status": "Call initiated", "target": target}

@app.post("/end_call")
def end_call(call_id: str):
    sip_handler.terminate_call(call_id)
    return {"status": "Call terminated", "call_id": call_id}

@app.post("/send_audio")
def send_audio(audio_chunk: bytes):
    audio_streamer.send_audio(audio_chunk)
    return {"status": "Audio sent"}

@app.post("/receive_audio")
def receive_audio():
    audio_chunk = audio_streamer.receive_audio()
    return {"status": "Audio received", "audio_chunk": audio_chunk}
