from fastapi import FastAPI
from voip_service.sip_handler import SIPHandler
from voip_service.audio_streamer import AudioStreamer
from voip_service.config import Config
from voip_service.sip.baresip_driver import BaresipDriver
import asyncio

app = FastAPI()

config = Config()
sip_handler = SIPHandler(config)
audio_streamer = AudioStreamer(config)
baresip_driver = BaresipDriver(config.baresip_host, config.baresip_port)

@app.on_event("startup")
async def startup_event():
    await baresip_driver.connect()
    app.state.event_listener = asyncio.create_task(baresip_driver.listen_for_events())

@app.on_event("shutdown")
async def shutdown_event():
    app.state.event_listener.cancel()
    await baresip_driver.stop()

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
