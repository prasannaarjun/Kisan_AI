import asyncio
import logging

class BaresipDriver:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.logger = logging.getLogger("BaresipDriver")
        self.reader = None
        self.writer = None

    async def connect(self):
        try:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            self.logger.info("Connected to Baresip ctrl_tcp.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Baresip: {e}")

    async def send_command(self, command: str):
        if self.writer is None:
            raise ConnectionError("Writer is not initialized. Call connect() first.")
        self.writer.write((command + "\n").encode())
        await self.writer.drain()
        self.logger.info(f"Sent command: {command}")

    async def read_response(self):
        if self.reader is None:
            raise ConnectionError("Reader is not initialized. Call connect() first.")
        response = await self.reader.readline()
        self.logger.info(f"Received response: {response.decode().strip()}")
        return response.decode().strip()

    async def register(self):
        await self.send_command("/reg")

    async def call(self, target: str):
        await self.send_command(f"/dial {target}")

    async def answer(self, call_id: str):
        await self.send_command(f"/accept {call_id}")
        self.logger.info(f"Answered call with ID: {call_id}")

    async def hangup(self, call_id: str):
        await self.send_command(f"/hangup {call_id}")

    async def listen_for_events(self):
        while True:
            response = await self.read_response()
            if "incoming call" in response:
                call_id = self.extract_call_id(response)
                self.logger.info(f"Incoming call detected: {call_id}")
                await self.answer(call_id)

    def extract_call_id(self, response: str) -> str:
        # Extract the call ID from the response string
        # Example: "incoming call from sip:1234@sip_server;call-id=abcd1234"
        parts = response.split(";")
        for part in parts:
            if part.startswith("call-id="):
                return part.split("=")[1]
        return ""

    async def stop(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            self.logger.info("Connection to Baresip closed.")
