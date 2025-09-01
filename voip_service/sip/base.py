from abc import ABC, abstractmethod

class SIPClient(ABC):
    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    async def register(self):
        pass

    @abstractmethod
    async def call(self, target: str):
        pass

    @abstractmethod
    async def answer(self, call_id: str):
        pass

    @abstractmethod
    async def hangup(self, call_id: str):
        pass

    @abstractmethod
    async def on_event(self, event: dict):
        pass
