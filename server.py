from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
import time

app = FastAPI()

@app.get("/")
async def read_main():
    return "Tornikeo was here <3"

@app.websocket("/ws")
async def websocket(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json({"msg": "Hello WebSocket", "i": i})
    await websocket.close()

