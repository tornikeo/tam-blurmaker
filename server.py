from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

app = FastAPI()

@app.get("/")
async def read_main():
    return "Tornikeo was here <3"

@app.websocket("/ws")
async def websocket(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"msg": "Hello WebSocket"})
    await websocket.close()

