import websocket
from contextlib import closing
from websocket import create_connection

ws_location = "ws://127.0.0.1:8000/ws"
old_location = "wss://testnet-explorer.binance.org/ws/block"

if __name__ == "__main__":
    with closing(create_connection(ws_location)) as conn:
        while True:
            print(conn.recv())

