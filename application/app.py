import json
from pathlib import Path
import gc

from flask import Flask
from flask import render_template
from flask_socketio import SocketIO
from simview.utils import find_free_port

HERE = Path(__file__).parent

TEMPLATES = HERE / "templates"
STATIC = HERE / "static"


class SimViewServer:
    def __init__(self, sim_path: str | Path):
        self.sim_path = sim_path
        self.app = Flask(__name__, template_folder=str(TEMPLATES), static_folder=str(STATIC))
        self.socketio = SocketIO(self.app, json=json, cors_allowed_origins="*", async_mode="eventlet")
        self.setup_routes()
        self.setup_socket_handlers()

    def setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template("index.html")

    def setup_socket_handlers(self):
        @self.socketio.on("connect")
        def handle_connect():
            print("Client connected")

        @self.socketio.on("disconnect")
        def handle_disconnect():
            print("Client disconnected")

        @self.socketio.on("get_model")
        def handle_get_model():
            try:
                with open(self.sim_path, "r") as f:
                    js = json.load(f)
                    self.socketio.emit("model", js["model"])
                    del js
                    gc.collect()
            except Exception as e:
                print(f"Error loading model: {e}")
                self.socketio.emit("error", {"message": "Error loading model"})

        @self.socketio.on("get_states")
        def handle_get_states():
            try:
                with open(self.sim_path, "r") as f:
                    js = json.load(f)
                    self.socketio.emit("states", js["states"])
                    del js
                    gc.collect()
            except Exception as e:
                print(f"Error loading states: {e}")
                self.socketio.emit("error", {"message": "Error loading states"})

    def run(self, debug: bool = False, host: str = "0.0.0.0", port: int = 5420):
        self.socketio.run(self.app, debug=debug, host=host, port=port, log_output=True)

    @staticmethod
    def start(sim_path: str | Path, host: str = "0.0.0.0", preferred_port: int = 5420):
        server = SimViewServer(sim_path=sim_path)
        port = find_free_port(host, preferred_port)
        if port != preferred_port:
            print(f"Preferred port {preferred_port} is not available. Using port {port} instead.")
        server.run(host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the SimView server.")
    parser.add_argument("--sim_path", type=str, required=True, help="Path to the simulation JSON file.")
    args = parser.parse_args()
    SimViewServer.start(args.sim_path)
