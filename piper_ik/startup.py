"""Local listener check; no simulator or hardware initialization."""
import socket


def check_port_available(port):
    # Match Uvicorn's SO_REUSEADDR behavior: closed TCP connections in
    # TIME_WAIT must not look like an active console. No SO_REUSEPORT.
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(('127.0.0.1', port))
