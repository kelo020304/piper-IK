"""Only ephemeral localhost sockets; no service or robot startup."""
import errno
import socket
import pytest
from piper_ik.startup import check_port_available


def test_active_listener_is_rejected():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',0));listener.listen()
        with pytest.raises(OSError) as exc:check_port_available(listener.getsockname()[1])
        assert exc.value.errno==errno.EADDRINUSE


def test_closed_connection_time_wait_does_not_block_restart():
    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',0));listener.listen()
        address=listener.getsockname();client.connect(address)
        connection,_=listener.accept()
        with connection:connection.shutdown(socket.SHUT_WR)
        assert client.recv(1)==b''
    # The server side actively closed: old plain bind falsely rejects this.
    with socket.socket() as old_check:
        with pytest.raises(OSError) as exc:old_check.bind(address)
        assert exc.value.errno==errno.EADDRINUSE
    check_port_available(address[1])
