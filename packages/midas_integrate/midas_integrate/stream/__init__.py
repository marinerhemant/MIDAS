"""Socket-based streaming integration — live-feed server + client.

Typical use:

    >>> import midas_integrate as mi
    >>> srv = mi.stream.Server(config, map_artifacts)
    >>> srv.start()
    >>> with mi.stream.Client("127.0.0.1:60439") as c:
    ...     cake = c.send_frame(frame)
    >>> srv.shutdown()

v0.1.0 is single-client, CPU-backed, loopback-default. Multi-client
accept + GPU backend land in v0.2 alongside the ``midas-integrate-gpu``
wheel.
"""

from .client import Client, StreamError
from .protocol import (
    ControlMessage,
    FrameMessage,
    OP_FRAME,
    OP_PING,
    OP_PONG,
    OP_SHUTDOWN,
    ProfileMessage,
    ProtocolError,
)
from .server import Server

__all__ = [
    "Client",
    "ControlMessage",
    "FrameMessage",
    "OP_FRAME",
    "OP_PING",
    "OP_PONG",
    "OP_SHUTDOWN",
    "ProfileMessage",
    "ProtocolError",
    "Server",
    "StreamError",
]
