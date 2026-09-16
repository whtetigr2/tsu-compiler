"""Port selection must not hand out a port another process is using.

Regression test for a real failure. The first version of `pick_free_port` set
SO_REUSEADDR on its probe socket. On Linux that option only bypasses TIME_WAIT,
so the probe behaved. On Windows SO_REUSEADDR permits binding a port another
PROCESS is actively listening on, so the probe called an occupied port free and
uvicorn then failed for real:

    [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8092)

Measured directly against an occupied port, with the other socket held by a
separate process:

    with SO_REUSEADDR    : bind succeeded  (reports free -- wrong)
    without SO_REUSEADDR : bind failed 10048  (correct)

The window never opened and the only symptom was a ninety-second wait.

NOTE ON HOW THIS IS TESTED. The first version of this test held the port on a
socket in the SAME process, and passed against the broken code -- the hijack
only shows across processes. A test that cannot fail proves nothing, so the
port is held by a subprocess here. Do not simplify that away.
"""
import contextlib
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from desktop.main import HOST, pick_free_port  # noqa: E402

# The holder sets SO_REUSEADDR deliberately. Windows permits the hijack only
# when the EXISTING socket also set that option -- and every Python server the
# user is likely to have running does: socketserver.TCPServer sets
# `allow_reuse_address`, which is what http.server and uvicorn inherit. A
# holder without it does not reproduce the defect, and an earlier version of
# this test passed against the broken code for exactly that reason.
_HOLDER = (
    "import socket, time\n"
    "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "s.bind(('127.0.0.1', 0))\n"
    "s.listen(1)\n"
    "print(s.getsockname()[1], flush=True)\n"
    "time.sleep(120)\n"
)


@contextlib.contextmanager
def port_held_by_another_process():
    """Yield a port number that a separate live process is listening on."""
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER],
                            stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline().strip()
        if not line:
            pytest.skip("could not start the port-holding subprocess")
        yield int(line)
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_picks_the_preferred_port_when_it_is_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        free_port = probe.getsockname()[1]
    # probe is closed, so free_port is available again
    assert pick_free_port(HOST, preferred=free_port, span=5) == free_port


def test_skips_a_port_another_process_is_listening_on():
    """The exact defect. Fails against a probe that sets SO_REUSEADDR."""
    with port_held_by_another_process() as occupied:
        chosen = pick_free_port(HOST, preferred=occupied, span=20)
        assert chosen != occupied, (
            f"pick_free_port handed out port {occupied} while another process "
            f"was listening on it. uvicorn then fails to bind and the window "
            f"never opens.")


def test_the_port_it_returns_can_actually_be_bound():
    """Whatever comes back must be usable, or the choice was meaningless."""
    with port_held_by_another_process() as occupied:
        chosen = pick_free_port(HOST, preferred=occupied, span=20)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, chosen))
            s.listen(1)


def test_exhausting_the_span_refuses_rather_than_returning_a_bad_port():
    with port_held_by_another_process() as occupied:
        with pytest.raises(RuntimeError) as exc:
            pick_free_port(HOST, preferred=occupied, span=1)
        assert "unavailable" in str(exc.value)
