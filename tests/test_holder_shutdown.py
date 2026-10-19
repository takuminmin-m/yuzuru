import asyncio
import sys

from .conftest import FakeProbe
from yuzuru.server import State


async def test_stopping_the_server_frees_hard_reservations(cfg, world, http, monkeypatch):
    st = State(cfg, probe=FakeProbe(world), http=http)
    await st.start()
    holder = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(60)",
                                                  start_new_session=True)

    async def fake_spawn(gpus, sizes, ttl):
        st.broker._holders[holder.pid] = holder
        return holder.pid

    monkeypatch.setattr(st.broker, "_spawn_holder", fake_spawn)
    await st.broker.create_reservation(gpus=["1"], vram=None, ttl=None, label="x", hard=True)
    await st.stop()
    assert holder.returncode is not None  # terminated, not orphaned
