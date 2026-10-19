import os

from .conftest import FakeProbe
from yuzuru.broker import Broker
from yuzuru.catalog import Catalog
from yuzuru.ollama import Ollama


async def setup(cfg, world, http):
    ollama = Ollama("http://ollama", http)
    catalog = Catalog(cfg, ollama)
    await catalog.refresh()
    broker = Broker(cfg, FakeProbe(world), ollama, catalog)
    await broker.refresh()
    return catalog, broker


async def test_release_does_not_reload_a_model_unloaded_meanwhile(cfg, world, http):
    catalog, broker = await setup(cfg, world, http)
    spec = catalog.specs["medium"]
    await broker.acquire(spec)
    # A training job takes a lease while the request is still generating.
    await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None)
    assert spec.variant not in world.loaded
    await broker.release(spec)  # the request finishes
    assert spec.variant not in world.loaded


async def test_release_refreshes_keep_alive_of_a_resident_model(cfg, world, http):
    catalog, broker = await setup(cfg, world, http)
    spec = catalog.specs["small"]
    await broker.acquire(spec)
    calls = []
    orig = broker.ollama.load

    async def spy(model, keep_alive):
        calls.append(model)
        await orig(model, keep_alive)

    broker.ollama.load = spy
    await broker.release(spec)
    assert calls == [spec.variant]
