import json

from .conftest import FakeProbe
from yuzuru.broker import Broker
from yuzuru.catalog import Catalog
from yuzuru.ollama import Ollama
from yuzuru.router import Router


async def test_classifier_uses_the_resident_small_model(cfg, world, http):
    ollama = Ollama("http://ollama", http)
    catalog = Catalog(cfg, ollama)
    await catalog.refresh()
    broker = Broker(cfg, FakeProbe(world), ollama, catalog)
    small = catalog.specs["small"]
    assert small.base == catalog.router_spec.base
    await broker.acquire(small)
    await broker.release(small, touch=False)
    router = Router(cfg, catalog, broker, ollama)
    asked = []
    orig = router.ollama.chat

    async def spy(payload, timeout=None):
        asked.append(payload["model"])
        return await orig(payload, timeout)

    router.ollama.chat = spy
    level, reason = await router.classify("この関数にテストを書いて", 2)
    assert asked == [small.variant]
    assert catalog.router_spec.variant not in world.loaded  # no second copy loaded


async def test_classifier_falls_back_to_its_cpu_copy(cfg, world, http):
    ollama = Ollama("http://ollama", http)
    catalog = Catalog(cfg, ollama)
    await catalog.refresh()
    broker = Broker(cfg, FakeProbe(world), ollama, catalog)
    router = Router(cfg, catalog, broker, ollama)
    await router.classify("hi", 1)
    assert catalog.router_spec.variant in world.loaded
