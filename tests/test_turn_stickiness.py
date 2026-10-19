from .conftest import FakeProbe
from yuzuru.broker import Broker
from yuzuru.catalog import Catalog
from yuzuru.config import GB, RouterConfig
from yuzuru.ollama import Ollama
from yuzuru.router import Router


async def test_continuation_stays_on_the_downgraded_model(cfg, world, http):
    cfg.router = RouterConfig(classifier="heuristic", prefer_loaded=False)
    ollama = Ollama("http://ollama", http)
    catalog = Catalog(cfg, ollama)
    await catalog.refresh()
    broker = Broker(cfg, FakeProbe(world), ollama, catalog)
    world.foreign = [0, 11 * GB]  # a job holds most of the GPU memory
    await broker.refresh()
    router = Router(cfg, catalog, broker, ollama)
    body = {"model": "auto", "messages": [{"role": "user", "content": "この関数にテストを書いて"}]}
    d = await router.route(body, {"x-yuzuru-session": "s"})
    assert d.wanted == "medium" and d.tier == "small"
    served = d.tier
    # The job ends mid-turn: medium would fit now, but the tool loop keeps its model.
    world.foreign = [0, 0]
    await broker.refresh()
    body["messages"] += [{"role": "assistant", "content": ""}, {"role": "tool", "content": "ok"}]
    d = await router.route(body, {"x-yuzuru-session": "s"})
    assert d.tier == served and "continuation" in d.reason
    # Next turn: the wanted tier again.
    body["messages"] += [{"role": "user", "content": "この関数にテストを書いて"}]
    d = await router.route(body, {"x-yuzuru-session": "s"})
    assert d.tier == "medium"
