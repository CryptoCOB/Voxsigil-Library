import asyncio
import sys
import os
from types import SimpleNamespace, ModuleType
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

# Provide dummy UnifiedAsyncBus module for agent imports
bus_module = ModuleType('UnifiedAsyncBus')
class MessageType:
    USER_INTERACTION = 'user_interaction'
    COMPONENT_STATUS = 'component_status'
class AsyncMessage:
    def __init__(self, *args, **kwargs):
        self.content = args[2] if len(args) > 2 else None
bus_module.MessageType = MessageType
bus_module.AsyncMessage = AsyncMessage
sys.modules['UnifiedAsyncBus'] = bus_module

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.voxagent import VoxAgent
from voxsigil_mesh import VoxSigilMesh


class DummyBus:
    def __init__(self):
        self.publish = AsyncMock()


class DummyEventBus:
    def __init__(self):
        self.events = []
    def emit(self, event_type, data=None, **kwargs):
        self.events.append((event_type, data))
    def subscribe(self, *args, **kwargs):
        pass


class DummyCore(SimpleNamespace):
    pass


class MeshEchoChainTest(IsolatedAsyncioTestCase):
    async def test_mesh_echo_chain(self):
        core = DummyCore()
        core.async_bus = DummyBus()
        core.event_bus = DummyEventBus()
        core.mesh = VoxSigilMesh(gui_hook=lambda msg: core.event_bus.emit('mesh_echo', msg))
        core.mesh.register('A', SimpleNamespace(receive=AsyncMock()))
        core.mesh.register('B', SimpleNamespace(receive=AsyncMock()))
        core.mesh.register('Logger', SimpleNamespace(receive=AsyncMock()))

        with patch(
            'blt_compression_middleware._compressor.compress',
            AsyncMock(return_value='compressed'),
        ) as mock_compress:
            agent = VoxAgent(core)
            agent.send('hello')
            await asyncio.sleep(0.05)
            mock_compress.assert_called()

        core.mesh.test_broadcast()

        core.async_bus.publish.assert_called()
        assert any('mesh_echo' == e[0] and 'compressed' in e[1] for e in core.event_bus.events)
