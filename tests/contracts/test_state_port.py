"""Contract test: StatePort protocol."""

import pytest

pytestmark = pytest.mark.contract

pytestmark = pytest.mark.asyncio

FAKE_STATE = type(
    "FakeState",
    (),
    {
        "project_id": "test_proj",
        "status": "running",
        "tasks": [],
        "budget": None,
    },
)()


class StatePortContract:
    """Base class — subclass and override create_state()"""

    @pytest.fixture
    def state(self):
        raise NotImplementedError

    async def test_save_then_load(self, state):
        await state.save_project("p1", FAKE_STATE)
        loaded = await state.load_project("p1")
        assert loaded is not None
        assert loaded.project_id == "test_proj"

    async def test_load_missing_returns_none(self, state):
        loaded = await state.load_project("nonexistent")
        assert loaded is None

    async def test_save_checkpoint(self, state):
        await state.save_checkpoint("p1", "t1", FAKE_STATE)
        # StatePort does not specify a load_checkpoint — just verify no error
        assert True

    async def test_close_is_idempotent(self, state):
        await state.close()
        await state.close()

    async def test_project_lease_is_exclusive_while_held(self, state):
        async with state.project_lease("p1") as first:
            assert first is True
            async with state.project_lease("p1") as second:
                assert second is False
            # A different project is not blocked by the first lease.
            async with state.project_lease("p2") as other:
                assert other is True

    async def test_project_lease_is_released_on_exit(self, state):
        async with state.project_lease("p1") as acquired:
            assert acquired is True
        async with state.project_lease("p1") as again:
            assert again is True

    async def test_project_lease_is_released_on_error(self, state):
        with pytest.raises(RuntimeError):
            async with state.project_lease("p1"):
                raise RuntimeError("run failed")
        async with state.project_lease("p1") as again:
            assert again is True

    async def test_is_runtime_checkable(self, state):
        from orchestrator.domain.ports import StatePort

        assert isinstance(state, StatePort)


class TestNullStateContract(StatePortContract):
    @pytest.fixture
    def state(self):
        from orchestrator.domain.ports import NullState

        return NullState()
