"""Unit tests for duro.state_store — no external services required."""

from __future__ import annotations

import tempfile

import pytest

from duro.state_store import FileStateStore, InlineStore, PayloadTooLargeError


@pytest.fixture
def tmp_dir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestInlineStore:
    @pytest.mark.asyncio
    async def test_put_raises(self) -> None:
        store = InlineStore(max_inline_bytes=100)
        with pytest.raises(PayloadTooLargeError) as exc_info:
            await store.put("key", b"x" * 200)
        assert exc_info.value.size_bytes == 200
        assert exc_info.value.max_bytes == 100

    @pytest.mark.asyncio
    async def test_get_raises(self) -> None:
        store = InlineStore()
        with pytest.raises(RuntimeError):
            await store.get("key")

    @pytest.mark.asyncio
    async def test_delete_noop(self) -> None:
        store = InlineStore()
        await store.delete("key")  # should not raise


class TestFileStateStore:
    @pytest.mark.asyncio
    async def test_put_get_delete(self, tmp_dir: str) -> None:
        store = FileStateStore(directory=tmp_dir)
        data = b"hello world"

        await store.put("test/blob", data)
        result = await store.get("test/blob")
        assert result == data

        await store.delete("test/blob")
        with pytest.raises(FileNotFoundError):
            await store.get("test/blob")

    @pytest.mark.asyncio
    async def test_nested_keys(self, tmp_dir: str) -> None:
        store = FileStateStore(directory=tmp_dir)
        data = b"nested data"

        await store.put("wf/act/state", data)
        result = await store.get("wf/act/state")
        assert result == data

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tmp_dir: str) -> None:
        store = FileStateStore(directory=tmp_dir)
        await store.delete("nonexistent")  # should not raise
