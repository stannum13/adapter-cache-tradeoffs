import pytest

from adapter_cache_bench.backends.base import make_backend
from adapter_cache_bench.backends.mock_backend import MockBackend
from adapter_cache_bench.backends.vllm_backend import VLLMBackend
from adapter_cache_bench.config import BackendConfig


def test_make_backend_returns_mock_backend_by_default():
    backend = make_backend(BackendConfig(kind="mock"))

    assert isinstance(backend, MockBackend)


def test_make_backend_returns_vllm_backend_for_vllm_kind():
    backend = make_backend(BackendConfig(kind="vllm"))

    assert isinstance(backend, VLLMBackend)


def test_make_backend_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown backend kind"):
        make_backend(BackendConfig(kind="other"))
