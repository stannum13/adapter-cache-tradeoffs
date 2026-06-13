from specialization_cache_frontier.cache.activated_lora_cache import ActivatedLoRACache
from specialization_cache_frontier.cache.copy_on_write_cache import CopyOnWriteCache
from specialization_cache_frontier.cache.standard_lora_cache import StandardLoRACache
from specialization_cache_frontier.config import CacheConfig


def test_same_prompt_same_adapter_reuses_cache():
    cache = StandardLoRACache(CacheConfig(block_size=2))
    prompt = "alpha beta gamma delta"
    cache.observe_request("qa", prompt, "tenant-a", "trust-a")
    assert cache.estimate_cached_prefix_tokens("qa", prompt, "tenant-a", "trust-a") == 4


def test_same_prompt_different_adapter_fragments_standard_lora_cache():
    cache = StandardLoRACache(CacheConfig(block_size=2))
    prompt = "alpha beta gamma delta"
    cache.observe_request("qa", prompt, "tenant-a", "trust-a")
    assert cache.estimate_cached_prefix_tokens("json", prompt, "tenant-a", "trust-a") == 0
    cache.observe_request("json", prompt, "tenant-a", "trust-a")
    assert cache.fragmentation_index() == 2.0


def test_activated_lora_shares_prefix_before_invocation():
    cache = ActivatedLoRACache(CacheConfig(block_size=2))
    qa_prompt = "shared document tokens here <ADAPTER:qa> answer question"
    json_prompt = "shared document tokens here <ADAPTER:json> extract fields"
    cache.observe_request("qa", qa_prompt, "tenant-a", "trust-a")
    assert cache.estimate_cached_prefix_tokens("json", json_prompt, "tenant-a", "trust-a") == 4


def test_cache_trust_groups_isolate_reuse():
    cache = StandardLoRACache(CacheConfig(block_size=2, isolation_scope="trust_group"))
    prompt = "alpha beta gamma delta"
    cache.observe_request("qa", prompt, "tenant-a", "trust-a")
    assert cache.estimate_cached_prefix_tokens("qa", prompt, "tenant-a", "trust-b") == 0


def test_copy_on_write_uses_less_memory_than_standard_lora_for_multi_adapter_shared_prefix():
    config = CacheConfig(block_size=4)
    standard = StandardLoRACache(config)
    cow = CopyOnWriteCache(config)
    prompts = [
        "shared prefix tokens repeat repeat <ADAPTER:qa> task qa",
        "shared prefix tokens repeat repeat <ADAPTER:json> task json",
        "shared prefix tokens repeat repeat <ADAPTER:summary> task summary",
    ]
    for prompt, adapter in zip(prompts, ["qa", "json", "summary"], strict=True):
        standard.observe_request(adapter, prompt, "tenant-a", "trust-a")
        cow.observe_request(adapter, prompt, "tenant-a", "trust-a")
    assert cow.memory_tokens() < standard.memory_tokens()
