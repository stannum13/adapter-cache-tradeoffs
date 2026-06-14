from specialization_cache_frontier.cache.prefix_table import CacheBlock, PrefixTable


def test_prefix_table_tracks_hits_and_tokens():
    table = PrefixTable()
    block = CacheBlock(key="a", logical_key="x", token_count=4)
    assert not table.contains(block)
    table.add(block)
    assert table.contains(block)
    assert table.cached_tokens() == 4
    assert table.logical_tokens() == 4
    assert table.hit_rate() == 0.5


def test_prefix_table_evicts_lru_blocks_when_token_budget_is_exceeded():
    table = PrefixTable(max_memory_tokens=4)
    first = CacheBlock(key="a", logical_key="x", token_count=2)
    second = CacheBlock(key="b", logical_key="y", token_count=2)
    third = CacheBlock(key="c", logical_key="z", token_count=2)

    table.add(first)
    table.add(second)
    assert table.contains(first)
    table.add(third)

    assert first.key in table.blocks
    assert second.key not in table.blocks
    assert third.key in table.blocks
    assert table.eviction_count == 1
    assert table.evicted_token_count == 2
