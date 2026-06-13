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
