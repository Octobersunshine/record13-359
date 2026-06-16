import time
from text_deduplicator import (
    levenshtein_distance,
    similarity,
    ngram_fingerprint,
    jaccard_similarity,
    fast_similarity,
    TextDeduplicator
)


def test_levenshtein_distance():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("flaw", "lawn") == 2
    assert levenshtein_distance("hello", "hello") == 0
    print("✓ levenshtein_distance tests passed")


def test_similarity():
    assert similarity("", "") == 1.0
    assert similarity("abc", "") == 0.0
    assert similarity("", "abc") == 0.0
    assert similarity("hello", "hello") == 1.0
    assert 0.5 < similarity("hello", "hallo") < 1.0
    assert similarity("abc", "xyz") < 0.5
    print("✓ similarity tests passed")


def test_ngram_fingerprint():
    fp = ngram_fingerprint("hello", n=2)
    assert fp == {"he", "el", "ll", "lo"}
    fp = ngram_fingerprint("a", n=3)
    assert fp == {"a"}
    fp = ngram_fingerprint("", n=3)
    assert fp == set()
    print("✓ ngram_fingerprint tests passed")


def test_jaccard_similarity():
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0
    assert jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == 0.5
    assert jaccard_similarity(set(), set()) == 1.0
    assert jaccard_similarity({"a"}, set()) == 0.0
    print("✓ jaccard_similarity tests passed")


def test_fast_similarity():
    assert fast_similarity("hello", "hello") == 1.0
    assert fast_similarity("", "") == 1.0
    assert fast_similarity("abc", "") == 0.0
    assert fast_similarity("hello world", "hello world!") > 0.7
    assert fast_similarity("completely different", "totally unrelated") < 0.3
    print("✓ fast_similarity tests passed")


def test_deduplicator_exact_mode():
    texts = [
        "Hello World",
        "hello world",
        "Hello World!",
        "Python is great",
        "Python is good",
        "Completely different text"
    ]

    dedup = TextDeduplicator(threshold=0.8, mode="exact")
    result = dedup.deduplicate(texts)
    assert "Hello World" in result
    assert len(result) == 4
    print("✓ exact mode deduplication test passed")


def test_deduplicator_fast_mode():
    texts = [
        "Hello World",
        "hello world",
        "Hello World!",
        "Python is great",
        "Python is good",
        "Completely different text"
    ]

    dedup = TextDeduplicator(threshold=0.5, mode="fast")
    result = dedup.deduplicate(texts)
    assert "Hello World" in result
    print(f"  fast mode result count: {len(result)}")
    print("✓ fast mode deduplication test passed")


def test_deduplicator_hybrid_mode():
    texts = [
        "Hello World",
        "hello world",
        "Hello World!",
        "Python is great",
        "Python is good",
        "Completely different text"
    ]

    dedup = TextDeduplicator(threshold=0.8, mode="hybrid")
    result = dedup.deduplicate(texts)
    assert "Hello World" in result
    assert len(result) == 4
    print("✓ hybrid mode deduplication test passed")


def test_deduplicator_with_details():
    texts = [
        "The quick brown fox",
        "The quick brown fox!",
        "Totally unrelated sentence"
    ]

    dedup = TextDeduplicator(threshold=0.8, mode="exact")
    unique, duplicates = dedup.deduplicate_with_details(texts)

    assert len(unique) == 2
    assert len(duplicates) == 1
    assert duplicates[0][0] == "The quick brown fox!"
    assert duplicates[0][1] == "The quick brown fox"
    assert duplicates[0][2] >= 0.8
    print("✓ deduplication with details test passed")


def test_deduplicator_add_text():
    dedup = TextDeduplicator(threshold=0.85)
    assert dedup.add_text("Hello World") is True
    assert dedup.add_text("hello world") is False
    assert dedup.add_text("Another text") is True
    assert len(dedup.unique_texts) == 2
    print("✓ add_text test passed")


def test_deduplicator_is_duplicate():
    dedup = TextDeduplicator(threshold=0.8)
    existing = ["This is a test", "Another sentence"]

    assert dedup.is_duplicate("this is a test", existing) is True
    assert dedup.is_duplicate("completely new", existing) is False
    print("✓ is_duplicate test passed")


def test_deduplicator_case_sensitive():
    dedup = TextDeduplicator(threshold=0.9, ignore_case=False)
    assert dedup.add_text("Hello") is True
    assert dedup.add_text("hello") is True
    assert len(dedup.unique_texts) == 2
    print("✓ case sensitive test passed")


def test_deduplicator_clear():
    dedup = TextDeduplicator()
    dedup.add_text("test")
    assert len(dedup.unique_texts) == 1
    dedup.clear()
    assert len(dedup.unique_texts) == 0
    print("✓ clear test passed")


def test_invalid_mode():
    try:
        TextDeduplicator(mode="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("✓ invalid mode test passed")


def test_performance_comparison():
    base_text = "这是一段用于性能测试的长文本。" * 50
    texts = []
    for i in range(50):
        if i < 10:
            texts.append(base_text + f" 变体{i}")
        else:
            texts.append(f"完全不同的文本内容第{i}条。" * 30)

    print(f"  测试文本数量: {len(texts)}，平均长度: {sum(len(t) for t in texts) // len(texts)} 字符")

    start = time.time()
    dedup_exact = TextDeduplicator(threshold=0.8, mode="exact")
    result_exact = dedup_exact.deduplicate(texts)
    exact_time = time.time() - start
    print(f"  exact 模式耗时: {exact_time:.4f} 秒，保留 {len(result_exact)} 条")

    start = time.time()
    dedup_fast = TextDeduplicator(threshold=0.5, mode="fast")
    result_fast = dedup_fast.deduplicate(texts)
    fast_time = time.time() - start
    print(f"  fast 模式耗时: {fast_time:.4f} 秒，保留 {len(result_fast)} 条")

    start = time.time()
    dedup_hybrid = TextDeduplicator(threshold=0.8, mode="hybrid")
    result_hybrid = dedup_hybrid.deduplicate(texts)
    hybrid_time = time.time() - start
    print(f"  hybrid 模式耗时: {hybrid_time:.4f} 秒，保留 {len(result_hybrid)} 条")

    if exact_time > 0:
        speedup = exact_time / fast_time
        print(f"  fast 模式相对 exact 模式加速比: {speedup:.1f}x")

    print("✓ performance comparison test passed")


def run_all_tests():
    print("Running tests...\n")
    test_levenshtein_distance()
    test_similarity()
    test_ngram_fingerprint()
    test_jaccard_similarity()
    test_fast_similarity()
    test_deduplicator_exact_mode()
    test_deduplicator_fast_mode()
    test_deduplicator_hybrid_mode()
    test_deduplicator_with_details()
    test_deduplicator_add_text()
    test_deduplicator_is_duplicate()
    test_deduplicator_case_sensitive()
    test_deduplicator_clear()
    test_invalid_mode()
    test_performance_comparison()
    print("\n=== All tests passed! ===")


def run_demo():
    print("\n=== Text Deduplication Demo ===\n")

    texts = [
        "今天天气真好，适合出去玩",
        "今天天气真的好，适合出去玩",
        "Python 是一门优秀的编程语言",
        "Python 是一门很棒的编程语言",
        "机器学习正在改变世界",
        "深度学习是机器学习的一个分支",
        " 今天天气真好，适合出去玩 ",
    ]

    print("原始文本列表：")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    print(f"\n相似度阈值: 0.8，模式: hybrid (混合模式)")

    dedup = TextDeduplicator(threshold=0.8, mode="hybrid")
    unique_texts, duplicates = dedup.deduplicate_with_details(texts)

    print(f"\n去重后保留的文本 ({len(unique_texts)} 条):")
    for i, text in enumerate(unique_texts, 1):
        print(f"  {i}. {text}")

    if duplicates:
        print(f"\n被剔除的重复文本 ({len(duplicates)} 条):")
        for dup_text, original, sim in duplicates:
            print(f"  文本: '{dup_text}'")
            print(f"    与 '{original}' 相似度: {sim:.4f}")
            print()


if __name__ == "__main__":
    run_all_tests()
    run_demo()
