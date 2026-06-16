import time
from text_deduplicator import (
    levenshtein_distance,
    levenshtein_similarity,
    ngram_fingerprint,
    ngram_vector,
    jaccard_similarity,
    cosine_similarity,
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


def test_levenshtein_similarity():
    assert levenshtein_similarity("", "") == 1.0
    assert levenshtein_similarity("abc", "") == 0.0
    assert levenshtein_similarity("", "abc") == 0.0
    assert levenshtein_similarity("hello", "hello") == 1.0
    assert 0.5 < levenshtein_similarity("hello", "hallo") < 1.0
    assert levenshtein_similarity("abc", "xyz") < 0.5
    print("✓ levenshtein_similarity tests passed")


def test_ngram_fingerprint():
    fp = ngram_fingerprint("hello", n=2)
    assert fp == {"he", "el", "ll", "lo"}
    fp = ngram_fingerprint("a", n=3)
    assert fp == {"a"}
    fp = ngram_fingerprint("", n=3)
    assert fp == set()
    print("✓ ngram_fingerprint tests passed")


def test_ngram_vector():
    vec = ngram_vector("hello", n=2)
    assert vec == {"he": 1, "el": 1, "ll": 1, "lo": 1}
    vec = ngram_vector("aaa", n=2)
    assert vec == {"aa": 2}
    vec = ngram_vector("", n=3)
    assert vec == {}
    print("✓ ngram_vector tests passed")


def test_jaccard_similarity():
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0
    assert jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == 0.5
    assert jaccard_similarity(set(), set()) == 1.0
    assert jaccard_similarity({"a"}, set()) == 0.0
    print("✓ jaccard_similarity tests passed")


def test_cosine_similarity():
    assert abs(cosine_similarity({"a": 1, "b": 2}, {"a": 1, "b": 2}) - 1.0) < 1e-9
    assert cosine_similarity({"a": 1}, {"b": 1}) == 0.0
    assert abs(cosine_similarity({"a": 3, "b": 4}, {"a": 6, "b": 8}) - 1.0) < 1e-9
    assert cosine_similarity({}, {}) == 0.0
    assert cosine_similarity({"a": 1}, {}) == 0.0
    sim = cosine_similarity({"a": 1, "b": 1}, {"b": 1, "c": 1})
    assert 0 < sim < 1
    print("✓ cosine_similarity tests passed")


def test_algorithm_levenshtein():
    texts = [
        "Hello World",
        "hello world",
        "Hello World!",
        "Python is great",
        "Python is good",
        "Completely different text"
    ]

    dedup = TextDeduplicator(threshold=0.8, algorithm="levenshtein", mode="exact")
    result = dedup.deduplicate(texts)
    assert "Hello World" in result
    assert len(result) == 4
    print("✓ levenshtein algorithm test passed")


def test_algorithm_jaccard():
    texts = [
        "Hello World",
        "hello world",
        "Hello World!",
        "Python is great",
        "Python is good",
        "Completely different text"
    ]

    dedup = TextDeduplicator(threshold=0.5, algorithm="jaccard")
    result = dedup.deduplicate(texts)
    assert "Hello World" in result
    print(f"  jaccard result count: {len(result)}")
    print("✓ jaccard algorithm test passed")


def test_algorithm_cosine():
    texts = [
        "Hello World",
        "hello world",
        "Hello World!",
        "Python is great",
        "Python is good",
        "Completely different text"
    ]

    dedup = TextDeduplicator(threshold=0.5, algorithm="cosine")
    result = dedup.deduplicate(texts)
    assert "Hello World" in result
    print(f"  cosine result count: {len(result)}")
    print("✓ cosine algorithm test passed")


def test_deduplicator_with_details():
    texts = [
        "The quick brown fox",
        "The quick brown fox!",
        "Totally unrelated sentence"
    ]

    dedup = TextDeduplicator(threshold=0.8, algorithm="levenshtein", mode="exact")
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


def test_invalid_algorithm():
    try:
        TextDeduplicator(algorithm="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("✓ invalid algorithm test passed")


def test_invalid_mode():
    try:
        TextDeduplicator(mode="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("✓ invalid mode test passed")


def test_all_algorithms_consistency():
    texts = [
        "完全相同的文本",
        "完全相同的文本",
        "完全不同的内容"
    ]

    for algo in ("levenshtein", "jaccard", "cosine"):
        threshold = 0.8 if algo == "levenshtein" else 0.5
        dedup = TextDeduplicator(threshold=threshold, algorithm=algo)
        result = dedup.deduplicate(texts)
        assert len(result) == 2, f"{algo} should have 2 unique texts"
    print("✓ all algorithms consistency test passed")


def test_performance_comparison():
    base_text = "这是一段用于性能测试的长文本。" * 50
    texts = []
    for i in range(50):
        if i < 10:
            texts.append(base_text + f" 变体{i}")
        else:
            texts.append(f"完全不同的文本内容第{i}条。" * 30)

    print(f"  测试文本数量: {len(texts)}，平均长度: {sum(len(t) for t in texts) // len(texts)} 字符")

    algorithms = [
        ("levenshtein exact", 0.8, "levenshtein", "exact"),
        ("levenshtein hybrid", 0.8, "levenshtein", "hybrid"),
        ("jaccard", 0.5, "jaccard", "exact"),
        ("cosine", 0.5, "cosine", "exact"),
    ]

    results = []
    for name, threshold, algo, mode in algorithms:
        start = time.time()
        dedup = TextDeduplicator(threshold=threshold, algorithm=algo, mode=mode)
        result = dedup.deduplicate(texts)
        elapsed = time.time() - start
        results.append((name, elapsed, len(result)))
        print(f"  {name}: {elapsed:.4f} 秒，保留 {len(result)} 条")

    if results[0][1] > 0:
        speedup = results[0][1] / results[2][1]
        print(f"  jaccard 相对 levenshtein exact 加速比: {speedup:.1f}x")

    print("✓ performance comparison test passed")


def run_all_tests():
    print("Running tests...\n")
    test_levenshtein_distance()
    test_levenshtein_similarity()
    test_ngram_fingerprint()
    test_ngram_vector()
    test_jaccard_similarity()
    test_cosine_similarity()
    test_algorithm_levenshtein()
    test_algorithm_jaccard()
    test_algorithm_cosine()
    test_deduplicator_with_details()
    test_deduplicator_add_text()
    test_deduplicator_is_duplicate()
    test_deduplicator_case_sensitive()
    test_deduplicator_clear()
    test_invalid_algorithm()
    test_invalid_mode()
    test_all_algorithms_consistency()
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

    algorithms = [
        ("levenshtein (编辑距离)", 0.8, "levenshtein", "hybrid"),
        ("jaccard (Jaccard 相似度)", 0.5, "jaccard", "exact"),
        ("cosine (余弦相似度)", 0.5, "cosine", "exact"),
    ]

    for name, threshold, algo, mode in algorithms:
        print(f"\n--- {name} ---")
        print(f"阈值: {threshold}")
        dedup = TextDeduplicator(threshold=threshold, algorithm=algo, mode=mode)
        unique_texts, duplicates = dedup.deduplicate_with_details(texts)
        print(f"去重后保留: {len(unique_texts)} 条，剔除: {len(duplicates)} 条")

        if duplicates:
            for dup_text, original, sim in duplicates[:2]:
                print(f"   '{dup_text[:20]}...' 与 '{original[:20]}...' 相似度: {sim:.4f}")


if __name__ == "__main__":
    run_all_tests()
    run_demo()
