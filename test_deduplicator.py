from text_deduplicator import (
    levenshtein_distance,
    similarity,
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


def test_deduplicator_basic():
    texts = [
        "Hello World",
        "hello world",
        "Hello World!",
        "Python is great",
        "Python is good",
        "Completely different text"
    ]

    dedup = TextDeduplicator(threshold=0.8)
    result = dedup.deduplicate(texts)
    assert "Hello World" in result
    assert "Python is great" in result
    assert "Python is good" in result
    assert "Completely different text" in result
    assert len(result) == 4
    print("✓ basic deduplication test passed")


def test_deduplicator_with_details():
    texts = [
        "The quick brown fox",
        "The quick brown fox!",
        "Totally unrelated sentence"
    ]

    dedup = TextDeduplicator(threshold=0.8)
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


def run_all_tests():
    print("Running tests...\n")
    test_levenshtein_distance()
    test_similarity()
    test_deduplicator_basic()
    test_deduplicator_with_details()
    test_deduplicator_add_text()
    test_deduplicator_is_duplicate()
    test_deduplicator_case_sensitive()
    test_deduplicator_clear()
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

    print(f"\n相似度阈值: 0.8")

    dedup = TextDeduplicator(threshold=0.8)
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
