from typing import List, Tuple, Optional


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


class TextDeduplicator:
    def __init__(self, threshold: float = 0.8, ignore_case: bool = True, strip_whitespace: bool = True):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        self.ignore_case = ignore_case
        self.strip_whitespace = strip_whitespace
        self._unique_texts: List[str] = []

    def _normalize(self, text: str) -> str:
        normalized = text
        if self.strip_whitespace:
            normalized = normalized.strip()
        if self.ignore_case:
            normalized = normalized.lower()
        return normalized

    def is_duplicate(self, text: str, existing_texts: Optional[List[str]] = None) -> bool:
        normalized_text = self._normalize(text)
        texts_to_check = existing_texts if existing_texts is not None else self._unique_texts

        for existing in texts_to_check:
            normalized_existing = self._normalize(existing)
            sim = similarity(normalized_text, normalized_existing)
            if sim >= self.threshold:
                return True
        return False

    def add_text(self, text: str) -> bool:
        if not self.is_duplicate(text):
            self._unique_texts.append(text)
            return True
        return False

    def deduplicate(self, texts: List[str]) -> List[str]:
        self._unique_texts = []
        for text in texts:
            self.add_text(text)
        return self._unique_texts.copy()

    def deduplicate_with_details(self, texts: List[str]) -> Tuple[List[str], List[Tuple[str, str, float]]]:
        self._unique_texts = []
        duplicates: List[Tuple[str, str, float]] = []

        for text in texts:
            normalized_text = self._normalize(text)
            found_duplicate = False

            for existing in self._unique_texts:
                normalized_existing = self._normalize(existing)
                sim = similarity(normalized_text, normalized_existing)
                if sim >= self.threshold:
                    duplicates.append((text, existing, sim))
                    found_duplicate = True
                    break

            if not found_duplicate:
                self._unique_texts.append(text)

        return self._unique_texts.copy(), duplicates

    @property
    def unique_texts(self) -> List[str]:
        return self._unique_texts.copy()

    def clear(self) -> None:
        self._unique_texts = []
