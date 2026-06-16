from typing import List, Tuple, Optional, Set


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


def ngram_fingerprint(text: str, n: int = 3) -> Set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(fp1: Set[str], fp2: Set[str]) -> float:
    if not fp1 and not fp2:
        return 1.0
    if not fp1 or not fp2:
        return 0.0
    intersection = len(fp1 & fp2)
    union = len(fp1 | fp2)
    return intersection / union if union > 0 else 0.0


def fast_similarity(s1: str, s2: str, n: int = 3) -> float:
    fp1 = ngram_fingerprint(s1, n)
    fp2 = ngram_fingerprint(s2, n)
    return jaccard_similarity(fp1, fp2)


class TextDeduplicator:
    def __init__(
        self,
        threshold: float = 0.8,
        ignore_case: bool = True,
        strip_whitespace: bool = True,
        mode: str = "hybrid",
        ngram_size: int = 3,
        length_diff_threshold: float = 0.3,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if mode not in ("exact", "fast", "hybrid"):
            raise ValueError("mode must be 'exact', 'fast', or 'hybrid'")

        self.threshold = threshold
        self.ignore_case = ignore_case
        self.strip_whitespace = strip_whitespace
        self.mode = mode
        self.ngram_size = ngram_size
        self.length_diff_threshold = length_diff_threshold

        self._unique_texts: List[str] = []
        self._unique_fingerprints: List[Set[str]] = []
        self._unique_lengths: List[int] = []

    def _normalize(self, text: str) -> str:
        normalized = text
        if self.strip_whitespace:
            normalized = normalized.strip()
        if self.ignore_case:
            normalized = normalized.lower()
        return normalized

    def _length_pass(self, len1: int, len2: int) -> bool:
        if len1 == 0 and len2 == 0:
            return True
        if len1 == 0 or len2 == 0:
            return False
        max_len = max(len1, len2)
        min_len = min(len1, len2)
        return (min_len / max_len) >= (1.0 - self.length_diff_threshold)

    def _compute_similarity(self, norm_text: str, fp: Set[str], existing_norm: str, existing_fp: Set[str]) -> float:
        if self.mode == "fast":
            return jaccard_similarity(fp, existing_fp)
        elif self.mode == "exact":
            return similarity(norm_text, existing_norm)
        else:
            fast_sim = jaccard_similarity(fp, existing_fp)
            lower_bound = self.threshold * 0.4
            if fast_sim < lower_bound:
                return 0.0
            return similarity(norm_text, existing_norm)

    def is_duplicate(self, text: str, existing_texts: Optional[List[str]] = None) -> bool:
        normalized_text = self._normalize(text)
        fp = ngram_fingerprint(normalized_text, self.ngram_size)
        text_len = len(normalized_text)

        if existing_texts is not None:
            for existing in existing_texts:
                norm_existing = self._normalize(existing)
                existing_fp = ngram_fingerprint(norm_existing, self.ngram_size)
                existing_len = len(norm_existing)

                if not self._length_pass(text_len, existing_len):
                    continue

                sim = self._compute_similarity(normalized_text, fp, norm_existing, existing_fp)
                if sim >= self.threshold:
                    return True
            return False
        else:
            for i in range(len(self._unique_texts)):
                existing_len = self._unique_lengths[i]
                if not self._length_pass(text_len, existing_len):
                    continue

                existing_fp = self._unique_fingerprints[i]
                existing_norm = self._normalize(self._unique_texts[i])

                sim = self._compute_similarity(normalized_text, fp, existing_norm, existing_fp)
                if sim >= self.threshold:
                    return True
            return False

    def add_text(self, text: str) -> bool:
        if not self.is_duplicate(text):
            normalized = self._normalize(text)
            self._unique_texts.append(text)
            self._unique_fingerprints.append(ngram_fingerprint(normalized, self.ngram_size))
            self._unique_lengths.append(len(normalized))
            return True
        return False

    def deduplicate(self, texts: List[str]) -> List[str]:
        self.clear()
        for text in texts:
            self.add_text(text)
        return self._unique_texts.copy()

    def deduplicate_with_details(self, texts: List[str]) -> Tuple[List[str], List[Tuple[str, str, float]]]:
        self.clear()
        duplicates: List[Tuple[str, str, float]] = []

        for text in texts:
            normalized_text = self._normalize(text)
            fp = ngram_fingerprint(normalized_text, self.ngram_size)
            text_len = len(normalized_text)
            found_duplicate = False

            for i in range(len(self._unique_texts)):
                existing_len = self._unique_lengths[i]
                if not self._length_pass(text_len, existing_len):
                    continue

                existing_fp = self._unique_fingerprints[i]
                existing_norm = self._normalize(self._unique_texts[i])

                sim = self._compute_similarity(normalized_text, fp, existing_norm, existing_fp)
                if sim >= self.threshold:
                    duplicates.append((text, self._unique_texts[i], sim))
                    found_duplicate = True
                    break

            if not found_duplicate:
                self._unique_texts.append(text)
                self._unique_fingerprints.append(fp)
                self._unique_lengths.append(text_len)

        return self._unique_texts.copy(), duplicates

    @property
    def unique_texts(self) -> List[str]:
        return self._unique_texts.copy()

    def clear(self) -> None:
        self._unique_texts = []
        self._unique_fingerprints = []
        self._unique_lengths = []
