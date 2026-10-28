"""Resolve the assertion of a symptom mention within its clause."""

import re

from AI.symptom.normalizer import SymptomNormalizer


BOUNDARY = re.compile(r"[.!?;\n]|\b(?:nhưng|tuy nhiên|còn|mà|song)\b")
NEGATION = re.compile(r"\b(?:không|chưa|chẳng|chả|hết)\b")
# These constructions do not deny the following symptom.
NON_NEGATION = re.compile(r"\bkhông\s+(?:chỉ|những|những)\b")
AFFIRMATION = re.compile(r"\b(?:có|bị|thấy|xuất hiện)\b")


def mention_is_affirmed(text: str, start: int) -> bool:
    prefix = text[:start]
    boundaries = list(BOUNDARY.finditer(prefix))
    if boundaries:
        prefix = prefix[boundaries[-1].end():]
    prefix = NON_NEGATION.sub("", prefix)
    negations = list(NEGATION.finditer(prefix))
    if not negations:
        return True
    # A new explicit affirmative predicate ends a coordinated negative list:
    # "không sốt, không ho, có đau đầu".
    tail = prefix[negations[-1].end():]
    new_clause = re.search(r"(?:,|\bvà\b|\bkèm\b)\s*(.*)$", tail)
    if new_clause and AFFIRMATION.search(new_clause.group(1)):
        return True
    return False


def symptom_polarity(text: str, phrases: list[str]) -> bool | None:
    text = SymptomNormalizer.normalize_text(text)
    mentions = []
    for phrase in phrases:
        phrase = SymptomNormalizer.normalize_text(phrase)
        for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text):
            mentions.append((match.start(), mention_is_affirmed(text, match.start())))
    # The last mention wins when the patient corrects a previous statement.
    return max(mentions, key=lambda item: item[0])[1] if mentions else None
