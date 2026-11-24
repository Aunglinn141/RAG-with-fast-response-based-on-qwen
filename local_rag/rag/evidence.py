"""Map formatting-only citation differences back to an exact source substring."""

_QUOTATION_MARKS = frozenset('"\'“”‘’「」『』')


def _word_character(char):
    return char.isascii() and char.isalnum()


def _normalized(text):
    # Keep apostrophes inside Latin words: "can't" must never match "cant".
    tokens = [(char, index) for index, char in enumerate(text)
              if char not in _QUOTATION_MARKS
              or (char in {"'", "’"} and 0 < index < len(text) - 1
                  and _word_character(text[index - 1]) and _word_character(text[index + 1]))]
    characters, positions = [], []
    index = 0
    while index < len(tokens):
        char, position = tokens[index]
        if char.isspace():
            end = index + 1
            while end < len(tokens) and tokens[end][0].isspace():
                end += 1
            # Preserve word and number boundaries even across line wrapping.
            if (characters and end < len(tokens)
                    and _word_character(characters[-1]) and _word_character(tokens[end][0])):
                characters.append(" ")
                positions.append(position)
            index = end
            continue
        characters.append(char)
        positions.append(position)
        index += 1
    return "".join(characters), positions


def match_evidence_quote(source_text: str, quote: str) -> str | None:
    """Return original text, never a rewritten or model-generated quotation.

    Only quotation marks and layout whitespace may differ. Other punctuation,
    words, numbers, negation, and omitted content must still match. Ambiguous
    normalized matches and restored excerpts over 800 characters fail closed.
    """
    if not 4 <= len(quote.strip()) <= 800:
        return None
    candidate, _ = _normalized(quote)
    if len(candidate) < 4:
        return None
    if quote in source_text:
        return quote
    normalized_source, positions = _normalized(source_text)
    start = normalized_source.find(candidate)
    if start < 0 or normalized_source.find(candidate, start + 1) >= 0:
        return None
    original = source_text[positions[start]:positions[start + len(candidate) - 1] + 1]
    return original if len(original) <= 800 else None
