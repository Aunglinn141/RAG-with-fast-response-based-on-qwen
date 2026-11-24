"""Extract only the top-level answer string from an incomplete JSON response."""

import json

from local_rag.core.exceptions import DocumentError


def string_prefix(text, start):
    output, position = [], start
    escapes = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
    while position < len(text):
        char = text[position]
        if char == '"':
            break
        if char != "\\":
            if ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF:
                raise ValueError("Invalid character in answer")
            output.append(char)
            position += 1
            continue
        if position + 1 >= len(text):
            break
        escaped = text[position + 1]
        if escaped in escapes:
            output.append(escapes[escaped])
            position += 2
            continue
        if escaped != "u":
            raise ValueError("Invalid JSON escape")
        if position + 6 > len(text):
            break
        code = int(text[position + 2:position + 6], 16)
        if 0xD800 <= code <= 0xDBFF:
            if position + 12 > len(text):
                break
            if text[position + 6:position + 8] != "\\u":
                raise ValueError("Incomplete surrogate pair")
            low = int(text[position + 8:position + 12], 16)
            if not 0xDC00 <= low <= 0xDFFF:
                raise ValueError("Invalid surrogate pair")
            code = 0x10000 + ((code - 0xD800) << 10) + low - 0xDC00
            position += 12
        else:
            if 0xDC00 <= code <= 0xDFFF:
                raise ValueError("Unexpected low surrogate")
            position += 6
        output.append(chr(code))
    return "".join(output)


def answer_prefix(text):
    """Skip complete top-level values, never match an answer key inside evidence."""
    decoder = json.JSONDecoder()
    position = 0
    def whitespace(pos):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos
    position = whitespace(position)
    if position >= len(text):
        return ""
    if text[position] != "{":
        raise ValueError("Expected JSON object")
    position += 1
    while True:
        position = whitespace(position)
        if position >= len(text) or text[position] == "}":
            return ""
        try:
            key, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return ""
        if not isinstance(key, str):
            raise ValueError("Expected JSON key")
        position = whitespace(position)
        if position >= len(text):
            return ""
        if text[position] != ":":
            raise ValueError("Expected colon")
        position = whitespace(position + 1)
        if position >= len(text):
            return ""
        if key == "answer":
            if text[position] != '"':
                raise ValueError("Answer must be a string")
            return string_prefix(text, position + 1)
        try:
            _, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return ""
        position = whitespace(position)
        if position >= len(text) or text[position] == "}":
            return ""
        if text[position] != ",":
            raise ValueError("Expected comma")
        position += 1


class AnswerPreview:
    def __init__(self):
        self.raw = ""
        self.text = ""

    def feed(self, fragment):
        self.raw += fragment
        try:
            if len(self.raw) > 65536:
                raise ValueError("Structured output exceeded limit")
            current = answer_prefix(self.raw)
            if len(current) > 3000 or not current.startswith(self.text):
                raise ValueError("Invalid answer progression")
        except (ValueError, TypeError) as exc:
            raise DocumentError("invalid_answer", "本地模型未返回有效的回答格式，请重试或切换模型。", 502) from exc
        delta = current[len(self.text):]
        self.text = current
        return delta
