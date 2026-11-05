"""Parse Liquid's native XML-like calls without repairing malformed answers."""
import json
import re


def parse_output(text, thinking=False):
    for end in ["<|im_end|>", "<|endoftext|>"]:
        text = text.removesuffix(end).rstrip()
    if thinking and "</think>" not in text:
        return None, "unfinished_thinking"
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None, "unfinished_thinking"
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        return None, "call_count"
    block = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    match = re.fullmatch(r"<function=([A-Za-z_]\w*)>\s*(.*?)\s*</function>", block, re.S)
    if not match:
        return None, "function_syntax"
    name, body = match.groups()
    arguments = {}
    while body.strip():
        parameter = re.match(r"\s*<parameter=([A-Za-z_]\w*)>\s*(.*?)\s*</parameter>", body, re.S)
        if not parameter:
            return None, "parameter_syntax"
        key, value = parameter.groups()
        if key in arguments:
            return None, "duplicate_parameter"
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = value.strip()
        arguments[key] = value
        body = body[parameter.end():]
    return {"name": name, "arguments": arguments}, None
