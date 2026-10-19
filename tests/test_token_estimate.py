from yuzuru.router import estimate_tokens


def body(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def test_ascii_is_about_four_chars_per_token():
    n = estimate_tokens(body("x" * 40000))
    assert 9500 < n < 10500


def test_japanese_counts_one_token_per_character():
    # 3-byte UTF-8 characters must not be counted as ~1 token per byte.
    n = estimate_tokens(body("あ" * 10000))
    assert 9900 < n < 10200


def test_tools_count_too():
    tools = [{"type": "function", "function": {"name": "t", "description": "d" * 4000, "parameters": {}}}]
    assert estimate_tokens({"messages": [], "tools": tools}) > 1000
