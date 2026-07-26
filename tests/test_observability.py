"""
Unit tests for Kalki Nexus Observability module.
"""
from core.observability import extract_token_usage


class MockResponseWithUsage:
    def __init__(self, input_tokens=10, output_tokens=20, total_tokens=30):
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }


class MockResponseWithResponseMeta:
    def __init__(self, prompt_tokens=15, completion_tokens=25, total_tokens=40):
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        }


def test_extract_token_usage_standard():
    resp = MockResponseWithUsage()
    usage = extract_token_usage(resp)
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 20
    assert usage["total_tokens"] == 30


def test_extract_token_usage_fallback():
    resp = MockResponseWithResponseMeta()
    usage = extract_token_usage(resp)
    assert usage["input_tokens"] == 15
    assert usage["output_tokens"] == 25
    assert usage["total_tokens"] == 40


def test_extract_token_usage_empty():
    usage = extract_token_usage(None)
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
