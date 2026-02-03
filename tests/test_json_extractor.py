from src.understand_agent.understand_model import extract_json_file
import pytest

def test_extrac_json_file_valid_json():
    text = "dummy text {'top': 'oversize', 'bottom': 'shorts'} dummy text"
    output = extract_json_file(text)
    assert isinstance(output, dict)
    assert output["top"] == "oversize"

def test_sxtract_json_file_none_when_no_braces():
    text = "no json here"
    output = extract_json_file(text)
    assert output is None

def test_extract_json_file_invalid_returns_none():
    text = "bad: {top: oversize}"
    output = extract_json_file(text)
    assert output is None