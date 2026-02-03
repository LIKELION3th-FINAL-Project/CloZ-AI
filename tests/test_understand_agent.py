from src.understand_agent.understand_model import UnderstandModel
from src.understand_agent.understand_model import extract_json_file
import pytest

@pytest.mark.parametrize(
    "query", 
    [
        "상의는 오버핏이고 하의는 반바지로 코디해줘",
        "아우터는 얇고 상의는 밝은 색으로 추천해줘",
        "홍대에 갈 건데 옷 추천해줘",
    ]
)

def test_understand_agnet_returns_parsable_json(query):
    agent = UnderstandModel()
    
    text = agent.run_understand_agent(query)
    assert isinstance(text, (dict, str))
    
    if isinstance(text, dict):
        parsed = text
    else:
        parsed = extract_json_file(text)
    
    assert parsed is not None
    assert isinstance(parsed, dict)