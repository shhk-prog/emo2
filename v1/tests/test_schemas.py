import pytest
from affective_empathy_eval.schemas import parse_affective_state

def test_parse_valid_json():
    text = '{"valence": 5, "arousal": 7}'
    result = parse_affective_state(text)
    assert result["valence"] == 5
    assert result["arousal"] == 7

def test_parse_invalid_json():
    text = '{"valence": 5, "arousal": '
    with pytest.raises(ValueError):
        parse_affective_state(text)

def test_parse_out_of_bounds():
    text = '{"valence": 10, "arousal": 0}'
    with pytest.raises(ValueError):
        parse_affective_state(text)
        
def test_parse_float_rejected():
    text = '{"valence": 5.5, "arousal": 7.0}'
    with pytest.raises(ValueError):
        parse_affective_state(text)
