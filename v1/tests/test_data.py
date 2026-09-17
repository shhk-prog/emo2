from affective_empathy_eval.data import scale_vad

def test_scale_vad():
    assert scale_vad(5.0) == 1.0
    assert scale_vad(3.0) == 0.0
    assert scale_vad(1.0) == -1.0
