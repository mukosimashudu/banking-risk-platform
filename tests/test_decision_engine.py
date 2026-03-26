from src.scoring.decision_engine import make_decision

def test_reject_high_fraud():
    assert make_decision(0.10, 0.90) == "Reject"

def test_reject_high_default():
    assert make_decision(0.80, 0.10) == "Reject"

def test_review_mid_risk():
    assert make_decision(0.30, 0.20) == "Review"

def test_approve_low_risk():
    assert make_decision(0.10, 0.10) == "Approve"