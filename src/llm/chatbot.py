from sqlalchemy import text
from src.config.db import engine


def query_db(question: str):
    q = question.lower().strip()

    if "how many approved" in q:
        sql = "SELECT COUNT(*) AS total FROM ml.prediction_log WHERE decision = 'APPROVE'"
    elif "how many rejected" in q:
        sql = "SELECT COUNT(*) AS total FROM ml.prediction_log WHERE decision LIKE 'REJECT%'"
    elif "how many qualify" in q:
        sql = "SELECT COUNT(*) AS total FROM ml.prediction_log WHERE decision = 'APPROVE'"
    elif "total applications" in q or "how many applications" in q:
        sql = "SELECT COUNT(*) AS total FROM ml.prediction_log"
    elif "average fraud" in q:
        sql = "SELECT AVG(fraud_score) AS avg_fraud_score FROM ml.prediction_log"
    elif "average credit" in q or "average default" in q:
        sql = "SELECT AVG(probability_default) AS avg_probability_default FROM ml.prediction_log"
    else:
        return "I understand questions about approved, rejected, qualified, total applications, average fraud score, and average default probability."

    with engine.connect() as conn:
        result = conn.execute(text(sql)).mappings().first()

    return dict(result) if result else {}