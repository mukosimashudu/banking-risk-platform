from fastapi import APIRouter
from sqlalchemy import text

from api.schemas.chat_schema import ChatQuestion
from src.config.db import engine


router = APIRouter(prefix="/api/chat", tags=["SQL Chatbot"])


@router.post("/query")
def chat_query(payload: ChatQuestion):
    if engine is None:
        return {"answer": "Database engine is not configured."}

    question = payload.question.strip().lower()

    try:
        with engine.connect() as conn:
            if "how many rejected" in question or "how many declined" in question:
                row = conn.execute(
                    text("""
                    SELECT COUNT(*) AS cnt
                    FROM ml.prediction_log
                    WHERE final_decision = 'DECLINED'
                    """)
                ).fetchone()

                return {"answer": f"There are {row[0]} declined applications."}

            if "how many approved" in question:
                row = conn.execute(
                    text("""
                    SELECT COUNT(*) AS cnt
                    FROM ml.prediction_log
                    WHERE final_decision = 'APPROVED'
                    """)
                ).fetchone()

                return {"answer": f"There are {row[0]} approved applications."}

            if "how many review" in question:
                row = conn.execute(
                    text("""
                    SELECT COUNT(*) AS cnt
                    FROM ml.prediction_log
                    WHERE final_decision = 'REVIEW'
                    """)
                ).fetchone()

                return {"answer": f"There are {row[0]} applications in review."}

            summary = conn.execute(
                text("SELECT * FROM analytics.v_portfolio_summary")
            ).mappings().first()

            if not summary:
                return {"answer": "No portfolio data is available yet."}

            return {
                "answer": (
                    f"Portfolio summary: total applications {summary['total_applications']}, "
                    f"approved {summary['approved_cases']}, "
                    f"declined {summary['declined_cases']}, "
                    f"review {summary['review_cases']}, "
                    f"approval rate {summary['approval_rate_pct']}%."
                )
            }

    except Exception as exc:
        return {"answer": f"Chat query failed: {str(exc)}"}