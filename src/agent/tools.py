from typing import Dict, Any
from sqlalchemy import text
from src.config.db import engine


def fetch_application(application_reference: str) -> Dict[str, Any]:
    if engine is None:
        return {}

    sql = """
    SELECT TOP 1 *
    FROM ml.prediction_log
    WHERE application_reference = :ref
    ORDER BY created_at DESC
    """

    try:
        with engine.connect() as conn:
            row = conn.execute(text(sql), {"ref": application_reference}).mappings().first()
        return dict(row) if row else {}
    except Exception:
        return {}