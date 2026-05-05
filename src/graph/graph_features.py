from __future__ import annotations

from typing import Any, Dict, List
from sqlalchemy import text

from src.config.db import engine


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def get_recent_applications(limit: int = 500) -> List[Dict[str, Any]]:
    if engine is None:
        return []

    sql = """
    SELECT TOP (:limit)
        application_reference,
        customer_name,
        decision_type,
        product_type,
        requested_amount,
        approved_amount,
        approved_limit,
        credit_score,
        fraud_score,
        risk_probability,
        final_decision,
        created_at
    FROM ml.prediction_log
    ORDER BY created_at DESC
    """

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql), {"limit": limit}).mappings().all()
        return [dict(row) for row in rows]
    except Exception:
        return []


def build_synthetic_network_keys(row: Dict[str, Any]) -> Dict[str, str]:
    """
    Creates realistic graph identifiers from existing data.
    Later, you can replace these with real device_id, ip_address, merchant_id.
    """
    customer = str(row.get("customer_name") or "unknown_customer").strip().lower().replace(" ", "_")
    product = str(row.get("product_type") or "unknown_product").strip().lower()
    decision = str(row.get("final_decision") or "unknown_decision").strip().lower()
    credit_band = int(safe_float(row.get("credit_score"), 0) // 50) * 50
    fraud_band = int(safe_float(row.get("fraud_score"), 0) * 10)

    return {
        "customer_node": f"customer:{customer}",
        "product_node": f"product:{product}",
        "decision_node": f"decision:{decision}",
        "credit_band_node": f"credit_band:{credit_band}",
        "fraud_band_node": f"fraud_band:{fraud_band}",
    }


def get_application_graph_risk(application_reference: str) -> Dict[str, Any]:
    rows = get_recent_applications(limit=500)

    if not rows:
        return {
            "application_reference": application_reference,
            "graph_risk_score": 0.0,
            "graph_alert_level": "Unknown",
            "connected_applications": 0,
            "shared_risk_nodes": [],
            "message": "No graph data available.",
        }

    target = None
    for row in rows:
        if str(row.get("application_reference")) == str(application_reference):
            target = row
            break

    if not target:
        return {
            "error": "Application reference not found in recent graph window.",
            "application_reference": application_reference,
        }

    target_keys = build_synthetic_network_keys(target)
    target_nodes = set(target_keys.values())

    connected_rows = []
    shared_nodes = {}

    for row in rows:
        if row.get("application_reference") == application_reference:
            continue

        row_keys = build_synthetic_network_keys(row)
        row_nodes = set(row_keys.values())
        overlap = target_nodes.intersection(row_nodes)

        if overlap:
            connected_rows.append(row)
            for node in overlap:
                shared_nodes[node] = shared_nodes.get(node, 0) + 1

    connected_count = len(connected_rows)

    high_risk_connections = 0
    declined_connections = 0

    for row in connected_rows:
        fraud_score = safe_float(row.get("fraud_score"))
        risk_probability = safe_float(row.get("risk_probability"))
        decision = str(row.get("final_decision") or "").upper()

        if fraud_score >= 0.75 or risk_probability >= 0.60:
            high_risk_connections += 1

        if decision == "DECLINED":
            declined_connections += 1

    graph_risk_score = min(
        1.0,
        round(
            0.02 * connected_count
            + 0.08 * high_risk_connections
            + 0.05 * declined_connections,
            4,
        ),
    )

    if graph_risk_score >= 0.75:
        alert_level = "Critical"
    elif graph_risk_score >= 0.50:
        alert_level = "High"
    elif graph_risk_score >= 0.25:
        alert_level = "Medium"
    else:
        alert_level = "Low"

    top_shared_nodes = [
        {"node": node, "connections": count}
        for node, count in sorted(shared_nodes.items(), key=lambda x: x[1], reverse=True)[:10]
    ]

    return {
        "application_reference": application_reference,
        "customer_name": target.get("customer_name"),
        "graph_risk_score": graph_risk_score,
        "graph_alert_level": alert_level,
        "connected_applications": connected_count,
        "high_risk_connections": high_risk_connections,
        "declined_connections": declined_connections,
        "shared_risk_nodes": top_shared_nodes,
        "interpretation": (
            f"This application is connected to {connected_count} recent application(s) "
            f"through shared graph attributes. {high_risk_connections} connected case(s) "
            f"show elevated fraud or credit risk."
        ),
    }