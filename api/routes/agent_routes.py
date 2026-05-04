from fastapi import APIRouter


router = APIRouter(prefix="/api/agent", tags=["Agentic AI"])


@router.get("/status")
def agent_status():
    return {
        "status": "ready",
        "message": "Agentic AI module is ready for expansion.",
        "next_steps": [
            "Read customer/application profile from Azure SQL",
            "Check fraud score and credit risk score",
            "Check graph network risk",
            "Generate investigation summary",
            "Recommend approve, decline, review, or escalate",
        ],
    }