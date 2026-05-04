from pydantic import BaseModel


class FraudInvestigationRequest(BaseModel):
    application_reference: str