def investigation_prompt(data: dict) -> str:
    return f"""
    Investigate this financial application.

    Customer: {data.get('customer_name')}
    Credit Score: {data.get('credit_score')}
    Fraud Score: {data.get('fraud_score')}
    Risk Probability: {data.get('risk_probability')}
    Decision: {data.get('final_decision')}

    Provide:
    - Summary
    - Risk explanation
    - Recommended action
    """