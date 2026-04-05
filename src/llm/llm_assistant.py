from openai import OpenAI
import os

client = None

try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
except:
    client = None


def generate_explanation(data: dict) -> str:
    # ✅ fallback if no API key
    if client is None:
        return f"""
Decision: {data.get('decision')}
Risk: {round(data.get('risk', 0), 2)}
Credit Score: {data.get('credit_score')}
Reason: Based on affordability, credit profile, and risk level.
"""

    try:
        prompt = f"""
        Explain this credit decision clearly:

        Credit Score: {data.get('credit_score')}
        Income: {data.get('income')}
        Debt: {data.get('debt')}
        Risk: {data.get('risk')}
        Decision: {data.get('decision')}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM error fallback: {str(e)}"