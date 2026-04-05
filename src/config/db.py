import os
from openai import OpenAI

def generate_explanation(data: dict) -> str:
    try:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            return "AI explanation unavailable (no API key configured)."

        client = OpenAI(api_key=api_key)

        prompt = f"""
        Explain this credit decision in simple business terms:

        Credit Score: {data.get("credit_score")}
        Income: {data.get("income")}
        Debt: {data.get("debt")}
        Decision: {data.get("decision")}
        Risk: {data.get("risk")}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return "Customer decision based on affordability, credit score and risk thresholds."