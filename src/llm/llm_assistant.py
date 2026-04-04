from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_explanation(context: dict) -> str:
    try:
        prompt = f"""
You are a senior credit risk analyst working at a bank.

Explain the decision in a professional, concise, business-friendly tone.

Customer Profile:
- Credit Score: {context.get("credit_score")}
- Income: {context.get("income")}
- Debt: {context.get("debt")}
- Decision: {context.get("decision")}
- Risk Probability: {context.get("risk")}

Explain:
- Why decision was made
- Risk level
- Financial behaviour insight
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a banking risk expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM unavailable: {str(e)}"