import shap
import pandas as pd

def explain_credit(df, model=None):
    """
    Generate SHAP explanation for credit model
    """

    try:
        if model is None:
            return []

        explainer = shap.Explainer(model)
        shap_values = explainer(df)

        values = shap_values.values[0]

        result = pd.DataFrame({
            "feature": df.columns,
            "impact": values
        })

        result["abs"] = result["impact"].abs()
        result = result.sort_values("abs", ascending=False).head(5)

        return result[["feature", "impact"]].to_dict(orient="records")

    except Exception as e:
        return [{"error": str(e)}]