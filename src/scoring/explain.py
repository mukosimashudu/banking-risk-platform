from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

credit_model = joblib.load(MODEL_DIR / "credit_model_best.pkl")


def explain_credit(df: pd.DataFrame):
    """
    Returns a standardized SHAP output dictionary.
    Supports sklearn tree models and generic explainer fallback.
    """
    try:
        import shap

        explainer = shap.Explainer(credit_model)
        explanation = explainer(df)

        return {
            "ok": True,
            "explanation": explanation,
            "explainer_type": type(explainer).__name__,
            "error": None,
        }

    except Exception as e1:
        try:
            import shap

            explainer = shap.TreeExplainer(credit_model)
            shap_values = explainer.shap_values(df)

            return {
                "ok": True,
                "explanation": shap_values,
                "explainer_type": type(explainer).__name__,
                "error": None,
            }

        except Exception as e2:
            return {
                "ok": False,
                "explanation": None,
                "explainer_type": None,
                "error": f"Primary error: {e1} | Fallback error: {e2}",
            }