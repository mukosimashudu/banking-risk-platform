from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IFRS9Result:
    stage: str
    pd_12m: float
    pd_lifetime: float
    lgd: float
    ead: float
    ecl_12m: float
    ecl_lifetime: float
    significant_increase_in_credit_risk: bool
    reason: str


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def derive_pd_from_credit_score(credit_score: int) -> float:
    score = max(300, min(900, int(credit_score)))

    if score >= 750:
        return 0.015
    if score >= 700:
        return 0.030
    if score >= 650:
        return 0.060
    if score >= 600:
        return 0.120
    if score >= 550:
        return 0.200
    return 0.320


def derive_lgd(product_type: str, secured: bool, ltv: float | None = None) -> float:
    pt = (product_type or "").strip().lower()

    if pt == "home_loan":
        if ltv is not None and ltv <= 0.80:
            return 0.20
        if ltv is not None and ltv <= 0.90:
            return 0.28
        return 0.35

    if pt == "vehicle_loan":
        return 0.40

    if pt in {"personal_loan", "credit_card"}:
        return 0.65

    if secured:
        return 0.35

    return 0.55


def assign_stage(
    days_past_due: int,
    significant_increase_in_credit_risk: bool,
    default_flag: bool,
) -> tuple[str, str]:
    if default_flag or days_past_due >= 90:
        return "Stage 3", "Default or 90+ DPD"
    if significant_increase_in_credit_risk or days_past_due >= 30:
        return "Stage 2", "SICR or 30+ DPD"
    return "Stage 1", "Performing exposure"


def ifrs9_engine(
    approved_amount: float,
    credit_score: int,
    product_type: str,
    secured: bool,
    days_past_due: int = 0,
    significant_increase_in_credit_risk: bool = False,
    default_flag: bool = False,
    ltv: float | None = None,
    ccf: float = 1.0,
) -> IFRS9Result:
    ead = max(0.0, float(approved_amount)) * max(0.0, float(ccf))
    pd_12m = derive_pd_from_credit_score(credit_score)
    lgd = derive_lgd(product_type=product_type, secured=secured, ltv=ltv)

    stage, reason = assign_stage(
        days_past_due=days_past_due,
        significant_increase_in_credit_risk=significant_increase_in_credit_risk,
        default_flag=default_flag,
    )

    if stage == "Stage 1":
        pd_lifetime = min(1.0, pd_12m * 2.25)
        ecl_12m = ead * pd_12m * lgd
        ecl_lifetime = ead * pd_lifetime * lgd
    elif stage == "Stage 2":
        pd_lifetime = min(1.0, pd_12m * 4.00)
        ecl_12m = ead * pd_12m * lgd
        ecl_lifetime = ead * pd_lifetime * lgd
    else:
        pd_lifetime = min(1.0, max(pd_12m, 0.65))
        ecl_12m = ead * pd_12m * lgd
        ecl_lifetime = ead * pd_lifetime * lgd

    return IFRS9Result(
        stage=stage,
        pd_12m=round(clamp(pd_12m), 6),
        pd_lifetime=round(clamp(pd_lifetime), 6),
        lgd=round(clamp(lgd), 6),
        ead=round(ead, 2),
        ecl_12m=round(ecl_12m, 2),
        ecl_lifetime=round(ecl_lifetime, 2),
        significant_increase_in_credit_risk=bool(significant_increase_in_credit_risk),
        reason=reason,
    )