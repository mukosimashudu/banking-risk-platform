from src.scoring.decision_engine import make_decision

fraud_features = [100, 5000, 200, 150, 300, 100, 50]
credit_features = [0.2, 35, 0, 0.3, 5000, 5, 0, 1, 0, 0]

result = make_decision(fraud_features, credit_features)

print(result)