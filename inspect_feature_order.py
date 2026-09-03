import joblib

feature_order = joblib.load("models/feature_order.pkl")

print(type(feature_order))

if isinstance(feature_order, dict):
    print(feature_order.keys())

elif isinstance(feature_order, list):
    print(len(feature_order))
    print(feature_order[:20])