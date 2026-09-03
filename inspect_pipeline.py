import joblib

pipeline = joblib.load("models/pipeline.pkl")

print(type(pipeline))
print()

print(pipeline.keys())

print()

for k, v in pipeline.items():
    print(k, type(v))