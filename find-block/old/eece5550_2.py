import joblib, csv, numpy as np

mlp = joblib.load("mlp_model.pkl")
sc = joblib.load("scaler.pkl")

def predict_block(rows):
    binned = np.zeros(360)
    counts = np.zeros(360)
    for a, d in rows:
        idx = int(a) % 360
        binned[idx] += d
        counts[idx] += 1
    binned[counts > 0] /= counts[counts > 0]
    binned[counts == 0] = np.mean(binned[counts > 0])
    binned = (binned - binned.mean()) / (binned.std() + 1e-8)
    X = np.array([[binned[(i + j) % 360] for j in range(-15, 16)] for i in range(360)])
    probs = mlp.predict_proba(sc.transform(X))[:, list(mlp.classes_).index(1)]
    return np.argmax(probs)

filepath = "data_4_(315_deg).csv"
with open(filepath) as f:
    rows = [(float(r["Angle"]), float(r["Distance"])) for r in csv.DictReader(f)]
for i in range(10):
    print((rows)[i])

# filepath = "Lidar Data (With Block Against Wall) - lidar_data.csv.csv"
# with open(filepath) as f:
#     rows = [(float(r[0]), float(r[1])) for r in csv.reader(f)]
# for i in range(100):
#     print(predict_block(rows[300*i+60:300*i+360]))
