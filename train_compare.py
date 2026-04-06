import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("placement_data.csv")

X = df.drop("placed", axis=1)
y = df["placed"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------- SCALING (ONLY FOR LOGISTIC REGRESSION) ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- LOGISTIC REGRESSION ----------------
lr_model = LogisticRegression(penalty="l2", solver="liblinear")
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

# ---------------- DECISION TREE ----------------
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

# ---------------- EVALUATION FUNCTION ----------------
def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

lr_metrics = evaluate(y_test, lr_pred, lr_prob)
dt_metrics = evaluate(y_test, dt_pred, dt_prob)

# ---------------- RESULTS TABLE ----------------
results = pd.DataFrame([lr_metrics, dt_metrics],
                        index=["Logistic Regression", "Decision Tree"])

print("\n📊 MODEL COMPARISON RESULTS\n")
print(results)

# ---------------- SAVE MODELS ----------------
joblib.dump(lr_model, "placement_lr_model.pkl")
joblib.dump(dt_model, "placement_dt_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Models and scaler saved successfully")
