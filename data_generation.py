import numpy as np
import pandas as pd

np.random.seed(42)
n = 1500

data = {
    "cgpa": np.round(np.random.uniform(5.5, 9.8, n), 2),
    "coding_skill": np.random.randint(1, 11, n),
    "dsa_skill": np.random.randint(1, 11, n),
    "projects": np.random.randint(0, 6, n),
    "internship": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    "communication": np.random.randint(1, 11, n),
    "certifications": np.random.randint(0, 6, n),
    "mock_score": np.random.randint(30, 101, n)
}

df = pd.DataFrame(data)

# Logical rule to assign placement label
score = (
    df["cgpa"] * 0.25 +
    df["coding_skill"] * 0.2 +
    df["dsa_skill"] * 0.2 +
    df["projects"] * 0.1 +
    df["internship"] * 0.1 +
    df["communication"] * 0.1 +
    df["mock_score"] * 0.15 / 10
)

df["placed"] = (score > score.mean()).astype(int)

df.to_csv("placement_data.csv", index=False)
print("Dataset generated successfully!")
