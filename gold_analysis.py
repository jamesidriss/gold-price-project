import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 1) Lire le fichier
df = pd.read_csv("gold_historical_data.csv")

# 2) Transformer la date
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# 3) Créer une colonne:
# Est-ce que demain le prix monte ? (1 = oui, 0 = non)
df["Tomorrow"] = df["Close"].shift(-1)
df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

# 4) On utilise les prix passés pour prédire
df["Lag1"] = df["Close"].shift(1)
df["Lag2"] = df["Close"].shift(2)
df["Lag3"] = df["Close"].shift(3)

df = df.dropna()

# 5) ML simple
features = ["Lag1", "Lag2", "Lag3"]
X = df[features]
y = df["Target"]

split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# 6) Graphique
plt.plot(df["Date"], df["Close"])
plt.title("Gold Price")
plt.show()

# 7) Export pour Tableau
df.to_csv("gold_for_tableau.csv", index=False)
