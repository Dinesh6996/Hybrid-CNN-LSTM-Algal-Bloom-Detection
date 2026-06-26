import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout, Input

# ===============================
# CREATE OUTPUT FOLDER
# ===============================
os.makedirs("output", exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
X = np.load("X_timeseries.npy")
y = np.load("y_timeseries.npy")

samples, timesteps, features = X.shape
print("Dataset Loaded:", X.shape)

# ===============================
# HEATMAP
# ===============================
X_last = X[:, -1, :]
columns = ["Temp","pH","DO","Turbidity","Nitrate","Phosphate","Chlorophyll"]

df = pd.DataFrame(X_last, columns=columns)
df["Bloom"] = y

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Parameter Correlation Heatmap")
plt.savefig("output/heatmap.png")
plt.show()
plt.close()

# ===============================
# PREPROCESS
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, features))
X = X_scaled.reshape(samples, timesteps, features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# RANDOM FOREST
# ===============================
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train[:, -1, :], y_train)

rf_pred = rf.predict(X_test[:, -1, :])

# ===============================
# CNN
# ===============================
cnn = Sequential([
    Input(shape=(timesteps, features)),
    Conv1D(64,2,activation='relu'),
    Flatten(),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn.fit(X_train,y_train,epochs=5,batch_size=32,verbose=0)

cnn_pred = (cnn.predict(X_test, verbose=0) > 0.5).astype(int)

# ===============================
# LSTM
# ===============================
lstm = Sequential([
    Input(shape=(timesteps, features)),
    LSTM(64),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])

lstm.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
lstm.fit(X_train,y_train,epochs=5,batch_size=32,verbose=0)

lstm_pred = (lstm.predict(X_test, verbose=0) > 0.5).astype(int)

# ===============================
# CNN-LSTM (HYBRID)
# ===============================
cnn_lstm = Sequential([
    Input(shape=(timesteps, features)),
    Conv1D(64,2,activation='relu'),
    Conv1D(128,2,activation='relu'),
    LSTM(128,return_sequences=True),
    LSTM(64),
    Dense(64,activation='relu'),
    Dropout(0.3),
    Dense(1,activation='sigmoid')
])

cnn_lstm.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn_lstm.fit(X_train,y_train,epochs=8,batch_size=32,verbose=0)

hybrid_pred = (cnn_lstm.predict(X_test, verbose=0) > 0.5).astype(int)

# ===============================
# METRICS FUNCTION
# ===============================
def get_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    )

rf_metrics = get_metrics(y_test, rf_pred)
cnn_metrics = get_metrics(y_test, cnn_pred)
lstm_metrics = get_metrics(y_test, lstm_pred)
hybrid_metrics = get_metrics(y_test, hybrid_pred)

# ===============================
# RESULTS TABLE
# ===============================
results = pd.DataFrame({
    "Model": ["Random Forest", "CNN", "LSTM", "CNN-LSTM"],
    "Accuracy": [rf_metrics[0], cnn_metrics[0], lstm_metrics[0], hybrid_metrics[0]],
    "Precision": [rf_metrics[1], cnn_metrics[1], lstm_metrics[1], hybrid_metrics[1]],
    "Recall": [rf_metrics[2], cnn_metrics[2], lstm_metrics[2], hybrid_metrics[2]],
    "F1 Score": [rf_metrics[3], cnn_metrics[3], lstm_metrics[3], hybrid_metrics[3]]
}).round(4)

print("\n===== MODEL PERFORMANCE =====")
print(results)

results.to_csv("output/model_metrics.csv", index=False)

# ===============================
# ACCURACY GRAPH
# ===============================
plt.figure()
plt.bar(results["Model"], results["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.savefig("output/accuracy.png")
plt.show()
plt.close()

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, hybrid_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha='center')
plt.savefig("output/confusion.png")
plt.show()
plt.close()

# ===============================
# ROC CURVE
# ===============================
y_prob = cnn_lstm.predict(X_test, verbose=0)
fpr,tpr,_ = roc_curve(y_test,y_prob)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")
plt.savefig("output/roc.png")
plt.show()
plt.close()

# ===============================
# MANUAL INPUT
# ===============================
print("\nEnter Water Parameters")

temperature = float(input("Temperature: "))
ph = float(input("pH: "))
do = float(input("DO: "))
turbidity = float(input("Turbidity: "))
nitrate = float(input("Nitrate: "))
phosphate = float(input("Phosphate: "))
chlorophyll = float(input("Chlorophyll: "))

sequence = np.array([[temperature,ph,do,turbidity,nitrate,phosphate,chlorophyll]] * timesteps)

sequence_scaled = scaler.transform(sequence)
sequence_scaled = sequence_scaled.reshape(1,timesteps,features)

prob = cnn_lstm.predict(sequence_scaled, verbose=0)[0][0]
model_pred = 1 if prob > 0.5 else 0

rule_pred = 0
if do < 3 or (nitrate > 2.5 and chlorophyll > 30):
    rule_pred = 1

final_pred = max(model_pred, rule_pred)

print("\nPrediction:")
print("Probability:", round(prob,3))
print("Final Output:", final_pred)

if final_pred == 1:
    print("⚠ NOT SAFE")
else:
    print("✔ SAFE")