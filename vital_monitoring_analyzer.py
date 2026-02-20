import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
File_path = "blood_pressure_data.csv"
data = pd.read_csv("blood_pressure_data.csv")
print("Original Data:")
print(data.head())

data = data[
    (data["Systolic_BP"] > 0) &
    (data["Diastolic_BP"] > 0) &
    (data["Heart_Rate"] > 0) &
    (data["SpO2"] > 0) &
    (data["Temperature"] > 0) & 
    (data["Resp_Rate"] > 0)
]

data. fillna(method="ffill", inplace=True)
print("\nCleaned Data:")
print(data.head())

ean_sys = np.mean(data["Systolic_BP"])
mean_hr = np.mean(data["Heart_Rate"])
print("\nAverage Systolic BP:", round(mean_sys), 2)
print("Average Heart Rate:", round(mean_hr, 2))

plt.figure(figsize=(10,5))
plt.plot(data["Time"], data["Systolic_BP"])
plt.plot(data["Time"], data["Diastolic_BP"])
plt.xlabel("Time")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure Monitoring")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["Time"], data["Heart_Rate"])
plt.xlabel("Time")
plt.ylabel("Heart Rate (BPM)")
plt.title("Heart Rate Monitoring")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["Time"], data["SpO2"])
plt.xlabel("Time")
plt.ylabel("SpO2 (%)")
plt.title("Oxygen Level Monitoring")
plt.show
            


# -------- Generate Data --------
n = 60   # 60 rows of data
time_stamps = np.arange(n)

# Blood Pressure (Systolic & Diastolic)
systolic = np.random.normal(120, 12, n)     # Avg 120
diastolic = np.random.normal(80, 8, n)      # Avg 80

# Other vitals
heart_rate = np.random.normal(80, 8, n)
spo2 = np.random.normal(97, 1.5, n)
temperature = np.random.normal(36.3, 0.3, n)
resp_rate = np.random.normal(16, 2, n)

# Round values
systolic = np.round(systolic, 1)
diastolic = np.round(diastolic, 1)
heart_rate = np.round(heart_rate, 1)
spo2 = np.round(spo2, 1)
temperature = np.round(temperature, 1)
resp_rate = np.round(resp_rate, 1)


# -------- Statistics Function --------
def stats(arr):
    return {
        "mean": np.round(np.mean(arr), 2),
        "min": np.round(np.min(arr), 2),
        "max": np.round(np.max(arr), 2),
        "std": np.round(np.std(arr), 2)
    }

print("\n--- STATISTICS ---")
print("Systolic BP:", stats(systolic))
print("Diastolic BP:", stats(diastolic))
print("Heart Rate:", stats(heart_rate))
print("SpO2:", stats(spo2))
print("Temperature:", stats(temperature))
print("Resp Rate:", stats(resp_rate))


# -------- Detect Alerts --------
alerts = []

for i in range(n):

    # Blood Pressure Alerts
    if systolic[i] > 160 or diastolic[i] > 100:
        alerts.append((i, "Hypertensive Crisis", systolic[i], diastolic[i]))

    elif systolic[i] > 140 or diastolic[i] > 90:
        alerts.append((i, "High Blood Pressure Stage 2", systolic[i], diastolic[i]))

    elif systolic[i] > 130 or diastolic[i] > 85:
        alerts.append((i, "High Blood Pressure Stage 1", systolic[i], diastolic[i]))

    elif systolic[i] < 90 or diastolic[i] < 60:
        alerts.append((i, "Low Blood Pressure", systolic[i], diastolic[i]))

    # Heart Rate Alerts
    if heart_rate[i] > 120:
        alerts.append((i, "Severe Tachycardia", heart_rate[i], None))
    elif heart_rate[i] > 100:
        alerts.append((i, "Tachycardia", heart_rate[i], None))
    elif heart_rate[i] < 50:
        alerts.append((i, "Bradycardia", heart_rate[i], None))

    # SpO2 Alert
    if spo2[i] < 92:
        alerts.append((i, "Low SpO2", spo2[i], None))

    # Temperature Alert
    if temperature[i] >= 38.0:
        alerts.append((i, "Fever", temperature[i], None))


# -------- Print Alerts --------
print("\n---- ALERTS DETECTED ----")
if len(alerts) == 0:
    print("No alerts found.")
else:
    for a in alerts:
        print(a)


# -------- Create Pandas DataFrame --------
df = pd.DataFrame({
    "Time": time_stamps,
    "Systolic_BP": systolic,
    "Diastolic_BP": diastolic,
    "Heart_Rate": heart_rate,
    "SpO2": spo2,
    "Temperature": temperature,
    "Resp_Rate": resp_rate
})

print("\n--- DATAFRAME CREATED ---")
print(df.head())


# -------- Save Data to CSV --------
save = input("\nDo you want to save the data into a CSV file? (y/n): ")

if save.lower() == "y":
    df.to_csv("blood_pressure_data.csv", index=False)

    alert_df = pd.DataFrame(alerts, columns=["Time", "Alert_Type", "Value1", "Value2"])
    alert_df.to_csv("bp_alerts.csv", index=False)

    print("\nFiles saved successfully:")
    print("✔ blood_pressure_data.csv")
    print("✔ bp_alerts.csv")

else:
    print("Data not saved.")

   # ---------- STEP 2: ML PREDICTION ----------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Create Risk Level column
def bp_risk(sys, dia):
    if sys >= 160 or dia >= 100:
        return "Dangerous"
    elif sys >= 140 or dia >= 90:
        return "High"
    elif sys >= 130 or dia >= 85:
        return "Borderline"
    elif sys >= 90 and dia >= 60:
        return "Normal"
    else:
        return "Low"

df["Risk_Level"] = df.apply(
    lambda x: bp_risk(x["Systolic_BP"], x["Diastolic_BP"]),
    axis=1
)

# Features and target
X = df[["Systolic_BP", "Diastolic_BP", "Heart_Rate", "SpO2", "Temperature"]]
y = df["Risk_Level"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("\nML Model Accuracy:", accuracy_score(y_test, y_pred))
 

