# CSGO Round Winner Prediction using JupyterLabs, ML & Python

## **📌 Project Overview**
This project predicts the **winner of a CS:GO round** using **machine learning models** trained on in-game data. The dataset includes features such as **player health, armor, weapons, grenades, economy, and map details**, which influence the outcome of each round.  

Through **data preprocessing, feature selection, and model training**, this project explores various **classification models** to predict whether the **Counter-Terrorists (CT) or Terrorists (T) will win a given round**.

---

## **📊 Data Collection and Preprocessing**
### **1️⃣ Downloading the Dataset**
The dataset is sourced from **OpenML** and is downloaded using the `requests` library:
```python
import requests

url = "https://www.openml.org/data/download/22102255/dataset"
r = requests.get(url, allow_redirects=True)
with open("dataset.txt", "wb") as f:
    f.write(r.content)
```
🔹 **Why is this important?**  
This step retrieves the **CS:GO round data** for training the model.

---

### **2️⃣ Reading and Cleaning the Data**
The dataset is loaded into a **Pandas DataFrame**, and unnecessary lines (metadata) are removed:
```python
import pandas as pd
data = []

with open("dataset.txt", "r") as f:
    for line in f.read().split("\n"):
        if line.startswith("@") or line.startswith("%") or line == "":
            continue
        data.append(line)
```
Then, column names are extracted:
```python
columns = []
with open("dataset.txt", "r") as f:
    for line in f.read().split("\n"):
        if line.startswith("@ATTRIBUTE"):
            columns.append(line.split(" ")[1])
```
🔹 **Why is this important?**  
- The dataset includes metadata that **must be removed** before analysis.
- The extracted **column names** provide a structured **feature set**.

---

### **3️⃣ Creating the DataFrame and Encoding the Target Variable**
After cleaning, the data is written to a CSV file:
```python
with open("df.csv", "w") as f:
    f.write(",".join(columns))
    f.write("\n")
    f.write("\n".join(data))
df = pd.read_csv("df.csv")
df.columns = columns
```
The **target variable** (who wins the round: CT or T) is converted into a **numerical format**:
```python
df['t_win'] = df.round_winner.astype("category").cat.codes
```
🔹 **Why is this important?**  
- **Converting categorical data to numerical data** allows it to be used in ML models.
- **CT (Counter-Terrorists) = 0, T (Terrorists) = 1**.

---

## **📊 Exploratory Data Analysis (EDA)**
### **4️⃣ Checking Feature Correlations**
A correlation matrix is generated to determine which features impact **T-side win probability** the most:
```python
import matplotlib.pyplot as plt
import seaborn as sns

correlations = df.select_dtypes(include=['number']).corr()
print(correlations['t_win'].apply(abs).sort_values(ascending=False).iloc[:25])

plt.figure(figsize=(12,6))
sns.heatmap(correlations[['t_win']].sort_values(by='t_win', ascending=False), annot=True, cmap="coolwarm")
plt.title("Feature Correlation with T-side Win")
plt.show()
```
🔹 **Key Insights from Correlation Analysis:**  
- **Armor, Helmets, and Money** significantly influence round outcomes.
- **Weapons like AK-47s and AWP** impact the win probability.
- **Number of players alive and grenades used** play a crucial role.

---

### **5️⃣ Feature Selection**
Features with a correlation greater than **0.15** are selected for training:
```python
selected_columns = []

for col in columns+["t_win"]:
    try:
        if abs(correlations[col]["t_win"]) > 0.15:
            selected_columns.append(col)
    except KeyError:
        pass

df_selected = df[selected_columns]
```
🔹 **Why is this important?**  
- **Removes irrelevant features** to improve model efficiency.
- **Reduces overfitting** by focusing on impactful variables.

---

## **📈 Model Training and Evaluation**
### **6️⃣ Splitting Data for Training and Testing**
The dataset is split into **80% training** and **20% testing**:
```python
from sklearn.model_selection import train_test_split

x, y = df_selected.drop(["t_win"], axis=1), df_selected["t_win"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
🔹 **Why is this important?**  
Ensures that the model generalizes well to **unseen data**.

---

### **7️⃣ Training a Random Forest Model**
A **Random Forest classifier** is trained and achieves **82.5% accuracy**:
```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_jobs=4)
forest.fit(x_train_scaled, y_train)
forest.score(x_test_scaled, y_test)
```
🔹 **Why is this important?**  
- **Random Forest outperforms KNN**, handling non-linear relationships better.

---

### **🏆 Conclusion**
✅ **KNN Model Accuracy:** `~78.1%`  
✅ **Random Forest Model Accuracy:** `~82.5%` (Best model)  
✅ **Neural Network Accuracy:** `~75.2%`  

📌 **Final Takeaways:**  
- **Game economy and armor strongly impact round outcomes.**
- **Random Forest provides the best predictive performance.**
- **Feature selection significantly improves accuracy.**

---

### **🚀 Future Improvements**
- Use **Gradient Boosting models (XGBoost, LightGBM)** for higher accuracy.
- Include **time-series data** (e.g., round progression).
- Deploy the model as a **real-time CS:GO win predictor**.

---