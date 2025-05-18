import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# ----------- تحميل البيانات ------------
df = pd.read_csv("weather_classification_data.csv")

# ----------- عرض الخصائص الإحصائية للبيانات ------------
print("\n=== Statistical Summary ===")
print(df.describe())

# ----------- Visualize the data before preprocessing ------------
numeric_cols = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
                'Atmospheric Pressure', 'UV Index', 'Visibility (km)']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    sns.violinplot(ax=axes[idx], x='Weather Type', y=col, data=df, inner='quartile')
    axes[idx].set_title(f'Distribution of {col} by Weather Type')

for i in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("weather_violinplots.png")
plt.show()

# ----------- تحديد الهدف والفيتشرز بشكل صحيح ------------
y = df['Weather Type']
X = df.drop(columns=['Weather Type', 'Location'])

# ----------- تحويل الأعمدة النصية إلى أرقام ------------
categorical_cols = ['Season', 'Cloud Cover']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ----------- تقسيم البيانات ------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------- حفظ X و Y و X_test و Y_test ------------
X_train.to_csv("X.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("Y.csv", index=False)
y_test.to_csv("Y_test.csv", index=False)

# ----------- تطبيع البيانات ------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------- النماذج ------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True),
    "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

accuracy_list = []

# ----------- تدريب وتقييم النماذج وحفظ التوقعات ------------
for name, model in models.items():
    print(f"\n==== {name} ====")

    if name in ["Naive Bayes", "KNN", "Decision Tree", "Random Forest"]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    accuracy_list.append((name, acc))

    print(f"Accuracy: {acc:.4f}")
    print(report)

    # حفظ التوقعات
    pd.DataFrame(preds, columns=["Prediction"]).to_csv(f"predictions_{name.replace(' ', '_')}.csv", index=False)

# ----------- رسم دقة النماذج ------------
accuracy_df = pd.DataFrame(accuracy_list, columns=["Model", "Accuracy"])
sns.barplot(data=accuracy_df, x='Model', y='Accuracy', palette='viridis')
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()
