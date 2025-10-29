import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import plotly.express as px

# 🏷️ App Title
st.title("📱 Mobile Price Range Prediction App")
st.write("This app predicts mobile phone price ranges using multiple ML models.")

# 📂 Load Training Data
train_df = pd.read_csv("train.csv")

st.subheader("🔍 Training Data Preview")
st.dataframe(train_df.head())

# ✅ Data preparation
X = train_df.drop("price_range", axis=1)
y = train_df["price_range"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🧠 Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

# 📊 Train & Evaluate
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    accuracies[name] = accuracy_score(y_val, preds)

# 📈 Display Accuracies
st.subheader("📊 Model Accuracy Comparison")
acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
fig = px.bar(acc_df, x="Model", y="Accuracy", text="Accuracy", color="Model", title="Model Accuracy Comparison")
st.plotly_chart(fig)

# 🏆 Select best model automatically
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

st.success(f"🏆 Best Model: **{best_model_name}** with Accuracy: {accuracies[best_model_name]:.2f}")

# 📤 Upload Test File
st.subheader("📁 Upload Test Data")
test_file = st.file_uploader("Upload your test.csv file", type=["csv"])

if test_file is not None:
    test_df = pd.read_csv(test_file)
    st.write("### ✅ Test Data Preview")
    st.dataframe(test_df.head())

    # 🔮 Predict using best model
    test_scaled = scaler.transform(test_df)
    predictions = best_model.predict(test_scaled)
    test_df["predicted_price_range"] = predictions

    st.subheader("📄 Predicted Test Data")
    st.dataframe(test_df.head())

    # 📥 Download predictions
    csv = test_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📩 Download Predicted Test Data CSV",
        data=csv,
        file_name="predicted_test_data.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload your test.csv file to see predictions.")
