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
st.write("Predict mobile price range using Logistic Regression, KNN, SVM, and Naive Bayes models.")

# 📂 Load Training Data (not uploaded by user)
train_df = pd.read_csv("train.csv")
st.subheader("🔍 Training Data Preview")
st.dataframe(train_df.head())

# ✅ Prepare Training Data
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

# 📊 Train & Evaluate Models
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    accuracies[name] = accuracy_score(y_val, preds)

# 📈 Show Accuracy Comparison
st.subheader("📊 Model Accuracy Comparison")
acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
fig = px.bar(acc_df, x="Model", y="Accuracy", text="Accuracy", color="Model", title="Model Accuracy Comparison")
st.plotly_chart(fig)

# 🏆 Choose Best Model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
st.success(f"🏆 Best Model: **{best_model_name}** (Accuracy: {accuracies[best_model_name]:.2f})")

# 🔘 Choose Input Method
st.subheader("🧠 Choose Input Method")
input_method = st.radio("Select how you want to provide test data:",
                        ("📋 Manually Enter Data", "📁 Upload Test CSV File"))

# ✋ Manual Input Form
def manual_input():
    st.subheader("🧍‍♂️ Enter Mobile Specifications")
    data = {
        'battery_power': st.number_input("🔋 Battery Power (mAh)", 500, 2000, 1000),
        'blue': st.selectbox("🔵 Bluetooth Available", [0, 1]),
        'clock_speed': st.number_input("⏰ Clock Speed (GHz)", 0.5, 3.5, 2.0),
        'dual_sim': st.selectbox("📱 Dual SIM", [0, 1]),
        'fc': st.number_input("🤳 Front Camera (MP)", 0, 20, 5),
        'four_g': st.selectbox("🌐 4G Supported", [0, 1]),
        'int_memory': st.number_input("💾 Internal Memory (GB)", 2, 128, 32),
        'm_dep': st.number_input("📏 Mobile Depth (cm)", 0.1, 1.0, 0.5),
        'mobile_wt': st.number_input("⚖️ Mobile Weight (grams)", 80, 250, 150),
        'n_cores': st.slider("🧠 Number of Cores", 1, 8, 4),
        'pc': st.number_input("📸 Primary Camera (MP)", 0, 20, 12),
        'px_height': st.number_input("📐 Pixel Height", 0, 2000, 800),
        'px_width': st.number_input("📏 Pixel Width", 0, 2000, 1200),
        'ram': st.number_input("💽 RAM (MB)", 256, 8000, 4000),
        'sc_h': st.number_input("📱 Screen Height (cm)", 5, 20, 15),
        'sc_w': st.number_input("📱 Screen Width (cm)", 0, 20, 8),
        'talk_time': st.number_input("☎️ Talk Time (hrs)", 2, 24, 10),
        'three_g': st.selectbox("📶 3G Supported", [0, 1]),
        'touch_screen': st.selectbox("🖐️ Touch Screen", [0, 1]),
        'wifi': st.selectbox("📡 WiFi Supported", [0, 1])
    }
    return pd.DataFrame([data])

# 📤 Test File Upload Method
def file_upload():
    st.subheader("📁 Upload Test Data File")
    test_file = st.file_uploader("Upload your test.csv file", type=["csv"])
    if test_file is not None:
        test_df = pd.read_csv(test_file)
        st.write("### ✅ Test Data Preview")
        st.dataframe(test_df.head())

        # Match columns
        missing_cols = [col for col in X.columns if col not in test_df.columns]
        extra_cols = [col for col in test_df.columns if col not in X.columns]

        if extra_cols:
            st.warning(f"Dropping extra columns: {extra_cols}")
            test_df = test_df.drop(columns=extra_cols)

        for col in missing_cols:
            st.warning(f"Adding missing column with default 0: {col}")
            test_df[col] = 0

        test_df = test_df[X.columns]
        return test_df
    else:
        st.info("Please upload a CSV file to continue.")
        return None

# ⚙️ Execute Based on Selection
if input_method == "📋 Manually Enter Data":
    input_df = manual_input()
    if st.button("🔮 Predict Price Range"):
        input_scaled = scaler.transform(input_df)
        prediction = best_model.predict(input_scaled)[0]
        st.success(f"📱 Predicted Price Range: **{prediction}**")

elif input_method == "📁 Upload Test CSV File":
    test_df = file_upload()
    if test_df is not None:
        test_scaled = scaler.transform(test_df)
        predictions = best_model.predict(test_scaled)
        test_df["predicted_price_range"] = predictions

        st.subheader("📄 Predicted Test Data")
        st.dataframe(test_df.head())

        csv = test_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📩 Download Predicted Test Data CSV",
            data=csv,
            file_name="predicted_test_data.csv",
            mime="text/csv"
        )
