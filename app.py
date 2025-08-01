import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("🎓 Student Performance Prediction App")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("student_mat.csv")
    return data

df = load_data()

st.subheader("📊 Raw Data")
st.dataframe(df.head())

# Prepare data
df['average'] = df[['G1', 'G2', 'G3']].mean(axis=1)
df['pass'] = df['average'].apply(lambda x: 1 if x >= 10 else 0)  # threshold

X = df[['G1', 'G2', 'G3']]
y = df['pass']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"✅ Model Accuracy: {acc:.2f}")

# Charts with Seaborn + Streamlit
st.subheader("📈 Correlation Heatmap")
heatmap = sns.heatmap(df[['G1', 'G2', 'G3', 'average', 'pass']].corr(), annot=True, cmap='coolwarm')
st.pyplot(heatmap.figure)

st.subheader("🧠 Predict Student Performance")
g1 = st.slider("Grade 1 (G1)", 0, 20, 10)
g2 = st.slider("Grade 2 (G2)", 0, 20, 10)
g3 = st.slider("Grade 3 (G3)", 0, 20, 10)

if st.button("Predict"):
    result = model.predict([[g1, g2, g3]])
    if result[0] == 1:
        st.success("🎉 The student is likely to PASS.")
    else:
        st.error("❌ The student is likely to FAIL.")
