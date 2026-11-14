
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Stock Market Trend Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Trend Prediction")
st.write("Machine Learning-Based Stock Trend Prediction and Comparative Analysis")

st.info(
    "Educational project: Predictions are experimental "
    "and are not financial advice."
)

# Features used during model training
features = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "MA_5", "MA_10", "MA_20",
    "Volatility", "RSI", "MACD", "MACD_Signal"
]

# Available models
model_names = {
    "Logistic Regression": "Logistic_Regression.joblib",
    "Decision Tree": "Decision_Tree.joblib",
    "Random Forest": "Random_Forest.joblib",
    "SVM": "SVM.joblib",
    "KNN": "KNN.joblib"
}

# Load dataset
try:
    data = pd.read_csv("reliance_cleaned_data.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()


# Calculate indicators from historical data
def calculate_features(df):

    df = df.copy()

    df["Daily_Return"] = df["Close"].pct_change()

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    df["Volatility"] = (
        df["Daily_Return"].rolling(10).std()
    )

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100

    df.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI"
    ] = 50

    ema12 = df["Close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    return df


# Calculate latest features
featured_data = calculate_features(data)
featured_data = featured_data.dropna()

if featured_data.empty:
    st.error("Not enough historical data to calculate indicators.")
    st.stop()

latest = featured_data.iloc[-1]

# Display stock information
st.subheader("Latest Stock Information")

c1, c2, c3 = st.columns(3)

c1.metric("Latest Close Price", f"₹{latest['Close']:.2f}")
c2.metric("RSI", f"{latest['RSI']:.2f}")
c3.metric("MACD", f"{latest['MACD']:.2f}")

st.caption(
    f"Latest available data date: "
    f"{latest['Date'].strftime('%d %B %Y')}"
)

st.divider()

# Model selection
st.subheader("Make a Trend Prediction")

selected_model = st.selectbox(
    "Select Machine Learning Algorithm",
    list(model_names.keys())
)

if st.button("Predict Stock Trend", type="primary"):

    model_path = os.path.join(
        "saved_models",
        model_names[selected_model]
    )

    if not os.path.exists(model_path):
        st.error("Selected model file not found.")
        st.stop()

    try:
        model = joblib.load(model_path)

        input_data = pd.DataFrame(
            [[latest[feature] for feature in features]],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("📈 Predicted Trend: UP")
            st.write(
                "The model predicts that the next "
                "trading day's closing price may increase."
            )
        else:
            st.error("📉 Predicted Trend: DOWN")
            st.write(
                "The model predicts that the next "
                "trading day's closing price may decrease "
                "or remain unchanged."
            )

        st.caption(
            f"Selected algorithm: {selected_model}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

st.subheader("Historical Closing Price")

st.line_chart(
    data.set_index("Date")["Close"]
)

st.caption(
    "This application uses the latest available row "
    "in the downloaded dataset. It does not fetch live prices."
)
