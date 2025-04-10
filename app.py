
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Walmart Sales Forecast", layout="wide")
st.title("🛍️ Walmart Sales Forecasting App")

uploaded_file = st.file_uploader("📁 Upload your CSV file (with 'Date' and 'Weekly_Sales')", type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Weekly_Sales']]
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    weekly_sales = df['Weekly_Sales'].resample('W').sum()

    st.subheader("📊 Preview of Data")
    st.dataframe(df.head())
    st.write(f"🧾 Total weekly records: {len(weekly_sales)}")

    st.subheader("📈 Moving Averages")
    fig_ma, ax_ma = plt.subplots(figsize=(12, 5))
    ax_ma.plot(weekly_sales, label='Actual Sales', color='gray')
    ax_ma.plot(weekly_sales.rolling(window=4).mean(), label='4-week MA', color='blue')
    ax_ma.plot(weekly_sales.rolling(window=12).mean(), label='12-week MA', color='green')
    ax_ma.set_title("Weekly Sales with Moving Averages")
    ax_ma.legend()
    st.pyplot(fig_ma)

    st.subheader("🧠 Time Series Decomposition")
    model_type = st.radio("Choose decomposition model", ["Additive", "Multiplicative"])
    if len(weekly_sales) >= 52:
        decomp = seasonal_decompose(weekly_sales, model=model_type.lower())
        st.line_chart(decomp.trend)
        st.line_chart(decomp.seasonal)
        st.line_chart(decomp.resid)
    else:
        st.warning("Need at least 1 year (52 weeks) of data for decomposition.")

    st.subheader("📉 Forecasting")
    model_choice = st.selectbox("Select Forecasting Model", ["ARIMA", "SARIMA", "ETS", "Prophet", "LSTM"])

    split_idx = int(len(weekly_sales) * 0.8)
    train = weekly_sales[:split_idx]
    test = weekly_sales[split_idx:]
    forecast = None

    if model_choice == "ARIMA":
        model = ARIMA(train, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=len(test))

    elif model_choice == "SARIMA":
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=len(test))

    elif model_choice == "ETS":
        if len(train) >= 52:
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=52)
            fit = model.fit()
            forecast = fit.forecast(len(test))
        else:
            st.error("ETS requires at least 1 year of training data.")

    elif model_choice == "Prophet":
        df_prophet = weekly_sales.reset_index().rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
        train_prophet = df_prophet[:split_idx]
        model = Prophet()
        model.fit(train_prophet)
        future = model.make_future_dataframe(periods=len(test), freq='W')
        forecast_df = model.predict(future)
        forecast = forecast_df.set_index('ds').loc[test.index]['yhat']
        fig2 = model.plot(forecast_df)
        st.pyplot(fig2)

    elif model_choice == "LSTM":
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(weekly_sales.values.reshape(-1, 1))

        def create_sequences(data, steps=4):
            X, y = [], []
            for i in range(len(data) - steps):
                X.append(data[i:i + steps])
                y.append(data[i + steps])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        y_pred_scaled = model.predict(X_test)
        forecast = scaler.inverse_transform(y_pred_scaled).flatten()
        test = scaler.inverse_transform(y_test).flatten()

    if forecast is not None:
        st.subheader("📊 Forecast vs Actual")
        forecast_index = test.index if model_choice != "LSTM" else weekly_sales[-len(forecast):].index
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train, label='Train')
        ax.plot(forecast_index, test[:len(forecast)], label='Actual')
        ax.plot(forecast_index, forecast, label=f'{model_choice} Forecast', linestyle='--')
        ax.set_title(f'{model_choice} Forecast vs Actual')
        ax.legend()
        st.pyplot(fig)

        st.subheader("📏 Evaluation Metrics")
        rmse = np.sqrt(mean_squared_error(test[:len(forecast)], forecast))
        mae = mean_absolute_error(test[:len(forecast)], forecast)
        mape = np.mean(np.abs((test[:len(forecast)] - forecast) / test[:len(forecast)])) * 100
        mse = mean_squared_error(test[:len(forecast)], forecast)
        st.metric("RMSE", f"{rmse:,.2f}")
        st.metric("MAE", f"{mae:,.2f}")
        st.metric("MAPE", f"{mape:.2f}%")
        st.metric("MSE", f"{mse:,.2f}")

    st.subheader("🔮 Forecast for Next Quarter (13 Weeks)")
    forecast_steps = 13
    if model_choice == "ARIMA":
        full_model = ARIMA(weekly_sales, order=(1, 1, 1)).fit()
        future_forecast = full_model.forecast(steps=forecast_steps)
    elif model_choice == "SARIMA":
        full_model = SARIMAX(weekly_sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit()
        future_forecast = full_model.forecast(steps=forecast_steps)
    elif model_choice == "ETS":
        full_model = ExponentialSmoothing(weekly_sales, trend='add', seasonal='add', seasonal_periods=52).fit()
        future_forecast = full_model.forecast(forecast_steps)
    elif model_choice == "Prophet":
        df_prophet_full = weekly_sales.reset_index().rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
        full_model = Prophet()
        full_model.fit(df_prophet_full)
        future_dates_df = full_model.make_future_dataframe(periods=forecast_steps, freq='W')
        future_pred = full_model.predict(future_dates_df)
        future_forecast = future_pred.tail(forecast_steps)['yhat'].values
    elif model_choice == "LSTM":
        last_seq = scaled_data[-4:].reshape(1, 4, 1)
        lstm_preds = []
        for _ in range(forecast_steps):
            next_pred = model.predict(last_seq)[0][0]
            lstm_preds.append(next_pred)
            last_seq = np.append(last_seq[:, 1:, :], [[[next_pred]]], axis=1)
        future_forecast = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=weekly_sales.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W')
    fig_next, ax_next = plt.subplots(figsize=(12, 6))
    ax_next.plot(weekly_sales, label='Historical Sales')
    ax_next.plot(future_dates, future_forecast, label='Next Quarter Forecast', linestyle='--')
    ax_next.set_title("Next 13-Week Forecast")
    ax_next.legend()
    st.pyplot(fig_next)
