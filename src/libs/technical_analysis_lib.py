from asyncio import as_completed
from sklearn import preprocessing
import pandas as pd
import numpy as np
import talib as talib

BUY = 0  # -1
HOLD = 1  # 0
SELL = 2  # 1


class TechnicalAnalysis:

    @staticmethod
    def compute_oscillators(data):
        log_return = np.log(data["Close"]) - np.log(data["Close"].shift(1))
        data["Z_score"] = (
            log_return - log_return.rolling(20).mean()
        ) / log_return.rolling(20).std()
        data["RSI"] = (talib.RSI(data["Close"])) / 100
        upper_band, _, lower_band = talib.BBANDS(
            data["Close"], nbdevup=2, nbdevdn=2, matype=0
        )
        # MACD calculation
        macd, signal_line, _ = talib.MACD(
            data["Close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        data["MACD"] = macd
        data["MACD_Signal_Line"] = signal_line
        data["boll"] = (data["Close"] - lower_band) / (upper_band - lower_band)
        data["ULTOSC"] = (talib.ULTOSC(data["High"], data["Low"], data["Close"])) / 100
        data["pct_change"] = data["Close"].pct_change()
        # data["zsVol"] = (data["Volume"] - data["Volume"].mean()) / data["Volume"].std()
        data["PR_MA_Ratio_short"] = (
            data["Close"] - talib.SMA(data["Close"], 21)
        ) / talib.SMA(data["Close"], 21)
        data["MA_Ratio_short"] = (
            talib.SMA(data["Close"], 21) - talib.SMA(data["Close"], 50)
        ) / talib.SMA(data["Close"], 50)
        data["MA_Ratio"] = (
            talib.SMA(data["Close"], 50) - talib.SMA(data["Close"], 100)
        ) / talib.SMA(data["Close"], 100)
        data["PR_MA_Ratio"] = (
            data["Close"] - talib.SMA(data["Close"], 50)
        ) / talib.SMA(data["Close"], 50)

        # === Additional Momentum Indicators ===

        # 1. Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            data["High"],
            data["Low"],
            data["Close"],
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )
        data["Stoch_K"] = slowk / 100
        data["Stoch_D"] = slowd / 100

        # 2. Williams %R
        data["WilliamsR"] = talib.WILLR(data["High"], data["Low"], data["Close"]) / -100

        # 3. Commodity Channel Index (CCI)
        data["CCI"] = (
            talib.CCI(data["High"], data["Low"], data["Close"], timeperiod=14) / 100
        )

        # 4. Rate of Change (ROC)
        data["ROC"] = talib.ROC(data["Close"], timeperiod=12) / 100

        # 5. Momentum
        data["MOM"] = talib.MOM(data["Close"], timeperiod=10) / data["Close"]

        # 6. TRIX
        data["TRIX"] = talib.TRIX(data["Close"], timeperiod=15)

        return data

    @staticmethod
    def compute_macro_features(data):
        macro_cols = [
            "USURTOT INDEX",
            "IP_INDEX",
            "GDP_CQOQ_INDEX",
            "RSTAMOM_INDEX",
            "FDTR_INDEX",
            "USGG10YR_INDEX",
            "PPI_YOY_INDEX",
            "CONCCONF_INDEX",
            "USGG5YR_INDEX",
            "USGG2YR_INDEX",
            "CPI_XYOY_INDEX",  # optional
            "FEDL01_INDEX",  # optional
            "NFP TCH INDEX",  # optional
        ]
        """
        macro_cols = [ 
            "CPI_XYOY_INDEX",
            "CPI_YOY_INDEX",
            "PCE_CYOY_INDEX",
            "PCE_YOY_INDEX",
            "PPI_YOY_INDEX",
            "GDP_CQOQ_INDEX",
            "FDTR_INDEX",
            "USURTOT INDEX",
            "IP_INDEX",
            "NAPMNMI_INDEX",
            "NAPMPMI_INDEX",
            "RSTAMOM_INDEX",
            "CONCCONF_INDEX",
            "CONSSENT_INDEX",
            "SBOITOTL_INDEX",
            "INJCJC_INDEX",
            "NFP TCH INDEX",
            "FEDL01_INDEX",
            "USGG2YR_INDEX",
            "USGG5YR_INDEX",
            "USGG10YR_INDEX",
            "USYC2Y10_INDEX",
            "USYC3M10_INDEX",
        ]
        """

        data[macro_cols] = data[macro_cols].ffill()
        epsilon = 1e-6
        lags = [5, 21]

        for col in macro_cols:
            for lag in lags:
                data[f"{col}_lag_{lag}"] = data[col].shift(lag)
                data[f"{col}_mom_{lag}"] = data[col].pct_change(periods=lag)

            rolling_mean = data[col].rolling(63).mean()
            rolling_std = data[col].rolling(63).std()
            data[f"{col}_zscore"] = (data[col] - rolling_mean) / (rolling_std + epsilon)
            data[f"{col}_zscore"] = data[f"{col}_zscore"].clip(-10, 10)

        # === Interaction Terms ===
        data["CPI_Unemployment"] = data["CPI_YOY_INDEX"] * data["USURTOT INDEX"]
        data["GDP_RetailSales"] = data["GDP_CQOQ_INDEX"] * data["RSTAMOM_INDEX"]
        data["Sentiment_Retail"] = data["CONSSENT_INDEX"] * data["RSTAMOM_INDEX"]
        data["YieldSpread_Fed"] = data["USYC2Y10_INDEX"] * data["FDTR_INDEX"]

        # === Combo Terms / Spread Indicators ===
        data["CPI_minus_FedRate"] = data["CPI_YOY_INDEX"] - data["FDTR_INDEX"]
        data["GDP_Unemployment"] = data["GDP_CQOQ_INDEX"] * (1 - data["USURTOT INDEX"])
        data["Yield_Curve_10y_2y"] = data["USYC2Y10_INDEX"]
        data["Yield_Curve_10y_3m"] = data["USYC3M10_INDEX"]
        data["Real_Rate_Proxy"] = data["USGG10YR_INDEX"] - data["CPI_YOY_INDEX"]

        return data

    @staticmethod
    def add_timely_data(data):
        data["DayOfWeek"] = pd.to_datetime(data["Date"]).dt.dayofweek
        data["Month"] = pd.to_datetime(data["Date"]).dt.month
        # data["Hourly"] = pd.to_datetime(data["Date"]).dt.hour / 6
        return data

    @staticmethod
    def assign_labels(data, b_window, f_window, alpha, beta):
        x = data.copy()
        x["Close_MA"] = x["Close"].ewm(span=b_window).mean()
        x["s-1"] = x["Close"].shift(-1 * f_window)
        x["alpha"] = alpha
        x["beta"] = beta * (1 + (f_window * 0.1))
        x["label"] = x.apply(TechnicalAnalysis.check_label, axis=1)
        return x["label"]

    @staticmethod
    def check_label(z):
        if (abs((z["s-1"] - z["Close_MA"]) / z["Close_MA"]) > z["alpha"]) and (
            abs((z["s-1"] - z["Close_MA"]) / z["Close_MA"]) < (z["beta"])
        ):
            if z["s-1"] > z["Close_MA"]:
                return 0  # -1
            elif z["s-1"] < z["Close_MA"]:
                return 2  # 1
            else:
                return 1  # 0
        else:
            return 1  # 0

    @staticmethod
    def find_patterns(x):
        x["CDL2CROWS"] = (
            talib.CDL2CROWS(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        x["CDL3BLACKCROWS"] = (
            talib.CDL3BLACKCROWS(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDL3INSIDE'] = talib.CDL3INSIDE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDL3WHITESOLDIERS"] = (
            talib.CDL3WHITESOLDIERS(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        x["CDLABANDONEDBABY"] = (
            talib.CDLABANDONEDBABY(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLBELTHOLD"] = (
            talib.CDLBELTHOLD(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLCOUNTERATTACK"] = (
            talib.CDLCOUNTERATTACK(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        x["CDLDARKCLOUDCOVER"] = (
            talib.CDLDARKCLOUDCOVER(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLDOJISTAR'] = talib.CDLDOJISTAR(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLDRAGONFLYDOJI"] = (
            talib.CDLDRAGONFLYDOJI(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        x["CDLENGULFING"] = (
            talib.CDLENGULFING(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        x["CDLEVENINGDOJISTAR"] = (
            talib.CDLEVENINGDOJISTAR(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        x["CDLEVENINGSTAR"] = (
            talib.CDLEVENINGSTAR(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLGRAVESTONEDOJI"] = (
            talib.CDLGRAVESTONEDOJI(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        x["CDLHANGINGMAN"] = (
            talib.CDLHANGINGMAN(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        x["CDLHARAMICROSS"] = (
            talib.CDLHARAMICROSS(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLHIKKAKE'] = talib.CDLHIKKAKE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLINVERTEDHAMMER"] = (
            talib.CDLINVERTEDHAMMER(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLKICKING'] = talib.CDLKICKING(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLLONGLINE'] = talib.CDLLONGLINE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLMARUBOZU"] = (
            talib.CDLMARUBOZU(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLMATHOLD'] = talib.CDLMATHOLD(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLMORNINGDOJISTAR"] = (
            talib.CDLMORNINGDOJISTAR(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        x["CDLMORNINGSTAR"] = (
            talib.CDLMORNINGSTAR(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLONNECK'] = talib.CDLONNECK(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLPIERCING"] = (
            talib.CDLPIERCING(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLRISEFALL3METHODS"] = (
            talib.CDLRISEFALL3METHODS(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLSHOOTINGSTAR"] = (
            talib.CDLSHOOTINGSTAR(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLSHORTLINE'] = talib.CDLSHORTLINE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLSPINNINGTOP"] = (
            talib.CDLSPINNINGTOP(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLTHRUSTING'] = talib.CDLTHRUSTING(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLTRISTAR'] = talib.CDLTRISTAR(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x["CDLUPSIDEGAP2CROWS"] = (
            talib.CDLUPSIDEGAP2CROWS(x["Open"], x["High"], x["Low"], x["Close"]) / 100
        )
        # x['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(x['Open'], x['High'], x['Low'], x['Close']) / 100
        return x
