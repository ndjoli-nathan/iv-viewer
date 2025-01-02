import streamlit as st
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from py_vollib_vectorized import vectorized_implied_volatility as BSM_IV
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Implied Volatility Surface", layout="centered", page_icon=":material/storm:")

st.title("Implied Volatility Surface Visualizer")

st.markdown("This application retrieves options data and the risk-free rate from Yahoo Finance, calculating implied volatility using the Black-Scholes-Merton model. You can filter the data based on criteria such as strike price, volatility range, trading activity, and expiration dates. The app generates interactive 3D graphs of the implied volatility surface, letting you customize visualization parameters. You can also view the filtered data in a table and download it as a CSV file for further analysis.")

st.sidebar.markdown(f"Created by Nathan Ndjoli")
st.sidebar.markdown(f"[Linkedin](https://www.linkedin.com/in/nndj/)")
st.sidebar.markdown(f"[E-mail](mailto:nathan.ndjoli1@gmail.fr)")
st.sidebar.markdown(f"[Github](https://github.com/ndjoli-nathan)")


class Infos:
    _cache = {}
    _last_update = {}

    def __init__(self, ticker):
        self.ticker = ticker

    @property
    def Data(self):
        if (self.ticker not in self._cache) or self._needs_refresh():
            self._cache[self.ticker] = yf.Ticker(self.ticker).info
            self._last_update[self.ticker] = datetime.datetime.now()
        return self._cache[self.ticker]

    def _needs_refresh(self, delay_in_seconds=3600):
        if self.ticker not in self._last_update:
            return True
        elapsed = (
            datetime.datetime.now() - self._last_update[self.ticker]
        ).total_seconds()
        return elapsed > delay_in_seconds


class IV:
    def __init__(
        self,
        options_data,
        minstrike=0,
        maxstrike=np.inf,
        max_days_since_last_trade=5,
        min_t_days=1,
        min_iv=0.01,
        max_iv=2,
        min_last_price=5,
    ):
        self.minstrike = minstrike
        self.maxstrike = maxstrike
        self.max_days_since_last_trade = max_days_since_last_trade
        self.min_t_days = min_t_days
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.min_last_price = min_last_price
        self.df = self._build_df(options_data)

    def _build_df(self, options_list):
        df = pd.DataFrame(options_list).copy()
        cols_drop = [
            "contractSymbol",
            "change",
            "percentChange",
            "impliedVolatility",
            "inTheMoney",
            "contractSize",
            "currency",
        ]
        df.drop(columns=cols_drop, errors="ignore", inplace=True)
        df.rename(
            columns={
                "lastTradeDate": "last_trade",
                "strike": "K",
                "lastPrice": "last_price",
                "bid": "bid_price",
                "ask": "ask_price",
                "openInterest": "open_interest",
            },
            inplace=True,
        )
        df = df[
            [
                "flag",
                "S",
                "K",
                "expiry_date",
                "t_days",
                "t_years",
                "q",
                "r",
                "IV",
                "last_price",
                "bid_price",
                "ask_price",
                "volume",
                "open_interest",
                "last_trade",
                "days_since_last_trade",
            ]
        ]

        mask = (
            df["IV"].notna()
            & (df["days_since_last_trade"] < self.max_days_since_last_trade)
            & (df["t_days"] > self.min_t_days)
            & (df["last_price"] > self.min_last_price)
            & (df["IV"] >= self.min_iv)
            & (df["IV"] <= self.max_iv)
            & (df["K"] >= self.minstrike)
            & (df["K"] <= self.maxstrike)
        )
        df = df[mask]
        return df

    def TermStructure(self):
        if self.df.empty:
            return None

        term_df = self.df.groupby("t_days")["IV"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 6.75))
        ax.plot(term_df["t_days"], term_df["IV"])
        ax.set_xlabel("Days to Expiry")
        ax.set_ylabel("Average IV")
        ax.grid(True)
        return fig

    def Surface(
        self,
        granularity=64,
        smooth=0.1,
        function="multiquadric",
        cmap="viridis",
        scatter=True,
        elev=15,
        azim=45,
    ):
        if self.df.empty:
            st.warning("No data available for the surface.")
            return None

        grouped = self.df.groupby(["K", "t_days"])["IV"].mean().reset_index()
        if grouped.empty:
            st.warning("No data available after grouping for the surface.")
            return None

        K = grouped["K"].values
        T = grouped["t_days"].values
        IV_ = grouped["IV"].values

        rbf = Rbf(K, T, IV_, function=function, smooth=smooth)
        K_grid, T_grid = np.meshgrid(
            np.linspace(K.min(), K.max(), granularity),
            np.linspace(T.min(), T.max(), granularity),
        )
        IV_grid = rbf(K_grid, T_grid)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            T_grid, K_grid, IV_grid, cmap=cmap, edgecolor=None, alpha=0.8
        )

        if scatter:
            ax.scatter(T, K, IV_, color="black", s=2.5, alpha=1, marker="+")

        ax.set_xlabel("Days before expiry")
        ax.set_ylabel("Strike")
        ax.set_zlabel("Implied Volatility")
        ax.view_init(elev=elev, azim=azim)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        return fig


class Options:
    _cache = {}
    _last_update = {}

    def __init__(self, ticker):
        self.ticker = ticker
        self._iv_cache = {}

    @property
    def Data(self):
        if (self.ticker not in self._cache) or self._needs_refresh():
            infos = Infos(self.ticker).Data
            S = infos.get("currentPrice", 0)
            if S == 0:
                S = yf.Ticker(self.ticker).history("1d", "1m").iloc[-1]

            q = infos.get("dividendYield", 0)
            self._cache[self.ticker] = self._fetch_options(S, q)
            self._last_update[self.ticker] = datetime.datetime.now()
        return self._cache[self.ticker]

    def IV(
        self,
        minstrike=0,
        maxstrike=np.inf,
        max_days_since_last_trade=5,
        min_t_days=1,
        min_iv=0.01,
        max_iv=2,
        min_last_price=5,
    ):

        cache_key = (
            minstrike,
            maxstrike,
            max_days_since_last_trade,
            min_t_days,
            min_iv,
            max_iv,
            min_last_price,
        )
        if cache_key not in self._iv_cache:
            self._iv_cache[cache_key] = IV(
                self.Data,
                minstrike=minstrike,
                maxstrike=maxstrike,
                max_days_since_last_trade=max_days_since_last_trade,
                min_t_days=min_t_days,
                min_iv=min_iv,
                max_iv=max_iv,
                min_last_price=min_price,
            )
        return self._iv_cache[cache_key]

    def _needs_refresh(self, delay_in_seconds=3600):
        if self.ticker not in self._last_update:
            return True
        elapsed = (
            datetime.datetime.now() - self._last_update[self.ticker]
        ).total_seconds()
        return elapsed > delay_in_seconds

    def _fetch_options(self, S, q):
        options = []
        mats = yf.Ticker(self.ticker).options
        now_utc = pd.to_datetime(datetime.datetime.now(), utc=True)
        r = yf.Ticker("^FVX").history("1d", "1m").Close.iloc[-1] / 100

        for mat in mats:
            o_ = yf.Ticker(self.ticker).option_chain(mat)
            o_c = o_.calls.to_dict("records")
            o_p = o_.puts.to_dict("records")
            for o in o_c:
                o["expiry_date"] = pd.to_datetime(mat).tz_localize("UTC")
                o["flag"] = "c"
                options.append(o)
            for o in o_p:
                o["expiry_date"] = pd.to_datetime(mat).tz_localize("UTC")
                o["flag"] = "p"
                options.append(o)

        for o in options:
            o["lastTradeDate"] = pd.to_datetime(o["lastTradeDate"], utc=True)
            now = pd.to_datetime(datetime.datetime.now(), utc=True)
            o["days_since_last_trade"] = (now - o["lastTradeDate"]).days

            o["S"] = S
            o["q"] = q
            o["r"] = r
            o["t_days"] = (o["expiry_date"] - now).days
            o["t_years"] = o["t_days"] / 365

            price = o["lastPrice"]
            K = o["strike"]
            t = o["t_years"]
            if t == 0:
                continue
            r = o["r"]
            flag = o["flag"]

            o["IV"] = BSM_IV(
                price,
                S,
                K,
                t,
                r,
                flag,
                q,
                model="black_scholes_merton",
                return_as="numpy",
                on_error="ignore",
            )[0]

        return options


class ImpliedVolatilitySurface:
    def __init__(self, ticker):
        self._ticker = ticker
        self.Infos = Infos(ticker)
        self.Options = Options(ticker)



container_0 = st.container()
container_0.subheader("Parameters", divider=True)


with container_0:
    column_1, column_2, column_3, column_4, column_5 = st.columns(5, gap="large")

    with column_1:
        ticker = st.text_input("Ticker:", value="AAPL")
        minstrike = st.number_input("Min Strike:", value=200)
        maxstrike = st.number_input("Max Strike:", value=300)

    with column_2:
        max_days = st.number_input("Max Days Since Last Trade:", value=7)
        min_tdays = st.number_input("Min Days to Expiry:", value=1)
        min_price = st.number_input("Min Last Price:", value=0.5)

    with column_3:
        min_iv = st.number_input("Min IV:", value=0.01)
        max_iv = st.number_input("Max IV:", value=2.0)
        func = st.selectbox(
            "RBF Function:",
            [
                "cubic",
                "gaussian",
                "inverse_multiquadric",
                "linear",
                "multiquadric",
                "quintic",
                "thin_plate",
            ],
            index=4,
        )

    with column_4:
        granularity = st.number_input("Granularity:", value=64, step=8)
        smooth = st.number_input("Smoothing:", value=0.5)
        cmap = st.selectbox(
            "Colormap:",
            [
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "jet",
                "coolwarm",
                "RdBu",
                "Greys",
                "hot",
            ],
            index=0,
        )

    with column_5:
        elev = st.slider(
            "Elevation Angle:", min_value=0, max_value=90, value=15, step=15
        )
        azim = st.slider(
            "Azimuth Angle:", min_value=0, max_value=360, value=45, step=15
        )
        scatter = st.checkbox("Display scatter points ", value=True)


container_1 = st.container()
container_1.subheader(f"{ticker} Implied Volatility Surface", divider=True)
container_1.text(
    f"Below is a plot showing the implied volatility of {ticker} stock options for strikes ranging from {minstrike} to {maxstrike} as of {datetime.datetime.now().date()}. Options with a market price under {min_price}, an implied volatility outside ({min_iv}, {max_iv}), more than {max_days} days since the last trade, or fewer than {min_tdays} days to expiration have been excluded. The interpolation method that was used is {func}."
)


with container_1:

    surface = ImpliedVolatilitySurface(ticker)

    options_data = surface.Options.IV(
        minstrike=minstrike,
        maxstrike=maxstrike,
        max_days_since_last_trade=max_days,
        min_t_days=min_tdays,
        min_iv=min_iv,
        max_iv=max_iv,
        min_last_price=min_price,
    )

    figure_size = (10, 10)

    fig_surface = options_data.Surface(
        granularity=granularity,
        smooth=smooth,
        function=func,
        cmap=cmap,
        scatter=scatter,
        elev=elev,
        azim=azim,
    )
    if fig_surface:
        st.pyplot(fig_surface)
        plt.close(fig_surface)
    else:
        st.warning("Unable to display the surface.")

    st.markdown(
        f"The following table contains all the {ticker} Options data that were used to plot the volatility surface. The source of the data is [Yahoo Finance](https://finance.yahoo.com/quote/{ticker}/options/)."
    )

    if not options_data.df.empty:
        st.dataframe(options_data.df)

        csv_data = options_data.df.to_csv(index=False)
        st.download_button(
            label="Download Options Data",
            data=csv_data,
            file_name=f"{ticker}_{datetime.datetime.now().date()}_filtered-options-data.csv",
            mime="text/csv",
        )
    else:
        st.warning("No filtered data available.")
