import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from py_vollib_vectorized import vectorized_implied_volatility as BSM_IV
from scipy.interpolate import Rbf
import ipywidgets as w
from ipywidgets import interactive
from IPython.display import display  

class Infos:
    """
    Retrieves and caches ticker information from Yahoo Finance.
    Automatically refreshes data based on a given delay.
    """
    _cache = {}
    _last_update = {}

    def __init__(self, ticker):
        """
        Parameters
        ----------
        ticker : str
            The ticker symbol (e.g., 'AAPL').
        """
        self.ticker = ticker

    @property
    def Data(self):
        """
        Returns the cached data or fetches it if needed.

        Returns
        -------
        dict
            A dictionary containing the ticker info.
        """
        if (self.ticker not in self._cache) or self._needs_refresh():
            self._cache[self.ticker] = yf.Ticker(self.ticker).info
            self._last_update[self.ticker] = datetime.datetime.now()
        return self._cache[self.ticker]

    def _needs_refresh(self, delay_in_seconds=3600):
        """
        Checks if the data is stale based on a time delay.

        Parameters
        ----------
        delay_in_seconds : int
            Time delay in seconds after which the data is considered stale.

        Returns
        -------
        bool
            True if the data needs a refresh, False otherwise.
        """
        if self.ticker not in self._last_update:
            return True
        elapsed = (datetime.datetime.now() - self._last_update[self.ticker]).total_seconds()
        return elapsed > delay_in_seconds

class IV:
    """
    Filters and computes implied volatility (IV) from options data.
    Builds a DataFrame with the necessary columns and supports
    term structure and surface visualization.
    """
    def __init__(
        self, options_data, minstrike=0, maxstrike=np.inf,
        max_days_since_last_trade=5, min_t_days=1,
        min_iv=0.01, max_iv=2, min_last_price=5
    ):
        """
        Parameters
        ----------
        options_data : list
            List of options data dictionaries.
        minstrike : float
            Minimum strike filter.
        maxstrike : float
            Maximum strike filter.
        max_days_since_last_trade : int
            Maximum days since last trade filter.
        min_t_days : int
            Minimum time to maturity in days.
        min_iv : float
            Minimum implied volatility filter.
        max_iv : float
            Maximum implied volatility filter.
        min_last_price : float
            Minimum last price filter.
        """
        self.minstrike = minstrike
        self.maxstrike = maxstrike
        self.max_days_since_last_trade = max_days_since_last_trade
        self.min_t_days = min_t_days
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.min_last_price = min_last_price
        self.df = self._build_df(options_data)

    def _build_df(self, options_list):
        """
        Builds and filters the DataFrame from raw options data.

        Parameters
        ----------
        options_list : list
            List of options data dictionaries.

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame based on the criteria.
        """
        df = pd.DataFrame(options_list).copy()
        cols_drop = [
            "contractSymbol","change","percentChange","impliedVolatility",
            "inTheMoney","contractSize","currency"
        ]
        df.drop(columns=cols_drop, errors='ignore', inplace=True)
        df.rename(columns={
            'lastTradeDate':'last_trade',
            'strike':'K',
            'lastPrice':'last_price',
            'bid':'bid_price',
            'ask':'ask_price',
            'openInterest':'open_interest'
        }, inplace=True)
        df = df[[
            "flag","S","K","expiry_date","t_days","t_years","q","r","IV",
            "last_price","bid_price","ask_price","volume","open_interest",
            "last_trade","days_since_last_trade"
        ]]
        mask = (
            df['IV'].notna() &
            (df['days_since_last_trade'] < self.max_days_since_last_trade) &
            (df['t_days'] > self.min_t_days) &
            (df['last_price'] > self.min_last_price) &
            (df['IV'] >= self.min_iv) &
            (df['IV'] <= self.max_iv) &
            (df['K'] >= self.minstrike) &
            (df['K'] <= self.maxstrike)
        )
        df = df[mask]
        return df

    def TermStructure(self):
        """
        Plots the average IV term structure over time-to-expiry.
        """
        if self.df.empty:
            print("Aucune donnée disponible après application des filtres.")
            return
        term_df = self.df.groupby('t_days')['IV'].mean().reset_index()
        plt.figure(figsize=(10,4))
        plt.plot(term_df['t_days'], term_df['IV'])
        plt.ylabel('Average IV')
        plt.xlabel('Days to Expiry')
        plt.title('Implied Volatility Term Structure')
        plt.show()

    def Surface(
        self, granularity=64, smooth=.1, function='multiquadric',
        cmap='viridis', scatter=True, elev=15, azim=45
    ):
        """
        Plots the implied volatility surface in 3D.

        Parameters
        ----------
        granularity : int
            Number of grid points in each dimension.
        smooth : float
            Smoothing factor for the RBF interpolation.
        function : str
            Radial basis function type.
        cmap : str
            Colormap for the 3D surface.
        scatter : bool
            If True, scatter plot the original data points.
        elev : int
            Elevation angle for 3D view.
        azim : int
            Azimuth angle for 3D view.
        """
        if self.df.empty:
            print("Aucune donnée disponible après application des filtres.")
            return
        grouped = self.df.groupby(['K','t_days'])['IV'].mean().reset_index()
        if grouped.empty:
            print("Aucune donnée disponible après groupement.")
            return
        K = grouped['K'].values
        T = grouped['t_days'].values
        IV_ = grouped['IV'].values
        rbf = Rbf(K, T, IV_, function=function, smooth=smooth)
        K_grid, T_grid = np.meshgrid(
            np.linspace(K.min(), K.max(), granularity),
            np.linspace(T.min(), T.max(), granularity)
        )
        IV_grid = rbf(K_grid, T_grid)
        fig = plt.figure(figsize=(16,16))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(T_grid, K_grid, IV_grid, cmap=cmap, edgecolor=None, alpha=1)
        if scatter:
            ax.scatter(T, K, IV_, color='black', s=2.5, alpha=1, marker='+')
        ax.set_xlabel('Days before expiry')
        ax.set_ylabel('Strike')
        ax.set_zlabel('Implied Volatility')
        plt.title('Implied Volatility Surface')
        ax.view_init(elev=elev, azim=azim)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        plt.show()

    def InteractiveSurface(self):
        """
        Creates an interactive widget to explore the IV surface
        with adjustable parameters.
        """
        ticker = w.Text(value='GE', description='Ticker:')
        minstrike = w.FloatSlider(value=75, min=0, max=2000, step=1, description='Min Strike:')
        maxstrike = w.FloatSlider(value=125, min=0, max=2000, step=1, description='Max Strike:')
        max_days = w.IntSlider(value=5, min=1, max=30, step=1, description='Max Days:')
        min_tdays = w.IntSlider(value=1, min=1, max=1085, step=1, description='Min T Days:')
        min_iv = w.FloatSlider(value=0.01, min=0, max=1, step=0.01, description='Min IV:')
        max_iv = w.FloatSlider(value=2, min=1, max=5, step=0.01, description='Max IV:')
        min_price = w.FloatSlider(value=5, min=0, max=100, step=0.1, description='Min Last Price:')
        granularity = w.IntSlider(value=64, min=2, max=512, step=2, description='Granularity:')
        smooth = w.FloatSlider(value=20, min=0, max=50, step=0.5, description='Smooth:')
        elev = w.IntSlider(value=15, min=0, max=90, step=5, description='Elev:')
        azim = w.IntSlider(value=45, min=0, max=360, step=5, description='Azim:')
        scatter = w.Checkbox(value=True, description='Scatter:')
        cmap = w.Dropdown(options=['viridis','plasma','inferno','magma'], value='viridis', description='Colormap:')
        func = w.Dropdown(options=['linear','cubic','quintic','multiquadric'], value='multiquadric', description='Function:')

        def update(
            t, minst, maxst, mxdays, mintd, mniv, mxiv, mnpr,
            gran, sm, el, az, sc, cm, f
        ):
            """
            Updates the plotted surface based on widget parameters.
            """
            try:
                surface = ImpliedVolatilitySurface(t)
                data = surface.Options.IV(
                    minstrike=minst,
                    maxstrike=maxst,
                    max_days_since_last_trade=mxdays,
                    min_t_days=mintd,
                    min_iv=mniv,
                    max_iv=mxiv,
                    min_last_price=mnpr
                )
                data.Surface(
                    granularity=gran,
                    smooth=sm,
                    function=f,
                    cmap=cm,
                    scatter=sc,
                    elev=el,
                    azim=az
                )
            except Exception as e:
                print(f"Erreur : {e}")

        ui = w.VBox([
            ticker, minstrike, maxstrike, max_days, min_tdays,
            min_iv, max_iv, min_price, granularity, smooth,
            elev, azim, scatter, cmap, func
        ])

        interactive_plot = interactive(
            update,
            t=ticker,
            minst=minstrike,
            maxst=maxstrike,
            mxdays=max_days,
            mintd=min_tdays,
            mniv=min_iv,
            mxiv=max_iv,
            mnpr=min_price,
            gran=granularity,
            sm=smooth,
            el=elev,
            az=azim,
            sc=scatter,
            cm=cmap,
            f=func
        )
        display(w.VBox([ui, interactive_plot.children[-1]]))  

class Options:
    """
    Manages fetching and caching of options chains and
    instantiates an IV object for implied volatility analysis.
    """
    _cache = {}
    _last_update = {}

    def __init__(self, ticker):
        """
        Parameters
        ----------
        ticker : str
            Ticker symbol to fetch options data.
        """
        self.ticker = ticker
        self._iv_cache = {}

    @property
    def Data(self):
        """
        Returns the cached options data or fetches it if needed.

        Returns
        -------
        list
            List of options data dictionaries.
        """
        if (self.ticker not in self._cache) or self._needs_refresh():
            infos = Infos(self.ticker).Data
            S = infos.get('currentPrice', np.nan)
            q = infos.get('dividendYield', 0)
            self._cache[self.ticker] = self._fetch_options(S, q)
            self._last_update[self.ticker] = datetime.datetime.now()
        return self._cache[self.ticker]

    def IV(
        self, minstrike=0, maxstrike=np.inf,
        max_days_since_last_trade=5, min_t_days=1,
        min_iv=0.01, max_iv=2, min_last_price=5
    ):
        """
        Returns an IV object with filtered options data.

        Parameters
        ----------
        minstrike : float
            Minimum strike filter.
        maxstrike : float
            Maximum strike filter.
        max_days_since_last_trade : int
            Maximum days since last trade.
        min_t_days : int
            Minimum time to maturity in days.
        min_iv : float
            Minimum implied volatility filter.
        max_iv : float
            Maximum implied volatility filter.
        min_last_price : float
            Minimum last price filter.

        Returns
        -------
        IV
            An IV object with the filtered data.
        """
        cache_key = (
            minstrike, maxstrike, max_days_since_last_trade,
            min_t_days, min_iv, max_iv, min_last_price
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
                min_last_price=min_last_price
            )
        return self._iv_cache[cache_key]

    def _needs_refresh(self, delay_in_seconds=3600):
        """
        Checks if the options data is stale based on a time delay.

        Parameters
        ----------
        delay_in_seconds : int
            Time delay in seconds after which data is considered stale.

        Returns
        -------
        bool
            True if the data should be refreshed, False otherwise.
        """
        if self.ticker not in self._last_update:
            return True
        elapsed = (datetime.datetime.now() - self._last_update[self.ticker]).total_seconds()
        return elapsed > delay_in_seconds

    def _fetch_options(self, S, q):
        """
        Fetches, combines, and processes calls and puts for each expiry.

        Parameters
        ----------
        S : float
            Current underlying price.
        q : float
            Dividend yield.

        Returns
        -------
        list
            A list of processed options data dictionaries with IV added.
        """
        options = []
        fvx = yf.Ticker('^FVX')
        fvx_history = fvx.history(period='1d', interval='1m')
        if fvx_history.empty:
            r = 0.0  # Valeur par défaut si les données sont manquantes
        else:
            r = fvx_history['Close'].iloc[-1] / 100
        mats = yf.Ticker(self.ticker).options
        now = pd.to_datetime(datetime.datetime.now(), utc=True)
        
        for mat in mats:
            option_chain = yf.Ticker(self.ticker).option_chain(mat)
            calls = option_chain.calls.to_dict("records")
            puts = option_chain.puts.to_dict("records")

            for o in calls:
                o["expiry_date"] = pd.to_datetime(mat).tz_localize("UTC")
                o["flag"] = "c"
                options.append(o)
            
            for o in puts:
                o["expiry_date"] = pd.to_datetime(mat).tz_localize("UTC")
                o["flag"] = "p"
                options.append(o)
        
        for o in options:
            o["lastTradeDate"] = pd.to_datetime(o["lastTradeDate"], utc=True)
            o["days_since_last_trade"] = (now - o["lastTradeDate"]).days
            o["S"] = S
            o["q"] = q
            o["r"] = r
            o["t_days"] = (o["expiry_date"] - now).days
            o["t_years"] = o["t_days"] / 365

            price = o.get("lastPrice", np.nan)
            K = o.get("strike", np.nan)
            t = o.get("t_years", np.nan)
            if pd.isna(t) or t <= 0:
                o["IV"] = np.nan
                continue

            flag = o.get("flag", "c")
            try:
                iv = BSM_IV(
                    price, S, K, t, r, flag, q,
                    model="black_scholes_merton",
                    return_as="numpy",
                    on_error="ignore"
                )[0]
                o["IV"] = iv if not np.isnan(iv) else np.nan
            except Exception:
                o["IV"] = np.nan
        
        return options

class ImpliedVolatilitySurface:
    """
    Main entry point to retrieve ticker info and options chains
    for implied volatility surface analysis.
    """
    def __init__(self, ticker):
        """
        Parameters
        ----------
        ticker : str
            Ticker symbol used for all subsequent operations.
        """
        self._ticker = ticker
        self.Infos = Infos(ticker)
        self.Options = Options(ticker)
