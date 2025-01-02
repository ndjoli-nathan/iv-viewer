This repository includes a Python package and a Streamlit application designed to plot the Implied Volatility Surface using stock options data sourced from Yahoo Finance.

- [Application Link](https://implied-volatility-surface-viewer.streamlit.app/)

A Jupyter Notebook is available in the [Package](https://github.com/ndjoli-nathan/iv-viewer/tree/main/Package) directory to demonstrate how to use it.

Note: The method used to calculate Implied Volatility in both the application and the package relies on the Black-Scholes-Merton model. Yahoo Finance does not provide information concerning the exercise style of the listed options, which may necessitate caution when applying the model to potentially non-European derivatives. Furthermore, the data provided is delayed by 15 minutes, which could impact analyses that depend on real-time information.
