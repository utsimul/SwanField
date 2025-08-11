After running pre processing.py, I got this output for AAPL:

| Index |  close   |  high    |   low    |  open    |  volume  | volatility | momentum | sharpe_ratio | regime_state |  SMA_10  |  EMA_10  |  RSI_14  | BB_High  | BB_Low  | Log_Returns | Volatility_10 |  Hurst   | GARCH_Vol |
|-------|----------|----------|----------|----------|----------|------------|----------|--------------|--------------|----------|----------|----------|----------|----------|-------------|---------------|----------|-----------|
| 2     | 0.088763 | 0.084015 | 0.093706 | 0.077893 | 0.235949 | NaN        | NaN      | NaN          | 1.0          | 0.077628 | 0.075401 | 0.893922 | 0.070059 | 0.098925 | 0.007936    | 0.123903      | 0.423714 | NaN       |
| 3     | 0.087092 | 0.085129 | 0.099272 | 0.085124 | 0.212354 | NaN        | NaN      | NaN          | 1.0          | 0.077628 | 0.075401 | 0.893922 | 0.070059 | 0.098925 | -0.004714   | 0.123903      | 0.423714 | NaN       |
| 4     | 0.092781 | 0.089322 | 0.098895 | 0.081921 | 0.269901 | NaN        | NaN      | NaN          | 1.0          | 0.077628 | 0.075401 | 0.893922 | 0.070059 | 0.098925 | 0.015958    | 0.123903      | 0.423714 | NaN       |
| 5     | 0.100413 | 0.096419 | 0.109532 | 0.093969 | 0.364202 | NaN        | NaN      | NaN          | 1.0          | 0.077628 | 0.075401 | 0.893922 | 0.070059 | 0.098925 | 0.021019    | 0.123903      | 0.423714 | NaN       |
| 6     | 0.101242 | 0.099072 | 0.111944 | 0.097985 | 0.291141 | NaN        | NaN      | NaN          | 1.0          | 0.077628 | 0.075401 | 0.893922 | 0.070059 | 0.098925 | 0.002258    | 0.123903      | 0.423714 | NaN       |
| ...   | ...      | ...      | ...      | ...      | ...      | ...        | ...      | ...          | ...          | ...      | ...      | ...      | ...      | ...      | ...         | ...           | ...      | ...       |
| 1252  | 0.981664 | 0.978249 | 0.979707 | 0.983134 | 0.043702 | 0.008670   | 0.005271 | 9.652098     | 1.0          | 0.982158 | 0.982713 | 0.850242 | 0.976925 | 0.985875 | 0.003060    | 0.109011      | 0.504135 | 0.024705  |
| 1253  | 0.995991 | 0.990762 | 0.988640 | 0.986685 | 0.000000 | 0.008772   | 0.005534 | 10.014809    | 0.0          | 0.987527 | 0.990397 | 0.894786 | 0.989275 | 0.984115 | 0.011413    | 0.114373      | 0.581228 | 0.059444  |
| 1254  | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.009925 | 0.008617   | 0.005068 | 9.336627     | 1.0          | 0.993977 | 0.997455 | 0.906531 | 0.998442 | 0.987718 | 0.003171    | 0.105808      | 0.602810 | 0.058315  |
| 1255  | 0.983228 | 0.993157 | 0.977813 | 0.998225 | 0.047413 | 0.009435   | 0.003987 | 6.708575     | 0.0          | 0.997905 | 1.000000 | 0.758979 | 1.000000 | 0.993968 | -0.013331   | 0.132055      | 0.487631 | 0.055531  |
| 1256  | 0.966652 | 0.967740 | 0.966598 | 0.970608 | 0.030557 | 0.010131   | 0.003378 | 5.292879     | 1.0          | 1.000000 | 0.998890 | 0.635264 | 0.998155 | 1.000000 | -0.013352   | 0.151469      | 0.449437 | 0.057986  |


As you can see, most features are scaled and all seem proper. 
Here is the intuition behind some of these features:


##### GARCH volatility forecasting:

Main assumption by GARCH is that volatility clusters - so there are periods of high volatility where the large price fluctuations are followed by large price fluctuations. 
The GARCH value is relative predicted volatility so you can identify periods of high and low volatility that may occur in the future. It is not in the same unit as true volatility,
hence the significant difference in value. To get the original volatility from GARCH value, use `scalers['GARCH_Vol'].inverse_transform(df_normalized[['GARCH_Vol']])`

##### Hurst exponent:

Hurst = 0.5 (Random Walk): The series behaves like a pure random walk (e.g., Brownian motion) – no trends or patterns, purely unpredictable. Markets are often close to this in efficient conditions.
Hurst > 0.5 (Persistent/Trending): Positive autocorrelation; if prices have been rising, they're likely to continue rising. This indicates "momentum" or trending behavior, common in bull markets.
Hurst < 0.5 (Anti-Persistent/Mean-Reverting): Negative autocorrelation; trends reverse quickly. This suggests overreactions, where prices revert to a mean, common in volatile or ranging markets.

High Hurst (>0.5) implies self-similar trends that propagate (e.g., a weekly trend influencing hourly), while low Hurst (<0.5) signals complexity reduction as markets revert.

(Taken from Grok)

Now what I thought based on this was... you have a "chain" of values for every index. Now how do you exactly identify at what timescales the self-similarity occurs?
For example, patterns at t=10 might look similar to patterns at t=350 (say), but it may not look similar to t=200. Then how to find these?

Thus, we apply some transforms (Hurst_transforms.py) to analyse this. 

