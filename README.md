# MAARS

## Multi Agent Atmospheric Reinforcement Swans


![alt text](MAARS.png)

I aim to use Multi Agent Deep Reinforcment Learning and concepts of atmospheric modelling in order to implement portfolio optimization while detecting Black Swan events.

Will be updating my progress here for sanity.

### Step 1: Fetching and Pre processing:
Obtained data from yahoo finance API.
Arranging them in a format.
Normalization (Min Max scaling), 
Computed features ('SMA_20', 'EMA_20', 'RSI_14', 'BB_High', 'BB_Low', 'Volatility_20', 'Hurst')
Fit GARCH model for volatility forecasting.
