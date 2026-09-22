# SwanField - Prototype version

I created a multi-agent deep reinforcement learning based framework for portfolio management, consisting of asset level agents, domain level agents and a master agent - mimicking how one might stack actual portfolio managers specializing in a particular asset, a particular domain, or the “master” manager who would make the ultimate final decision.

### Asset level agents:
Each asset agent takes into input the data for that particular asset (eg, AAPL, TSLA), and the signals from the domain agent and sends signals back to the domain agent. It serves a dual purpose: 
- To capture asset specific data, “condense” it and pass on how profitable it could be to the domain agent (through the output neuron values)
- To make a decision on whether to buy, hold or sell that asset based on this asset data as well as the signal sent from the corresponding domain agent - (softmax probabilities)

It uses a PPO based actor critic neural network, with the final reward explicitly designed to reward profits (log returns for now) made during periods of high volatility, thus encouraging anti- fragility.
All asset agents are trained using parameter sharing (which means they all share the same set of neural network parameters but are trained on different assets)

### Domain level agents:
Each domain agent takes into input the signals from the corresponding asset agent (so for example a domain agent for “Tech” receives signals from NVDA, MSFT, AAPL,...), as well as the signal from the master agent. 
The domain agent again serves dual purpose:
- To capture domain specific data through neural network outputs
- To decide on the maximum allocations for each of the assets it contains (which are passed on to the asset level agents as domain signals).
The architecture is similar to asset agents, with a PPO based actor critic and training using parameter sharing.

### Master agent:
There is only one master agent which takes into input the singals from all the domain agent and decides on portfolio allocation for each domain. It can be further modified in the future to estimate the chances of a major financial event such as a market crash.


![alt text](Prototype-architecture.png)


### Asset level PPO:

```text
Shared Encoder (seq + non-seq) 
        ↓
    Shared Latent h
        ↓
   ┌───────────────┬───────────────┬───────────────┐
   │ Actor Head 1  │ Actor Head 2  │ Actor Head 3  │
   │ Categorical(3)│   Signal dist │ Memory dist   │
   └───────────────┴───────────────┴───────────────┘
                ↓
            Critic Head
```

