# SwanField — Hierarchical Multi-Agent Reinforcement Learning for Portfolio Management

SwanField is a **hierarchical multi-agent deep reinforcement learning framework for portfolio management**.

The framework consists of three levels of decision-making agents:

1. **Asset-level agents** — specialize in individual assets.
2. **Domain-level agents** — aggregate information across related assets/sectors.
3. **Master agent** — aggregates domain-level information and determines the overall portfolio allocation.

The architecture is inspired by how a hierarchical portfolio-management organization could operate: specialized asset managers communicate with domain/sector managers, who in turn communicate with a master portfolio manager.

The agents use **Proximal Policy Optimization (PPO)** with actor-critic architectures and stochastic policies.

---

## Architecture

The information flow is:

![Architecture](Prototype-architecture.png)

---

# 1. Asset-Level Agents

Each asset agent receives:

- sequential asset data,
- non-sequential information such as allocation/memory information,
- and produces four stochastic actions.

The asset policy consists of:

1. **BUY / HOLD / SELL**
2. **Signal sent to the domain agent**
3. **Memory update**
4. **Trade fraction**

The implementation uses an LSTM/GRU-based sequence encoder followed by a shared representation and multiple actor heads. The sequence encoder produces the final recurrent hidden representation `hseq`.

---

## 1.1 Sequential Encoding

Let the sequential input for an asset be

$$
Xt={xt-T+1,…,xt}
$$

where `T` is the sequence length.

The recurrent encoder produces

$$
hseq,t=RNN(Xt)
$$

where the RNN can be an LSTM or GRU.

For an LSTM:

$$
(ht,ct)=LSTM(xt,ht-1,ct-1)
$$

and the final hidden state is used as the sequence representation:

$$
hseq=hT.
$$

In the implementation, the final recurrent hidden state is returned as the asset's sequence encoding.

---

## 1.2 Shared Asset Representation

The sequential representation is concatenated with non-sequential information:

$$
xtasset=[hseq,t \Vert xtnonseq]
$$

where `∣∣` denotes concatenation.

The shared hidden representation is

$$
ht=tanh(Whxtasset+bh).
$$

In the implementation, this shared representation is used by all actor heads and the critic.

---

# 2. Asset Policy

The asset policy is a joint stochastic policy:

$$
\pi\theta(at∣st)=\pi{\theta}_{BHS} \pi {\theta}_{signal} \pi{\theta}_memory \pi{\theta}_fraction.
$$

The complete action is

$$
at=(atBHS,atsignal,atmemory,atfraction).
$$

---

## 2.1 BUY / HOLD / SELL Head

The first actor head produces three logits:

$$
zt=WBHSht+bBHS
$$

where

$$
zt\in R3.
$$

The logits are converted into probabilities using softmax:

$$
\pi\thetaBHS(a∣st)=ezt,a\sumj=13ezt,j.
$$

Therefore,

$$
atBHS\simCategorical(\pi\thetaBHS).
$$

The three actions are:

$$
ABHS={BUY,HOLD,SELL}.
$$

The implementation constructs this distribution directly from the actor logits.

---

## 2.2 Asset → Domain Signal

The asset agent also generates a continuous signal for the domain agent.

The mean is

$$
\mutAD=W\muADht+b\muAD
$$

and the raw log-standard-deviation output is

$$
ℓtAD=W\sigmaADht+b\sigmaAD.
$$

The implementation clamps this value to maintain a minimum standard deviation:

$$
ℓ~tAD=max(ℓtAD,log(\sigmamin)).
$$

The standard deviation is then

$$
\sigmatAD=exp(ℓ~tAD).
$$

Therefore the asset-to-domain signal is sampled from

$$
atAD\simN(\mutAD,(\sigmatAD)2).
$$

For `d` signal dimensions, this represents `d` independent Gaussian variables.

The implementation uses `Normal(mean, std)` for this actor head.

---

## 2.3 Memory Update

The memory head generates a continuous memory update:

$$
\mutM=W\muMht+b\muM
$$

$$
ℓtM=W\sigmaMht+b\sigmaM
$$

$$
\sigmatM=exp(max(ℓtM,log\sigmamin)).
$$

The memory update is sampled as

$$
atM\simN(\mutM,(\sigmatM)2).
$$

Thus, the agent can maintain a learned continuous internal state across timesteps.

The implementation contains separate mean and standard-deviation heads for this distribution.

---

## 2.4 Trade Fraction

The trade-fraction head determines what fraction of the available position/capital should be involved in the trade.

The mean is constrained to `[0,1]`:

$$
\mutF=\sigma(W\muFht+b\muF)
$$

where

$$
\sigma(x)=\frac{1}{1+e^{-x.
$$

The standard deviation is

$$
\sigmatF=exp(max(ℓtF,log\sigmamin)).
$$

The trade fraction is sampled from

$$
atF\simN(\mutF,(\sigmatF)2).
$$

The sampled value is finally clipped to the valid interval:

$$
atF=clip(atF,0,1).
$$

This corresponds to the implementation's sigmoid mean followed by sampling and clipping.

---

# 3. Joint Asset Log-Probability

Because the asset action consists of multiple stochastic components, the total log-probability is the sum of the component log-probabilities:

$$
log\pi\theta(at∣st)=log\piBHS+log\piAD+log\piM+log\piF.
$$

For the Gaussian components,

$$
log\piAD=\sumilogN(at,iAD;\mut,iAD,(\sigmat,iAD)2)
$$

and similarly for memory and trade fraction.

Therefore:

$$
log\pi\theta(at∣st)=log\piBHS+\sumilog\piAD,i+\sumilog\piM,i+\sumilog\piF,i
$$

The implementation explicitly constructs this sum before storing the PPO log-probability.

---

# 4. Asset Critic

The critic shares the common hidden representation `ht` with the actor heads.

It estimates the state value:

$$
V\phi(st)=WVht+bV.
$$

Thus,

$$
V\phi(st)\approxE[\sumk=0\infty\gammakrt+k∣st].
$$

The implementation produces this scalar through a linear critic head.

---

# 5. Domain-Level Agents

A domain agent aggregates information from multiple asset agents.

For example:

```text
AAPL ─┐
MSFT ─┤
NVDA ─┤
GOOG ─┼──> Tech Domain Agent
TSLA ─┤
...  ─┘

```

Each asset produces an embedding

$$
hiasset
$$

which is provided to the domain agent.

The domain agent uses an **attention pooling mechanism** to aggregate these asset representations:

$$
Hassets={h1,…,hN}.
$$

The attention mechanism produces:

$$
hdomain=AttentionPool(Hassets).
$$

This creates a fixed-size domain representation regardless of the number/order of individual asset representations.

---

## 5.1 Domain Input

The domain agent receives three sources of information:

$$
xtD=[hdomain \Vert mtD \Vert stM\to D].
$$

where:

- `hdomain` = pooled asset representation,
- `mtD` = domain memory,
- `stM→D` = signal from the master agent.

The shared domain representation is then

$$
htD=ReLU(WDxtD+bD).
$$

This corresponds to concatenating the attention-pooled asset representation, master signal, and memory before the shared network.

---

# 6. Domain-Level Actions

The domain agent produces three outputs:

1. **Asset allocation**
2. **Domain → Master signal**
3. **Memory update**

---

## 6.1 Asset Allocation with a Dirichlet Policy

Instead of independently predicting each asset's allocation, the domain agent produces the concentration parameters of a **Dirichlet distribution**.

The network first produces raw outputs:

$$
ztD=WallochtD+balloc.
$$

These are converted into positive concentration parameters:

$$
\alphat=softplus(ztD)+ϵ.
$$

where

$$
\alphat,i>0.
$$

The allocation vector is then sampled as

$$
atD\simDirichlet(\alphat).
$$

Therefore,

$$
atD=[at,1,at,2,…,at,N,at,cash]
$$

with

$$
at,i\ge0
$$

and

$$
\sumiat,i=1
$$

This naturally represents portfolio allocation because the outputs form a probability-simplex vector.

The implementation uses `softplus(raw_alpha) + 1e-3` to ensure positive Dirichlet concentration parameters.

---

## 6.2 Domain → Master Signal

The domain agent produces a continuous signal to the master agent:

$$
\mutDM=W\muDMhtD+b\muDM
$$

$$
\sigmatDM=exp(max(W\sigmaDMhtD+b\sigmaDM,log\sigmamin)).
$$

The signal is sampled as

$$
atDM\simN(\mutDM,(\sigmatDM)2).
$$

Thus the master agent receives learned stochastic signals from each domain agent.

---

## 6.3 Domain Memory Update

Similarly,

$$
\mutM,D=W\muM,DhtD+b\muM,D
$$

and

$$
\sigmatM,D=exp(max(W\sigmaM,DhtD+b\sigmaM,D,log\sigmamin)).
$$

Then

$$
mt+1D\simN(\mutM,D,(\sigmatM,D)2).
$$

---

# 7. Domain Critic

The domain critic estimates:

$$
V\phiD(stD)=WVDhtD+bVD.
$$

Therefore the domain agent has its own value function describing the expected future return from the domain's current state.

---

# 8. Domain Joint Policy

The domain action consists of:

$$
atD=(atalloc,atDM,atM,D).
$$

Its joint policy is therefore:

$$
\pi\thetaD=\pi\thetaalloc\pi\thetaDM\pi\thetaM,D.
$$

The joint log-probability is

$$
log\pi\thetaD(at∣st)=log\pi\thetaalloc+log\pi\thetaDM+log\pi\thetaM,D
$$

which is exactly the quantity used to construct the PPO probability ratio.

---

# 9. Master Agent

The master agent sits at the highest level of the hierarchy.

It receives the representations generated by the domain agents:

$$
HD={h1D,h2D,…,hKD}.
$$

These are aggregated using attention pooling:

$$
htM=AttentionPool(HD).
$$

The master representation is then combined with master memory:

$$
xtM=[htM \Vert mtM].
$$

The shared master representation is

$$
ht=ReLU(WMxtM+bM).
$$

---

# 10. Master Portfolio Allocation

The master agent produces allocations across domains plus a cash allocation.

Raw allocation outputs are:

$$
ztM=WallocMhtM+ballocM.
$$

These are transformed into Dirichlet concentration parameters:

$$
\alphatM=softplus(ztM)+ϵ.
$$

The master allocation is sampled from:

$$
atM\simDirichlet(\alphatM)
$$

where

$$
atM=[at,1domain,…,at,Kdomain,at,cash]
$$

and

$$
\sumiat,iM=1.
$$

Thus, the master agent determines the high-level allocation across domains while the domain agents determine allocations within those domains.

---

# 11. Master Memory

The master memory update is represented by a Gaussian policy:

$$
\mutM=W\muMht+b\muM
$$

$$
\sigmatM=exp(max(W\sigmaMht+b\sigmaM,log\sigmamin)).
$$

The new memory is sampled as

$$
mt+1M\simN(\mutM,(\sigmatM)2).
$$

---

# 12. Master Critic

The master critic estimates the value of the global portfolio state:

$$
V\phiM(st)=WVMht+bVM.
$$

This provides the value estimate used to calculate the PPO advantage and critic loss.

---

# 13. PPO Training

All three levels use **Proximal Policy Optimization (PPO)**.

For a transition `t`, let:

$$
rt(\theta)=\pi\theta(at∣st)\pi\thetaold(at∣st).
$$

Using log-probabilities:

$$
rt(\theta)=exp(log\pi\theta(at∣st)-log\pi\thetaold(at∣st))
$$

This is the exact probability ratio used by the implementation.

---

## 13.1 Advantage

The PPO actor uses an estimated advantage:

$$
At=R^t-V\phi(st)
$$

where `R^t` is the estimated return.

The advantage measures whether the selected action performed better or worse than expected under the critic's estimate.

---

# 14. PPO Clipped Objective

The two surrogate objectives are:

$$
LtCLIP=rt(\theta)At
$$

and

$$
L~tCLIP=clip(rt(\theta),1-ϵ,1+ϵ)At.
$$

The PPO objective is:

$$
LCLIP=Et[min(rtAt,clip(rt,1-ϵ,1+ϵ)At)]
$$

where the implementation uses

$$
ϵ=0.2.
$$

The actor loss is the negative of this objective:

$$
Lactor=-Et[min(rtAt,clip(rt,1-ϵ,1+ϵ)At)].
$$

This clipping prevents the updated policy from moving excessively far from the old policy.

---

# 15. Critic Loss

The critic is trained using mean squared error between the estimated return and value prediction:

$$
Lvalue=Et[(R^t-V\phi(st))2]
$$

The implementation uses this directly as:

$$
(return-value)2.
$$

---

# 16. Entropy Regularization

Entropy encourages the policy to retain exploration.

For a policy `π`:

$$
H(\pi)=-Ea\sim\pi[log\pi(a∣s)].
$$

For the multi-head asset policy:

$$
Hasset=HBHS+HAD+Hmemory+Hfraction.
$$

Similarly, the domain policy entropy is:

$$
Hdomain=Hallocation+HDM+Hmemory.
$$

The master policy entropy is:

$$
Hmaster=Hallocation+Hmemory.
$$

---

# 17. Total PPO Loss

The total optimization objective combines the actor, critic, and entropy terms:

$$
L=Lactor+c1Lvalue-c2H
$$

where:

$$
c1=0.5
$$

is the value-loss coefficient and

$$
c2=0.01
$$

is the entropy coefficient.

The negative entropy term means that increasing entropy decreases the optimization loss, encouraging exploration.

The implementation uses these same coefficients for the asset, domain, and master agents.

---

# 18. Gradient Stabilization

After computing the total loss, gradients are backpropagated through the shared actor-critic architecture.

Gradient clipping is applied:

$$
\|∇\thetaL\|2\le0.5
$$

by clipping the gradient norm to a maximum of `0.5`.

The optimizer is Adam with a default learning rate of:

$$
\alpha=3\times10-4.
$$

---

# 19. Hierarchical Portfolio Decision

The overall decision process can be summarized mathematically as:

### Asset level

$$
Xi\to hiasset\to {BUY/HOLD/SELLTrade fractionMemory updateAsset\to Domain signal
$$

### Domain level

$$
{hiasset}\to AttentionPool\to hdomain\to {Asset allocationsDomain\to Master signalMemory update
$$

### Master level

$$
{hjdomain}\to AttentionPool\to hmaster\to {Domain allocationsMemory update
$$

Therefore:

$$
Assets\to Domains\to Master
$$

for information aggregation, while:

$$
Master\to Domains\to Assets
$$

provides hierarchical allocation/control signals.

---

# 20. Anti-Fragility Objective

The asset-level reward is designed to emphasize profitability during periods of high volatility.

The current conceptual objective is to reward returns generated under volatile market conditions rather than simply maximizing raw returns.

For a portfolio return `Rt`, the framework can be expressed conceptually as:

$$
rtportfolio=log(PtPt-1)
$$

with the reward incorporating a volatility-dependent component:

$$
rt=f(rtportfolio,\sigmatmarket).
$$

The intention is to encourage policies that can exploit or remain resilient during high-volatility regimes, forming the basis of the project's **anti-fragility** objective.

---

# 21. Parameter Sharing

Asset agents use parameter sharing.

Instead of learning independent networks:

$$
\theta1,\theta2,…,\thetaN
$$

for `N` assets, the agents share a common parameter set:

$$
\thetaasset,1=\thetaasset,2=⋯=\thetaasset,N=\thetashared
$$

while receiving different asset-specific observations.

This allows the same policy to learn generalizable patterns across different assets.

The same principle is used for domain-level agents.

---

# 22. Complete Hierarchical Objective

Conceptually, SwanField therefore optimizes a hierarchy of stochastic policies:

$$
\pi={\piasset,\pidomain,\pimaster}.
$$

The complete system can be viewed as:

$$
Market Data\to Asset Policies\to Domain Policies\to Master Policy\to Portfolio Allocation
$$

while feedback flows in the opposite direction:

$$
Master Signal\to Domain Policies\to Asset Policies
$$

The resulting architecture combines:

- hierarchical multi-agent reinforcement learning,
- PPO actor-critic optimization,
- recurrent sequence modelling,
- attention-based hierarchical aggregation,
- stochastic continuous policies,
- Dirichlet portfolio allocation,
- learned memory,
- parameter sharing,
- and volatility-aware reward design.


