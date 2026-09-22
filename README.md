# SwanField — Hierarchical Multi-Agent Reinforcement Learning for Portfolio Management

SwanField is a **hierarchical multi-agent deep reinforcement learning framework for portfolio management**.

The framework consists of three levels of decision-making agents:

1. **Asset-level agents** — specialize in individual assets.

2. **Domain-level agents** — aggregate information across related assets/sectors.

3. **Master agent** — aggregates domain-level information and determines the overall portfolio allocation.

The architecture is inspired by how a hierarchical portfolio-management organization could operate: specialized asset managers communicate with domain/sector managers, who in turn communicate with a master portfolio manager.

The agents use **Proximal Policy Optimization (PPO)** with actor-critic architectures and stochastic policies.

---

## Architecture

The information flow is:

![Architecture](Prototype-architecture.png)

---

# 1. Asset-Level Agents

Each asset agent receives:

* sequential asset data,

* non-sequential information such as allocation/memory information,

* and produces four stochastic actions.

The asset policy consists of:

1. **BUY / HOLD / SELL**

2. **Signal sent to the domain agent**

3. **Memory update**

4. **Trade fraction**

The implementation uses an LSTM/GRU-based sequence encoder followed by a shared representation and multiple actor heads. The sequence encoder produces the final recurrent hidden representation `hseq`.

---

## 1.1 Sequential Encoding

Let the sequential input for an asset be

\(X_t = \{x_{t-T+1}, \ldots, x_t\}\)

where `T` is the sequence length.

The recurrent encoder produces

\(h_{\mathrm{seq},t} = \operatorname{RNN}(X_t)\)

where the RNN can be an LSTM or GRU.

For an LSTM:

\((h_t, c_t) = \operatorname{LSTM}(x_t, h_{t-1}, c_{t-1})\)

and the final hidden state is used as the sequence representation:

\(h_{\mathrm{seq}} = h_T\)

In the implementation, the final recurrent hidden state is returned as the asset's sequence encoding.

---

## 1.2 Shared Asset Representation

The sequential representation is concatenated with non-sequential information:

\(x_t^{\mathrm{asset}} = [h_{\mathrm{seq},t} \,\|\, x_t^{\mathrm{nonseq}}]\)

where `\|\|` denotes concatenation.

The shared hidden representation is

\(h_t = \tanh\!\left(W_h x_t^{\mathrm{asset}} + b_h\right)\)

In the implementation, this shared representation is used by all actor heads and the critic.

---

# 2. Asset Policy

The asset policy is a joint stochastic policy:

\(\pi_\theta(a_t \mid s_t) = \pi_\theta^{\mathrm{BHS}}\pi_\theta^{\mathrm{signal}}\pi_\theta^{\mathrm{memory}}\pi_\theta^{\mathrm{fraction}}\)

The complete action is

\(a_t = (a_t^{\mathrm{BHS}}, a_t^{\mathrm{signal}}, a_t^{\mathrm{memory}}, a_t^{\mathrm{fraction}})\)

---

## 2.1 BUY / HOLD / SELL Head

The first actor head produces three logits:

\(z_t = W_{\mathrm{BHS}}h_t + b_{\mathrm{BHS}}\)

where

\(z_t \in \mathbb{R}^3\)

The logits are converted into probabilities using softmax:

\(\pi_\theta^{\mathrm{BHS}}(a\mid s_t) = \frac{e^{z_{t,a}}}{\sum_{j=1}^{3}e^{z_{t,j}}}\)

Therefore,

\(a_t^{\mathrm{BHS}} \sim \operatorname{Categorical}(\pi_\theta^{\mathrm{BHS}})\)

The three actions are:

\(A_{\mathrm{BHS}} = \{\mathrm{BUY},\mathrm{HOLD},\mathrm{SELL}\}\)

The implementation constructs this distribution directly from the actor logits.

---

## 2.2 Asset → Domain Signal

The asset agent also generates a continuous signal for the domain agent.

The mean is

\(\mu_t^{AD} = W_\mu^{AD}h_t + b_\mu^{AD}\)

and the raw log-standard-deviation output is

\(\ell_t^{AD} = W_\sigma^{AD}h_t + b_\sigma^{AD}\)

The implementation clamps this value to maintain a minimum standard deviation:

\(\tilde{\ell}_t^{AD} = \max\!\left(\ell_t^{AD}, \log(\sigma_{\min})\right)\)

The standard deviation is then

\(\sigma_t^{AD} = \exp\!\left(\tilde{\ell}_t^{AD}\right)\)

Therefore the asset-to-domain signal is sampled from

\(a_t^{AD} \sim \mathcal{N}\!\left(\mu_t^{AD},(\sigma_t^{AD})^2\right)\)

For `d` signal dimensions, this represents `d` independent Gaussian variables.

The implementation uses `Normal(mean, std)` for this actor head.

---

## 2.3 Memory Update

The memory head generates a continuous memory update:

\(\mu_t^M = W_\mu^M h_t + b_\mu^M\)

\(\ell_t^M = W_\sigma^M h_t + b_\sigma^M\)

\(\sigma_t^M = \exp\!\left(\max\!\left(\ell_t^M,\log\sigma_{\min}\right)\right)\)

The memory update is sampled as

\(a_t^M \sim \mathcal{N}\!\left(\mu_t^M,(\sigma_t^M)^2\right)\)

Thus, the agent can maintain a learned continuous internal state across timesteps.

The implementation contains separate mean and standard-deviation heads for this distribution.

---

## 2.4 Trade Fraction

The trade-fraction head determines what fraction of the available position/capital should be involved in the trade.

The mean is constrained to `[0,1]`:

\(\mu_t^F = \sigma\!\left(W_\mu^Fh_t+b_\mu^F\right)\)

where

\(\sigma(x) = \frac{1}{1+e^{-x}}\)

The standard deviation is

\(\sigma_t^F = \exp\!\left(\max\!\left(\ell_t^F,\log\sigma_{\min}\right)\right)\)

The trade fraction is sampled from

\(a_t^F \sim \mathcal{N}\!\left(\mu_t^F,(\sigma_t^F)^2\right)\)

The sampled value is finally clipped to the valid interval:

\(a_t^F = \operatorname{clip}(a_t^F,0,1)\)

This corresponds to the implementation's sigmoid mean followed by sampling and clipping.

---

# 3. Joint Asset Log-Probability

Because the asset action consists of multiple stochastic components, the total log-probability is the sum of the component log-probabilities:

\(\log\pi_\theta(a_t\mid s_t) = \log\pi^{\mathrm{BHS}}+\log\pi^{AD}+\log\pi^M+\log\pi^F\)

For the Gaussian components,

\(\log\pi^{AD} = \sum_i \log\mathcal{N}\!\left(a_{t,i}^{AD};\mu_{t,i}^{AD},(\sigma_{t,i}^{AD})^2\right)\)

and similarly for memory and trade fraction.

Therefore:

\(\log\pi_\theta(a_t\mid s_t) = \log\pi^{\mathrm{BHS}}+\sum_i\log\pi_i^{AD}+\sum_i\log\pi_i^M+\sum_i\log\pi_i^F\)

The implementation explicitly constructs this sum before storing the PPO log-probability.

---

# 4. Asset Critic

The critic shares the common hidden representation `h_t` with the actor heads.

It estimates the state value:

\(V_\phi(s_t) = W_Vh_t+b_V\)

Thus,

\(V_\phi(s_t) \approx \mathbb{E}\!\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid s_t\right]\)

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

\(h_i^{\mathrm{asset}}\)

which is provided to the domain agent.

The domain agent uses an **attention pooling mechanism** to aggregate these asset representations:

\(H_{\mathrm{assets}} = \{h_1,\ldots,h_N\}\)

The attention mechanism produces:

\(h^{\mathrm{domain}} = \operatorname{AttentionPool}(H_{\mathrm{assets}})\)

This creates a fixed-size domain representation regardless of the number/order of individual asset representations.

---

## 5.1 Domain Input

The domain agent receives three sources of information:

\(x_t^D = [h^{\mathrm{domain}}\|\ m_t^D\|\ s_t^{M\to D}]\)

where:

* `h_domain` = pooled asset representation,

* `m_t^D` = domain memory,

* `s_t^{M→D}` = signal from the master agent.

The shared domain representation is then

\(h_t^D = \operatorname{ReLU}(W_Dx_t^D+b_D)\)

This corresponds to concatenating the attention-pooled asset representation, master signal, and memory before the shared network.

---

# 6. Domain-Level Actions

The domain agent produces three outputs:

1. **Asset allocation**

2. **Domain → Master signal**

3. **Memory update**

---

## 6.1 Asset Allocation with a Dirichlet Policy

Instead of independently predicting each asset's allocation, the domain agent produces the concentration parameters of a **Dirichlet distribution**.

The network first produces raw outputs:

\(z_t^D = W_{\mathrm{alloc}}h_t^D+b_{\mathrm{alloc}}\)

These are converted into positive concentration parameters:

\(\alpha_t = \operatorname{softplus}(z_t^D)+\epsilon\)

where

\(\alpha_{t,i}>0\)

The allocation vector is then sampled as

\(a_t^D \sim \operatorname{Dirichlet}(\alpha_t)\)

Therefore,

\(a_t^D = [a_{t,1},a_{t,2},\ldots,a_{t,N},a_{t,\mathrm{cash}}]\)

with

\(a_{t,i}\ge 0\)

and

\(\sum_i a_{t,i}=1\)

This naturally represents portfolio allocation because the outputs form a probability-simplex vector.

The implementation uses `softplus(raw_alpha) + 1e-3` to ensure positive Dirichlet concentration parameters.

---

## 6.2 Domain → Master Signal

The domain agent produces a continuous signal to the master agent:

\(\mu_t^{DM}=W_\mu^{DM}h_t^D+b_\mu^{DM}\)

\(\sigma_t^{DM}=\exp\!\left(\max\!\left(W_\sigma^{DM}h_t^D+b_\sigma^{DM},\log\sigma_{\min}\right)\right)\)

The signal is sampled as

\(a_t^{DM}\sim\mathcal{N}\!\left(\mu_t^{DM},(\sigma_t^{DM})^2\right)\)

Thus the master agent receives learned stochastic signals from each domain agent.

---

## 6.3 Domain Memory Update

Similarly,

\(\mu_t^{M,D}=W_\mu^{M,D}h_t^D+b_\mu^{M,D}\)

and

\(\sigma_t^{M,D}=\exp\!\left(\max\!\left(W_\sigma^{M,D}h_t^D+b_\sigma^{M,D},\log\sigma_{\min}\right)\right)\)

Then

\(m_{t+1}^D\sim\mathcal{N}\!\left(\mu_t^{M,D},(\sigma_t^{M,D})^2\right)\)

---

# 7. Domain Critic

The domain critic estimates:

\(V_\phi^D(s_t^D)=W_V^Dh_t^D+b_V^D\)

Therefore the domain agent has its own value function describing the expected future return from the domain's current state.

---

# 8. Domain Joint Policy

The domain action consists of:

\(a_t^D=(a_t^{\mathrm{alloc}},a_t^{DM},a_t^{M,D})\)

Its joint policy is therefore:

\(\pi_\theta^D=\pi_\theta^{\mathrm{alloc}}\pi_\theta^{DM}\pi_\theta^{M,D}\)

The joint log-probability is

\(\log\pi_\theta^D(a_t\mid s_t)=\log\pi_\theta^{\mathrm{alloc}}+\log\pi_\theta^{DM}+\log\pi_\theta^{M,D}\)

which is exactly the quantity used to construct the PPO probability ratio.

---

# 9. Master Agent

The master agent sits at the highest level of the hierarchy.

It receives the representations generated by the domain agents:

\(H^D=\{h_1^D,h_2^D,\ldots,h_K^D\}\)

These are aggregated using attention pooling:

\(h_t^M=\operatorname{AttentionPool}(H^D)\)

The master representation is then combined with master memory:

\(x_t^M=[h_t^M\|\ m_t^M]\)

The shared master representation is

\(h_t=\operatorname{ReLU}(W_Mx_t^M+b_M)\)

---

# 10. Master Portfolio Allocation

The master agent produces allocations across domains plus a cash allocation.

Raw allocation outputs are:

\(z_t^M=W_{\mathrm{alloc}}^Mh_t^M+b_{\mathrm{alloc}}^M\)

These are transformed into Dirichlet concentration parameters:

\(\alpha_t^M=\operatorname{softplus}(z_t^M)+\epsilon\)

The master allocation is sampled from:

\(a_t^M\sim\operatorname{Dirichlet}(\alpha_t^M)\)

where

\(a_t^M=[a_{t,1}^{\mathrm{domain}},\ldots,a_{t,K}^{\mathrm{domain}},a_{t,\mathrm{cash}}]\)

and

\(\sum_i a_{t,i}^M=1\)

Thus, the master agent determines the high-level allocation across domains while the domain agents determine allocations within those domains.

---

# 11. Master Memory

The master memory update is represented by a Gaussian policy:

\(\mu_t^M=W_\mu^Mh_t+b_\mu^M\)

\(\sigma_t^M=\exp\!\left(\max\!\left(W_\sigma^Mh_t+b_\sigma^M,\log\sigma_{\min}\right)\right)\)

The new memory is sampled as

\(m_{t+1}^M\sim\mathcal{N}\!\left(\mu_t^M,(\sigma_t^M)^2\right)\)

---

# 12. Master Critic

The master critic estimates the value of the global portfolio state:

\(V_\phi^M(s_t)=W_V^Mh_t+b_V^M\)

This provides the value estimate used to calculate the PPO advantage and critic loss.

---

# 13. PPO Training

All three levels use **Proximal Policy Optimization (PPO)**.

For a transition `t`, let:

\(r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\)

Using log-probabilities:

\(r_t(\theta)=\exp\!\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)\right)\)

This is the exact probability ratio used by the implementation.

---

## 13.1 Advantage

The PPO actor uses an estimated advantage:

\(A_t=R^t-V_\phi(s_t)\)

where `R^t` is the estimated return.

The advantage measures whether the selected action performed better or worse than expected under the critic's estimate.

---

# 14. PPO Clipped Objective

The two surrogate objectives are:

\(L_t^{\mathrm{CLIP}}=r_t(\theta)A_t\)

and

\(\tilde L_t^{\mathrm{CLIP}}=\operatorname{clip}\!\left(r_t(\theta),1-\epsilon,1+\epsilon\right)A_t\)

The PPO objective is:

\(L^{\mathrm{CLIP}}=\mathbb{E}_t\!\left[\min\!\left(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]\)

where the implementation uses

\(\epsilon=0.2\)

The actor loss is the negative of this objective:

\(L_{\mathrm{actor}}=-\mathbb{E}_t\!\left[\min\!\left(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]\)

This clipping prevents the updated policy from moving excessively far from the old policy.

---

# 15. Critic Loss

The critic is trained using mean squared error between the estimated return and value prediction:

\(L_{\mathrm{value}}=\mathbb{E}_t\!\left[(R^t-V_\phi(s_t))^2\right]\)

The implementation uses this directly as:

\((return-value)^2\)

---

# 16. Entropy Regularization

Entropy encourages the policy to retain exploration.

For a policy `π`:

\(H(\pi)=-\mathbb{E}_{a\sim\pi}\!\left[\log\pi(a\mid s)\right]\)

For the multi-head asset policy:

\(H_{\mathrm{asset}}=H_{\mathrm{BHS}}+H_{AD}+H_{\mathrm{memory}}+H_{\mathrm{fraction}}\)

Similarly, the domain policy entropy is:

\(H_{\mathrm{domain}}=H_{\mathrm{allocation}}+H_{DM}+H_{\mathrm{memory}}\)

The master policy entropy is:

\(H_{\mathrm{master}}=H_{\mathrm{allocation}}+H_{\mathrm{memory}}\)

---

# 17. Total PPO Loss

The total optimization objective combines the actor, critic, and entropy terms:

\(L=L_{\mathrm{actor}}+c_1L_{\mathrm{value}}-c_2H\)

where:

\(c_1=0.5\)

is the value-loss coefficient and

\(c_2=0.01\)

is the entropy coefficient.

The negative entropy term means that increasing entropy decreases the optimization loss, encouraging exploration.

The implementation uses these same coefficients for the asset, domain, and master agents.

---

# 18. Gradient Stabilization

After computing the total loss, gradients are backpropagated through the shared actor-critic architecture.

Gradient clipping is applied:

\(\lVert\nabla_\theta L\rVert_2\le 0.5\)

by clipping the gradient norm to a maximum of `0.5`.

The optimizer is Adam with a default learning rate of:

\(\alpha=3\times10^{-4}\)

---

# 19. Hierarchical Portfolio Decision

The overall decision process can be summarized mathematically as:

**### Asset level**

\(X_i\to h_i^{\mathrm{asset}}\to\{\mathrm{BUY/HOLD/SELL},\ \mathrm{Trade\ fraction},\ \mathrm{Memory\ update},\ \mathrm{Asset\to Domain\ signal}\}\)

**### Domain level**

\(\{h_i^{\mathrm{asset}}\}\to\operatorname{AttentionPool}\to h^{\mathrm{domain}}\to\{\mathrm{Asset\ allocations},\ \mathrm{Domain\to Master\ signal},\ \mathrm{Memory\ update}\}\)

**### Master level**

\(\{h_j^{\mathrm{domain}}\}\to\operatorname{AttentionPool}\to h^{\mathrm{master}}\to\{\mathrm{Domain\ allocations},\ \mathrm{Memory\ update}\}\)

Therefore:

\(\mathrm{Assets}\to\mathrm{Domains}\to\mathrm{Master}\)

for information aggregation, while:

\(\mathrm{Master}\to\mathrm{Domains}\to\mathrm{Assets}\)

provides hierarchical allocation/control signals.

---

# 20. Anti-Fragility Objective

The asset-level reward is designed to emphasize profitability during periods of high volatility.

The current conceptual objective is to reward returns generated under volatile market conditions rather than simply maximizing raw returns.

For a portfolio return `R_t`, the framework can be expressed conceptually as:

\(r_t^{\mathrm{portfolio}}=\log\!\left(\frac{P_t}{P_{t-1}}\right)\)

with the reward incorporating a volatility-dependent component:

\(r_t=f\!\left(r_t^{\mathrm{portfolio}},\sigma_t^{\mathrm{market}}\right)\)

The intention is to encourage policies that can exploit or remain resilient during high-volatility regimes, forming the basis of the project's **anti-fragility** objective.

---

# 21. Parameter Sharing

Asset agents use parameter sharing.

Instead of learning independent networks:

\(\theta_1,\theta_2,\ldots,\theta_N\)

for `N` assets, the agents share a common parameter set:

\(\theta_{\mathrm{asset},1}=\theta_{\mathrm{asset},2}=\cdots=\theta_{\mathrm{asset},N}=\theta_{\mathrm{shared}}\)

while receiving different asset-specific observations.

This allows the same policy to learn generalizable patterns across different assets.

The same principle is used for domain-level agents.

---

# 22. Complete Hierarchical Objective

Conceptually, SwanField therefore optimizes a hierarchy of stochastic policies:

\(\pi=\{\pi_{\mathrm{asset}},\pi_{\mathrm{domain}},\pi_{\mathrm{master}}\}\)

The complete system can be viewed as:

\(\mathrm{Market\ Data}\to\mathrm{Asset\ Policies}\to\mathrm{Domain\ Policies}\to\mathrm{Master\ Policy}\to\mathrm{Portfolio\ Allocation}\)

while feedback flows in the opposite direction:

\(\mathrm{Master\ Signal}\to\mathrm{Domain\ Policies}\to\mathrm{Asset\ Policies}\)

The resulting architecture combines:

* hierarchical multi-agent reinforcement learning,

* PPO actor-critic optimization,

* recurrent sequence modelling,

* attention-based hierarchical aggregation,

* stochastic continuous policies,

* Dirichlet portfolio allocation,

* learned memory,

* parameter sharing,

* and volatility-aware reward design.
