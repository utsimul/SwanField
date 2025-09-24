import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from helpernets import attention_pool

class DomainPolicyNet(nn.Module):

    def __init__(
            self,
            num_assets : int,
            h_asset_dim : int,
            master_signal_dim : int,
            memory_dim : int = 1,
            num_signal : int = 2, #signal sent from domain to master
            min_std : float = 1e-3,
            hidden_dim : int = 128,
    ):
        super().__init__()
        self.attentionpool = attention_pool(num_assets, h_asset_dim)

        self.shared_net = nn.Linear(h_asset_dim + memory_dim + master_signal_dim, hidden_dim)
        self.actor_port_alloc = nn.Linear(hidden_dim, num_assets + 1) #+1 for "cash" token for unallocated cash

        self.domain_to_master_mean = nn.Linear(hidden_dim, num_signal)
        self.domain_to_master_std = nn.Linear(hidden_dim, num_signal)


        self.mem_update_mean = nn.Linear(hidden_dim, memory_dim) #actor head 3: memory update value
        self.mem_update_std = nn.Linear(hidden_dim, memory_dim)

        self.critic = nn.Linear(hidden_dim, 1) #outputs value function
        self.min_std = min_std

    def forward(self, h_assets, master_signal, mem):

        h_sector, alphas = self.attentionpool(h_assets)
        net_x = torch.cat([h_sector, master_signal, mem], dim=1)
        h = F.relu(self.shared_net(net_x))

        #ASSET ALLOCATION (USE DIRICHLET DISTRIBUTION):
        logits = self.actor_port_alloc(h)
        # logits -> concentration parameters for Dirichlet
        raw_alpha = self.actor_port_alloc(h)                   # (batch, num_assets+1)
        alpha = F.softplus(raw_alpha) + 1e-3                   # ensure positivity
        alloc_distn = torch.distributions.Dirichlet(alpha)

        #MEMORY UPDATE SIGNAL
        mem_mean = self.mem_update_mean(h) #gives 2 means
        mem_logstd = self.mem_update_std(h).clamp(min=torch.log(torch.tensor(self.min_std))) #gives 2 log std
        mem_std = mem_logstd.exp() #calculate logstd to ensure that values are positive
        mem_update = Normal(mem_mean, mem_std)

        #DOMAIN TO MASTER SIGNAL
        dtom_mean = self.domain_to_master_mean(h)
        dtom_logstd = self.domain_to_master_std(h).clamp(min=torch.log(torch.tensor(self.min_std)))
        dtom_std = dtom_logstd.exp()
        dtom = Normal(dtom_mean, dtom_std)

        value = self.critic(h).squeeze(-1)

        return alloc_distn, mem_update, dtom, value

class DomainAgent(nn.Module):

    def __init__(
            self,
            num_assets : int,
            h_asset_dim : int,
            master_signal_dim : int,
            memory_dim : int = 1,
            num_signal : int = 2,
            min_std : float = 1e-3,
            hidden_dim : int = 128,
            lr = 3e-4,
            clip_eps = 0.2,
            c1 = 0.5, #value coefficient (multiplied with L^{value})
            c2 = 0.01, #entropy coefficient (multiplied with entropy of action distributions)
            device = "cpu"
    ):
        
        super().__init__()
        self.device = device
        self.policynet = DomainPolicyNet(
            num_assets = num_assets,
            h_asset_dim = h_asset_dim,
            master_signal_dim = master_signal_dim,
            memory_dim = memory_dim,
            num_signal = num_signal,
            min_std = min_std,
            hidden_dim = hidden_dim
        ).to(device)

        self.optimizer = optim.Adam(self.policynet.parameters(), lr=lr)

        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2

        #Initialize memory variables:
        self.memory_dim = memory_dim
        self.memory = torch.zeros(1, memory_dim, device=device)  # (batch=1, memory_dim)


    def act(self, h_assets, master_signal, mem):

        h_assets = h_assets.to(self.device)
        master_signal = master_signal.to(self.device)
        alloc_distn, mem_update_dist, dtom_dist, value = self.policynet(h_assets, master_signal, mem)

        #1. SAMPLING FROM DISTRIBUTIONS
        dtom = dtom_dist.sample()
        mem_update = mem_update_dist.sample()
        allocations = alloc_distn.rsample()                    # (batch, num_assets+1)


        #2. CALCULATING LOG PROBS
        alloc_log_prob = alloc_distn.log_prob(allocations)
        entropy = alloc_distn.entropy()
        dtom_logprob = dtom_dist.log_prob(dtom).sum(-1)
        mem_update_logprob = mem_update_dist.logprob(mem_update).sum(-1)

        total_logprob = alloc_log_prob + dtom_logprob + mem_update_logprob

        return (allocations, dtom, mem_update), total_logprob, value, alloc_distn

    def evaluate(self, h_assets, master_signal, mem, actions):

        #recompute values

        h_assets = h_assets.to(self.device)
        master_signal = master_signal.to(self.device)
        alloc_distn, mem_update_dist, dtom_dist, value = self.policynet(h_assets, master_signal, mem)

        alloc, dtom, mem = actions
        alloc_p = alloc_distn.log_prob(alloc)
        dtom_p = dtom_dist.log_prob(dtom).sum(-1)
        mem_upd_p = mem_update_dist.log_prob(mem).sum(-1)

        logprob = alloc_p + dtom_p + mem_upd_p

        alloc_entropy = alloc_distn.entropy()                # (batch,)
        dtom_entropy = dtom_dist.entropy().sum(-1)           # (batch,)
        mem_entropy = mem_update_dist.entropy().sum(-1) 

        entropy = alloc_entropy + dtom_entropy + mem_entropy
        return logprob, entropy, value

    def update_policy(self, rollouts):

        """
            rollouts should be a dict with:
            - h_assets
            - domain memory
            - master agent allocation
            - actions = (a1, a2, a3)
            - old_logprobs
            - returns
            - advantages

            this is basically all the data that we stored for the number of episodes during which we 
            had freezed policy updation. Now we review our calculations.
            """
        
        h_assets = rollouts["h_assets"].to(self.device)
        domain_mem = rollouts["domain_memory"].to(self.device)
        master_alloc = rollouts["master_alloc"].to(self.device)
        actions = rollouts["actions"]
        old_logprobs = rollouts["old_logprobs"].to(self.device)
        returns = rollouts["returns"].to(self.device)
        advantages = rollouts["advantages"].to(self.device)
        #returns and advantages calculated in train.py file in training loop

        logprobs, entropy, values = self.evaluate(h_assets, master_alloc, domain_mem, actions)
        ratios = torch.exp(logprobs - old_logprobs)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        value_loss = (returns - values).pow(2).mean()
        loss = actor_loss + self.c1 * value_loss - self.c2* entropy

        #magic statements
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policynet.parameters(), max_norm=0.5)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }