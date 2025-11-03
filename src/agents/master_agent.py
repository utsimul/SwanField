import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
#from helpernets import attention_pool
#MASTER AGENT WILL USE AN INSTANCE OF THE SAME CLASS AS DOMAIN ATTENTION POOL FOR POOLING

class MasterPolicyNet(nn.Module):

    def __init__(
           self,
           num_domains : int,
           h_domain_dim : int,
           memory_dim : int = 1,
           min_std : float = 1e-3,
           hidden_dim : int = 128,

    ):
        
        super().__init__()
        self.attentionpool = attention_pool(h_domain_dim)
        
        self.shared_net = nn.Linear(h_domain_dim + memory_dim, hidden_dim)
        self.actor_port_alloc = nn.Linear(hidden_dim, num_domains + 1) #+1 for "cash"

        self.mem_update_mean = nn.Linear(hidden_dim, memory_dim) #actor head 3: memory update value
        self.mem_update_std = nn.Linear(hidden_dim, memory_dim)

        self.critic = nn.Linear(hidden_dim, 1) #outputs value function
        self.min_std = min_std

    def forward(self, h_domains, mem):

        if isinstance(mem, float):
            #convert to tensor with batch as first dimension.
            mem = torch.tensor([mem], dtype=torch.float32)
            mem = mem.unsqueeze(0) #because batch dimension will always be 1

        h_master, alphas = self.attentionpool(h_domains) #(batch, D) => h_master
        net_x = torch.cat([h_master, mem], dim=1)
        h = F.relu(self.shared_net(net_x))

        #DOMAIN ALLOCATION (USE DIRICHLET DISTRIBUTION):
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

        value = self.critic(h).squeeze(-1)

        return alloc_distn, mem_update, value
    
class MasterAgent(nn.Module):

    def __init__(
            self,
            num_domains : int,
            h_domain_dim : int,
            memory_dim : int = 1,
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
        self.policynet = MasterPolicyNet(
            num_domains = num_domains,
            h_domain_dim= h_domain_dim,
            memory_dim=memory_dim,
            min_std=min_std,
            hidden_dim=hidden_dim
        ).to(device)

        self.optimizer = optim.Adam(self.policynet.parameters(), lr=lr)

        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2

        #Initialize memory variables:
        self.memory_dim = memory_dim
        self.memory = torch.zeros(1, memory_dim, device=device)  # (batch=1, memory_dim)
    
    def act(self, h_domains, mem):

        h_domains = h_domains.to(self.device)
        alloc_distn, mem_update_dist, value = self.policynet(h_domains,mem)

        #1. SAMPLING FROM DISTRIBUTIONS
        mem_update = mem_update_dist.sample()
        allocations = alloc_distn.rsample() 

        #2. CALCULATING LOG PROBS
        alloc_log_prob = alloc_distn.log_prob(allocations)
        entropy = alloc_distn.entropy()
        mem_update_logprob = mem_update_dist.log_prob(mem_update).sum(-1)

        total_logprob = alloc_log_prob + mem_update_logprob

        return (allocations, mem_update), total_logprob, value, alloc_distn
    
    def evaluate(self, h_domains, mem, actions):

        h_domains = h_domains.to(self.device)

        alloc_distn, mem_update_dist, value = self.policynet(h_domains,mem)

        alloc, mem = actions
        alloc_p = alloc_distn.log_prob(alloc)
        mem_upd_p = mem_update_dist.log_prob(mem).sum(-1)

        logprob = alloc_p + mem_upd_p

        alloc_entropy = alloc_distn.entropy()                # (batch,)
        mem_entropy = mem_update_dist.entropy().sum(-1) 

        entropy = alloc_entropy + mem_entropy
        return logprob, entropy, value

    def update_policy(self, rollouts):

        """
            rollouts should be a dict with:
            - h_domains
            - master memory
            - actions = (a1, a2)
            - old_logprobs
            - returns
            - advantages

            this is basically all the data that we stored for the number of episodes during which we 
            had freezed policy updation. Now we review our calculations.
            """
        
        h_domains = rollouts["h_domains"].to(self.device)
        master_mem = rollouts["master_memory"].to(self.device)
        actions = rollouts["actions"]
        old_logprobs = rollouts["old_logprobs"].to(self.device)
        returns = rollouts["returns"].to(self.device)
        advantages = rollouts["advantages"].to(self.device)
        #returns and advantages calculated in train.py file in training loop

        logprobs, entropy, values = self.evaluate(h_domains, master_mem, actions)
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