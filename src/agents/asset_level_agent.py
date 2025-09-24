import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

#Dont forget:
# - CONCAT MEMORY ONTO NON SEQ DATA either in train.py or here


class Asset_Seq_Encoder(nn.Module):

    def __init__(
            self,
            seq_input_dim : int,
            recurrent_layers : int = 1,
            network : str = "lstm",
            recurrent_neurons : int = 64,
            
    ):
        super().__init__()
        #Experiment with LSTM, 1D CNN, GRU
        if(network == "lstm"):
            self.network = nn.LSTM(seq_input_dim, recurrent_neurons, recurrent_layers, batch_first = True)
        elif(network == "gru"):
            self.network = nn.GRU(seq_input_dim, recurrent_neurons, recurrent_layers, batch_first = True)
        else:
            print("Encoder choice for sequence data doesn't match.")
    
    def forward(self, seq_data : torch.Tensor):

        _, (h_n, _) = self.network(seq_data)
        return h_n[-1] #(last hidden state)


class AssetPolicyNet(nn.Module):

    def __init__(
            self, 
            seq_input_dim: int, 
            non_seq_input_dim : int, 
            hidden_dim_seq : int=64, 
            hidden_dim : int = 128, 
            num_discrete: int = 3, #BUY, SELL, HOLD
            memory_dim : int=1, #dimensions of memory update output
            num_signal : int = 2, #signal from asset to domain
            min_std : float = 1e-3, #Minimum standard deviation for the Normal distribution of the memory update.
            ):
        super().__init__()
        self.encoder = Asset_Seq_Encoder(seq_input_dim, hidden_dim_seq)
        self.shared_net = nn.Linear(hidden_dim_seq + non_seq_input_dim, hidden_dim)
        self.actor_bhs = nn.Linear(hidden_dim, num_discrete) #actor head 1: BUY HOLD SELL

        #fraction of trades made 
        self.trade_frac_mean = nn.Linear(hidden_dim, 1)
        self.trade_frac_std = nn.Linear(hidden_dim, 1)

        self.asset_to_domain_mean = nn.Linear(hidden_dim, num_signal) #actor head 2: signal to domain agent
        self.asset_to_domain_std = nn.Linear(hidden_dim, num_signal)
        #if num_signals are 2, means we want the policy to determine 2 values. This means we create 2 gaussian 
        #distributions - one for each of the output variables (therefore 2 means and 2 std devs), and pick one value
        #from each distribution

        self.mem_update_mean = nn.Linear(hidden_dim, memory_dim) #actor head 3: memory update value
        self.mem_update_std = nn.Linear(hidden_dim, memory_dim)

        self.critic = nn.Linear(hidden_dim, 1) #critic outputs the estimated value
        self.min_std = min_std

    
    def forward(self, seq_data, non_seq_data):
        seq_encoding = self.encoder(seq_data)
        net_x = torch.cat([seq_encoding, non_seq_data], dim=1)
        net_enc_hidden = torch.tanh(self.shared_net(net_x))

        #BUY HOLD SELL (categorical policy):
        logits = self.actor_bhs(net_enc_hidden)
        bhs = Categorical(logits = logits)

        #ASSET TO DOMAIN SIGNAL
        ast_to_dom_mean = self.asset_to_domain_mean(net_enc_hidden) #gives 2 (num_signal) means
        ast_do_dom_logstd = self.asset_to_domain_std(net_enc_hidden).clamp(min=torch.log(torch.tensor(self.min_std))) #gives 2 logstds
        ast_to_dom_std = ast_do_dom_logstd.exp() #calculate logstd to ensure that values are positive
        ast_to_dom = Normal(ast_to_dom_mean, ast_to_dom_std) #outputs a Gaussian distribution 
        #every time the actor processes data, it calculates a new policy - a new distribution and then we sample the action
        #from that distribution, encouraging exploration and exploitation based on a distribution rather than strict epsilon
        #greedy percentages

        #MEMORY UPDATE SIGNAL
        mem_mean = self.mem_update_mean(net_enc_hidden) #gives 2 means
        mem_logstd = self.mem_update_std(net_enc_hidden).clamp(min=torch.log(torch.tensor(self.min_std))) #gives 2 log std
        mem_std = mem_logstd.exp() #calculate logstd to ensure that values are positive
        mem_update = Normal(mem_mean, mem_std)

        # TRADE FRACTION HEAD (continuous between 0 and 1)
        frac_mean = torch.sigmoid(self.trade_frac_mean(net_enc_hidden))  # ensures [0,1]
        frac_logstd = self.trade_frac_std(net_enc_hidden).clamp(min=torch.log(torch.tensor(self.min_std)))
        frac_std = frac_logstd.exp()
        trade_frac = Normal(frac_mean, frac_std)

    
        value = self.critic(net_enc_hidden).squeeze(-1)
    
        return bhs, ast_to_dom, mem_update, trade_frac, value


class AssetAgent(nn.Module):


    def __init__(
            self,
            seq_input_dim,
            non_seq_input_dim,
            hidden_dim_seq = 64,
            hidden_dim = 128,
            num_discrete = 3,
            memory_dim = 1,
            num_signal = 2, #signal sent from asset to domain
            min_std = 1e-3,
            lr = 3e-4,
            clip_eps = 0.2,
            c1 = 0.5, #value coefficient (multiplied with L^{value})
            c2 = 0.01, #entropy coefficient (multiplied with entropy of action distributions)
            device = "cpu"
    ):
        super().__init__()
        self.device = device
        self.policynet = AssetPolicyNet(
            seq_input_dim=seq_input_dim,
            non_seq_input_dim=non_seq_input_dim,
            hidden_dim_seq=hidden_dim_seq,
            hidden_dim=hidden_dim,
            num_discrete=num_discrete,
            memory_dim=memory_dim,
            num_signal=num_signal,
            min_std=min_std
        ).to(device)

        self.optimizer = optim.Adam(self.policynet.parameters(), lr=lr)

        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
    
    def act(self, seq_data, non_seq_data):

        #CONCAT MEMORY ONTO NON SEQ DATA - EITHER HERE OR IN TRAINING LOOP AND SEND

        seq_data = seq_data.to(self.device)
        non_seq_data = non_seq_data.to(self.device)
        bhs_dist, ast_to_dom_dist, mem_update_dist, trade_frac_dist, value = self.policynet(seq_data, non_seq_data)

        #1. SAMPLING FROM DISTRIBUTIONS
        bhs = bhs_dist.sample()
        ast_to_dom = ast_to_dom_dist.sample()
        mem_update = mem_update_dist.sample()
        trade_frac = trade_frac_dist.sample().clamp(0, 1)

        #2. CALCULATING LOG PROBS 
        bhs_logprob = bhs_dist.log_prob(bhs) #need log probabilities for policy gradient calculation

        ast_to_dom_logprobs = ast_to_dom_dist.log_prob(ast_to_dom).sum(-1) 
        #continuous product of probs = continuous sum of log probs
        mem_update_logprobs = mem_update_dist.log_prob(mem_update).sum(-1)
        trade_frac_logprob = trade_frac_dist.log_prob(trade_frac).sum(-1)

        total_logprob = bhs_logprob + ast_to_dom_logprobs + mem_update_logprobs + trade_frac_logprob

        return (bhs, ast_to_dom, mem_update, trade_frac), total_logprob, value
    
    def evaluate(self, seq_data, non_seq_data, actions):
        """Recompute logprobs + entropy + value for PPO update."""

        #CONCAT MEMORY ONTO NON SEQ DATA - EITHER HERE OR IN TRAINING LOOP AND SEND

        seq_data = seq_data.to(self.device)
        non_seq_data = non_seq_data.to(self.device)

        bhs_dist, atod_dist, mem_dist, value = self.policynet(seq_data, non_seq_data)

        a1, a2, a3 = actions
        bhs_p = bhs_dist.log_prob(a1)
        atod_p = atod_dist.log_prob(a2).sum(-1)
        mem_upd_p = mem_dist.log_prob(a3).sum(-1)

        logprob = bhs_p + atod_p + mem_upd_p
        entropy = (bhs_dist.entropy() + atod_dist.entropy().sum(-1) + mem_dist.entropy().sum(-1)).mean()
        return logprob, entropy, value

    def update_policy(self, rollouts):

        """
        rollouts should be a dict with:
        - seq_data
        - non_seq_data
        - actions = (a1, a2, a3)
        - old_logprobs
        - returns
        - advantages

        this is basically all the data that we stored for the number of episodes during which we 
        had freezed policy updation. Now we review our calculations.
        """

        seq_data = rollouts["seq_data"].to(self.device)
        non_seq_data = rollouts["non_seq_data"].to(self.device)
        actions = rollouts["actions"]
        old_logprobs = rollouts["old_logprobs"].to(self.device)
        returns = rollouts["returns"].to(self.device)
        advantages = rollouts["advantages"].to(self.device)
        #returns and advantages calculated in train.py file in training loop

        logprobs, entropy, values = self.evaluate(seq_data, non_seq_data, actions)
        #multiple rows (mini batches) can be evaluated by pytorch just as well as a single row.

        ratios = torch.exp(logprobs - old_logprobs)

        #if r_t > 1 => new policy assigns higher prob to that action than old policy
        #if r_1 < 1 => assigns lower prob

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        value_loss = (returns - values).pow(2).mean()
        #critic training -> critic learns to estimate return value so we train it using regression (MSE)

        loss = actor_loss + self.c1 * value_loss - self.c2* entropy
        #entropy increases => loss decreases (supports exploration => higher entropy means more exploration)
        #since actor and critic share the same network we backpropagate it with the same loss (thus add it).

        #magic statements
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }
