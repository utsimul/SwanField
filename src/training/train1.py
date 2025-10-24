import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data import Dataset, DataLoader, random_split

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

# Import agents and preprocessing
sys.path.append('src/agents')
from asset_level_agent import AssetAgent, AssetRolloutBuffer
from domain_level_agent import DomainAgent
from master_agent import MasterAgent
from helpernets import *
from preprocessing import run  # import only run()

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'

#----------------------------------------------------------------------------------------
#VARIABLES

start = "2007-01-01"
end = "2024-12-31"
augment = False
otpt_show = True
batch_size = 20
split_ratio = 0.8
num_epochs = 10
total_portfolio_value = 100000

all_tickers = ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC', 'GS'] 
domain_wise_tickers = {
    "Tech": ['AAPL', 'GOOGL', 'MSFT'],
    "Finance": ['JPM', 'BAC', 'GS'],
}

domain_indices = {}
asset_indices = {}
d = 0

#ASSET INDICES MAPPING:
for domain, assets in domain_wise_tickers.items():
    a = 0
    domain_indices[domain] = d
    for asset in assets:
        asset_indices[asset] = a
        a+=1
    d +=1

num_domains= len(domain_wise_tickers.keys())

#----------------------------------------------------------------------------------------
#LOADING DATA

outdir = "output_data"
os.makedirs(outdir, exist_ok=True)

results = {}
missing_tickers = []

for ticker in all_tickers:
    filename = os.path.join(outdir, f"{ticker}_processed.csv")
    if os.path.exists(filename):
        print(GREEN + f"Loading preprocessed data for {ticker} from {filename}" + ENDC)
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        results[ticker] = {"data": df, "scalers": None}  # scalers not saved to CSV
    else:
        print(YELLOW + f"No preprocessed file for {ticker}, will run preprocessing" + ENDC)
        missing_tickers.append(ticker)

if missing_tickers:
    print(BLUE + f"Running preprocessing for missing tickers: {missing_tickers}" + ENDC)
    new_results = run(missing_tickers, start, end, augment, otpt_show)
    results.update(new_results)

seq_data = results
print(GREEN + "Final dataset ready!" + ENDC)
print(seq_data.keys())

#----------------------------------------------------------------------------------------
#PREPARING DATA FOR TRAINING AND TESTING

num_seq_cols = len(seq_data["AAPL"]["data"].keys())
num_seq_rows = len(seq_data["AAPL"]["data"])
split_idx = int(num_seq_rows * split_ratio)

def make_windows(seq_data, window_size = 60):
    windows = {}
    for asset, info in seq_data.items():

        df = info["data"]
        data = df.values #numpy array
        asset_windows = []
        for i in range(len(data) - window_size):
            seq = data[i : i+window_size]
            asset_windows.append(seq)
        
        windows[asset] = np.array(asset_windows)
    return windows, len(windows)

def split(windows, split_ratio=0.8):
    train_data = {}
    test_data = {}
    for asset, seqs in windows.items():
        n = len(seqs)
        split_idx = int(n*split_ratio)
        train_data[asset] = seqs[:split_idx]
        test_data[asset]  = seqs[split_idx:]
    return train_data, test_data


window_size = 60
all_windows, num_windows = make_windows(seq_data, window_size)
train_data, test_data = split(all_windows, split_ratio=0.8)
num_episodes = train_data.shape[0]

print("Train data shape: " , train_data["AAPL"].shape)  #(no_of_windows, window_timesteps, columns)
print("Test data shape: ", test_data["AAPL"].shape)   

#----------------------------------------------------------------------------------------
#DEFINING ARCHITECTURES

num_assets = 3 
"""number of assets per domain
Currently, we assume same number of assets for each domain - later to be changed to where the 
inputs for pooling layer dimensions obtained "dynamically"

num_signal = 2 by default (for both asset to domain, domain to master)
master sends 1 allocation and domain also does."""


AssetAG = AssetAgent(seq_input_dim=num_seq_cols, non_seq_input_dim=2)
DomainAG = DomainAgent(num_assets, h_asset_dim=2, master_signal_dim=1)
MasterAG = MasterAgent(num_domains=2, h_domain_dim=2)

AssetBuffer = AssetRolloutBuffer()


asset_mem_param = 0.1
domain_mem_param = 0.1
master_mem_param = 0.1
gamma = 0.99

#define memories - assuming each memory contains 1 value (for now)
master_mem = 0
domain_mem = []
asset_mems = []

#set equal capital to all (initialization) - alt: can use init as per current price
domain_allocs = [] #one alloc per asset
asset_allocs = []

#current holdings
domain_cur_holdings = []
asset_cur_holdings = []

#filling values:

for domain, assets in domain_wise_tickers.items():
    domain_allocs.append(total_portfolio_value/num_domains)
    domain_mem.append(0)
    domain_cur_holdings.append(0)
    asset_alloc = []
    asset_mem = []
    asset_holding = []
    for asset in assets:
        asset_alloc.append(domain_allocs[domain_indices[domain]]/len(assets))
        asset_holding.append(0)
        asset_mem.append(0)
    asset_allocs.append(asset_alloc)
    asset_cur_holdings.append(asset_holding)
    asset_mems.append(asset_mem)


#----------------------------------------------------------------------------------------
#RETURN ESTIMATIONS:


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

#----------------------------------------------------------------------------------------
#TRAINING

for epoch in range(num_epochs):

    # PPO UPDATE OCCURS HERE 
    asset_rollouts = []
    domain_rollouts = []
    master_rollouts = []

    #1 batch of 20 overlaps of t timesteps, thetas are fixed for 1 batch
     #(no_of_windows, window_timesteps, columns)
    episode_rewards = [] 
    asset_buffer = {
        "seq_data": [],       # list of tensors: [seq_len, features]
        "non_seq_data": [],   # list of tensors: [non_seq_features]
        "actions": [],        # list of tuples: (a1, a2, a3, a4)
        "old_logprobs": [],   # list of tensors (scalar)
        "values": [],         # list of tensors (scalar)
        "rewards": []         # list of floats
    }


    domain_buffer = {
        "h_assets" : [],
        "domain_memory" : [],
        "master_alloc" : [],
        "actions:" : [],
        "old_logprobs" : [],
        "returns" : [],
        "advantages" : []
    }

    master_buffer = {
        "h_domains" : [],
        "master_memory" : [],
        "actions" : [],
        "old_logprobs" : [],
        "returns" : [],
        "advantages" : []
    }

    for window in range(batch_size): #number of windows in a batch
        batch_reward = [] #asset, domain, master
        asset_to_domain_signals = [] #3D list
        domain_to_master_signals = [] #2D list
        master_returns = 0

        for domain, assets in domain_wise_tickers.items():
            domain_idx = domain_indices[domain]
            asset_to_domain_sig_domain = []
            domain_rewards = 0
            for asset in assets:
                r_t_domain = 0
                #TRAIN ASSET LEVEL AGENT:
                #train_data["asset"]["data"] -> (no_of_windows, window_timesteps, columns)
                asset_idx = asset_indices[asset]
                seq_tensor = torch.tensor(train_data[asset]["data"][window], dtype=torch.float32)
                non_seq_tensor = torch.tensor([asset_allocs[domain_idx][asset_idx], asset_mems[domain_idx][asset_idx]],
                            dtype=torch.float32)
                seq_in = seq_tensor.unsqueeze(0).to(AssetAG.device)       # [1, T, F]
                non_seq_in = non_seq_tensor.unsqueeze(0).to(AssetAG.device)             
                (actions, total_logprob, value) = AssetAG.act(seq_data=seq_in, non_seq_data=non_seq_in)
                # actions is tuple (bhs, ast_to_dom, mem_update, trade_frac)
                # each entry is a tensor with batch dim -> squeeze the batch dim before storing

                asset_buffer["seq_data"].append(seq_tensor)   # store without batch dim [T, F]
                asset_buffer["non_seq_data"].append(non_seq_tensor)

                #we're inside batch loop. Thus the outermost batch dimension is 1. Therefore we squeeze that dimension.
                bhs = actions[0].squeeze(0).detach()        # bhs: categorical action (scalar tensor)
                ast_to_dom = actions[1].squeeze(0).detach()        # ast_to_dom continuous vector tensor
                mem_update = actions[2].squeeze(0).detach()        # mem_update
                trade_frac = actions[3].squeeze(0).detach()        # trade_frac


                #1. BUY HOLD SELL:
                asset_cap = asset_allocs[domain_idx][asset_idx]
                target_holding = trade_frac.item() * asset_cap
                trade_amount = target_holding - asset_cur_holdings[domain_idx][asset_idx]


                if trade_amount > 0 and bhs == 0: #buy signal
                    buy_amount = min(trade_amount, total_portfolio_value)
                    asset_cur_holdings[domain_idx][asset_idx] += buy_amount

                elif trade_amount < 0 and bhs == 2: #sell signal
                    sell_amount = -1 * trade_amount
                    asset_cur_holdings[domain_idx][asset_idx] -= sell_amount
                
                else:
                    #hold - do nothing
                    pass
            
                #calculate returns:
                p_t = train_data[window][0][-1]
                if(window != batch_size-1):
                    p_t_plus_1 = train_data[window][0][-1] #close price of the next immediate timestep
                    frac = p_t_plus_1/p_t
                    w_asset = asset_allocs[domain_idx][asset_idx] #weight
                    r_t = w_asset * (frac - 1)
                    r_t_domain += r_t
                
                #pass to domain
                asset_to_domain_sig_domain.append(ast_to_dom)
                asset_to_domain_signals.append(asset_to_domain_sig_domain)
                #memory update
                asset_mems[domain_idx][asset_idx] += asset_mem_param * mem_update

            
                asset_buffer["actions"].append((bhs.cpu(), ast_to_dom.cpu(), mem_update.cpu(), trade_frac.cpu()))
                asset_buffer["old_logprobs"].append(total_logprob.detach().cpu())
                asset_buffer["values"].append(value.detach().cpu().squeeze())  # scalar
                asset_buffer["rewards"].append(r_t)   # float

            #once the asset level processing is done, we move to their domain
            domain_buffer["h_assets"].append(asset_to_domain_signals)
            domain_buffer["domain_memory"].append(domain_mem[domain_idx])
            domain_buffer["master_alloc"].append(domain_allocs[domain_idx])

            actions, total_logprob, value, alloc_distn = DomainAG.act(asset_to_domain_signals[domain_idx])
            allocations = ast_to_dom = actions[0].squeeze(0).detach()
            dtom = actions[1].squeeze(0).detach()
            mem_update = actions[2].squeeze(0).detach()

            #set allocations:
            asset_allocs[domain_idx] = allocations

            #update in domain to master
            domain_to_master_signals.append(dtom)

            #memory update
            domain_mem[domain_idx] += domain_mem_param * mem_update

            domain_buffer["actions"].append(actions)
            domain_buffer["old_logprobs"].append(total_logprob)
            
            #calculate domain returns:
            #(later: modify returns to reward profits scored during high volatility)
            domain_buffer["returns"].append(r_t_domain)
            master_returns += domain_allocs[domain_idx] * r_t_domain
        
        #master
        master_buffer["h_domains"] = domain_to_master_signals
        master_buffer["master_memory"] = master_mem

        master_actions, total_logprob, value, alloc_distn = MasterAG.act(domain_to_master_signals, master_mem)
        allocations = master_actions[0].squeeze(0).detach()
        mem_update = master_actions[0].squeeze(0).detach()

        master_buffer["actions"] = master_actions
        master_buffer["old_logprobs"] = total_logprob
        master_buffer["returns"] = master_returns
        
        #update domain allocations:
        domain_allocs = allocations

        #memory update
        master_mem += master_mem_param * mem_update

        
        
    #All windows in that batch completed -> so now we update
    #ASSET BUFFER DIMENSIONS: keys - values are flat buffers -> asset1, asset 2,... for a domain, asset3, asset 4, ... for domain2 ... for all domains, asset 1, asset 2,... for a domain (next window),... for all windows
    #basically the flatenned version of (windows, domains, assets)
    #DOMAIN BUFFER DIMENSIONS: keys - for every key there is a list (batch_size, num_assets) for h_assets or (batch_size)
    #MASTER BUFFER DIMENSION: straightforward - keys: values are list of number of windows in that batch 

    AssetAG.update_policy(asset_buffer)
    DomainAG.update_policy(domain_buffer)
    MasterAG.update_policy(master_buffer)


