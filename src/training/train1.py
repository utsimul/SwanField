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
# sys.path.append('src/agents')
# from asset_level_agent import AssetAgent, AssetRolloutBuffer
# from domain_level_agent import DomainAgent
# from master_agent import MasterAgent
# from helpernets import *
# from preprocessing import run  # import only run()

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'
BRIGHT_MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
STD_BLUE = '\033[34m'
MAGENTA = '\033[35m'

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
num_episodes = len(train_data["AAPL"].shape)
print(BLUE, "Train data shape: " , train_data["AAPL"].shape, ENDC) #(no_of_windows, window_timesteps, columns)
print(GREEN + "no of windows, window timesteps, columns" + ENDC) 
print(BLUE, "Test data shape: ", test_data["AAPL"].shape, ENDC)   

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


# After creating agents: AssetAG, DomainAG, MasterAG
device_asset = AssetAG.device
device_domain = DomainAG.device
device_master = MasterAG.device

# memory dims (your agents initialize memory_dim attribute)
asset_memory_dim = 1
domain_memory_dim = DomainAG.memory_dim if hasattr(DomainAG, "memory_dim") else 1
master_memory_dim = MasterAG.memory_dim if hasattr(MasterAG, "memory_dim") else 1

# master memory as tensor (batch dim 1)
master_mem = torch.zeros(1, master_memory_dim, device=device_master, dtype=torch.float32)

# domain memories: list of tensors shaped (1, memory_dim)
domain_mem = [
    torch.zeros(1, domain_memory_dim, device=device_domain, dtype=torch.float32)
    for _ in range(num_domains)
]

# asset memories: list (per domain) of list (per asset) of tensors (1, memory_dim)
asset_mems = []
for domain, assets in domain_wise_tickers.items():
    am = [torch.zeros(1, asset_memory_dim, device=device_asset, dtype=torch.float32) for _ in assets]
    asset_mems.append(am)

#ALLOCATIONS ARE IN THE FORM OF RATIOS: TRUE ALLOCATIONS = ALLOCATIONS * PORTFOLIO VALUE

# domain_allocs: keep as list of 1-element tensors (shape (1,))
# domain_allocs = [
#     torch.tensor([1 / num_domains], dtype=torch.float32, device=device_domain)
#     for _ in range(num_domains)
# ]
domain_allocs = []
dtemp = [torch.tensor(1 / num_domains) for _ in range(num_domains+1)]
domain_allocs.append(dtemp)
#a tensor

print(BLUE, "domain allocs: ", domain_allocs, ENDC)

# asset_allocs: per-domain list of per-asset 1-element tensors
asset_allocs = []
for domain_idx, (domain, assets) in enumerate(domain_wise_tickers.items()):
    per_domain = [
        torch.tensor((1 / num_domains) / len(assets), dtype=torch.float32, device=device_asset)
        for _ in assets
    ]
    asset_allocs.append(per_domain)
#an array of tensors

print(BLUE, "asset allocs initialized: ", asset_allocs, ENDC)

# current holdings (keep as python numbers or tensors as you prefer)
domain_cur_holdings = [0 for _ in range(num_domains)]
asset_cur_holdings = [[0 for _ in assets] for assets in domain_wise_tickers.values()]


asset_mem_param = 0.7
domain_mem_param = 0.7
master_mem_param = 0.7

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

net_training_returns = [ [ 0 for asset in range(num_assets)] for domain in range(num_domains) ]  

def train():
    global domain_allocs, asset_allocs, domain_mem, master_mem
    for epoch in range(num_epochs):

        #batches- PPO UPDATE OCCURS HERE 
        print(GREEN + f"starting epoch {epoch}..." + ENDC)
        for start in range(0, num_windows, batch_size): #generating batches dynamically. 

            end = start + batch_size
            batch_idxs = range(start, min(end, num_windows))

            batch_rewards = [] 
            # asset_buffer = {
            #     "seq_data": [],       # list of tensors: [seq_len, features]
            #     "non_seq_data": [],   # list of tensors: [non_seq_features]
            #     "actions": [],        # list of tuples: (a1, a2, a3, a4)
            #     "old_logprobs": [],   # list of tensors (scalar)
            #     "values": [],         # list of tensors (scalar)
            #     "rewards": []         # list of floats
            # }
            asset_buffer = {}


            # domain_buffer = {
            #     "h_assets" : [],
            #     "domain_memory" : [],
            #     "master_alloc" : [],
            #     "actions" : [],
            #     "old_logprobs" : [],
            #     "returns" : [],
            #     "advantages" : []
            # }

            domain_buffer = {}

            # master_buffer = {
            #     "h_domains" : [],
            #     "master_memory" : [],
            #     "actions" : [],
            #     "old_logprobs" : [],
            #     "returns" : [],
            #     "advantages" : []
            # }

            master_buffer = {}

            for window_idx in batch_idxs:

                asset_to_domain_signals = []  # [num_domains][num_assets][ast_to_dom_dim]
                domain_to_master_signals = []  # [num_domains,batch, dtom_dim] 
                master_returns = 0

                for domain, assets in domain_wise_tickers.items():
                    domain_idx = domain_indices[domain]
                    asset_to_domain_sig_domain = []
                    r_t_domain = 0.0

                    for asset in assets:
                        asset_idx = asset_indices[asset]

                        seq_tensor = torch.tensor(train_data[asset][window_idx], dtype=torch.float32) #(window timesteps, columns)
                        #non_seq_tensor = torch.tensor([asset_allocs[domain_idx][asset_idx], asset_mems[domain_idx][asset_idx]], dtype=torch.float32)

                        # print(BLUE + "asset allocs: " , asset_allocs[domain_idx][asset_idx] , ENDC)
                        # print(BLUE + "asset mems: " , asset_mems[domain_idx][asset_idx] , ENDC)

                        alloc_val = asset_allocs[domain_idx][asset_idx]  #  this is not batch first when printed.
                        mem_val = asset_mems[domain_idx][asset_idx]      # this is in fact batch first.

                        if alloc_val.ndim == 1:
                            alloc_val = alloc_val.unsqueeze(-1)  # (batch, 1)
                        
                        elif alloc_val.ndim == 0:
                            alloc_val = alloc_val.unsqueeze(0) .unsqueeze(0) # (batch, 1)
                        else:
                            pass

                        # print(BLUE + "val allocs: " , alloc_val , ENDC)
                        # print(BLUE + "val mems: " , mem_val , ENDC)

                        non_seq_tensor = torch.cat([alloc_val, mem_val], dim=-1)  # (batch, 2)

                        #1D tensor :- [asset_allocation value, asset memory value]

                        seq_in = seq_tensor.unsqueeze(0).to(AssetAG.device)  # [1, T, F]
                        #in PPO we are processing data in batches BUT we have to pass each batch instance one at a time, but since 
                        #pytorch expects data as (batch, ...,...) we need to make faux batches.

                        non_seq_in = non_seq_tensor.to(AssetAG.device) #non_seq_tensor.unsqueeze(0).to(AssetAG.device) #(1, 1D tensor)

                        # print(RED, "seq_in shape: ", seq_in.shape, " non_seq_in shape: ", non_seq_in.shape, ENDC)
                        

                        actions, total_logprob, value = AssetAG.act(seq_in, non_seq_in) #put in the form of batch first

                        bhs = actions[0].detach() 
                        ast_to_dom = actions[1].detach() #detach completely removes the new tensor from the current computational graph 
                        mem_update = actions[2].detach()
                        trade_frac = actions[3].detach() #not adding squeeze(0) so the first dimension is still batch.

                        with torch.no_grad():
                            

                            # reward calc
                            p_t = train_data[asset][window_idx+1][0][0] #close price of first timestep of next window
                            if window_idx < num_windows - 1:
                                p_t_plus_1 = train_data[asset][window_idx][0][0]
                                frac = p_t_plus_1 / p_t

                                bhs_signal = bhs[0] #batch first
                                if bhs_signal == 1:
                                    #buy
                                    w_asset = asset_allocs[domain_idx][asset_idx]
                                    r_t = w_asset * (frac - 1) #we bought, so reward is added. 
                                # print(YELLOW + f"Asset: {asset}, p_t: {p_t}, p_t+1: {p_t_plus_1}, frac: {frac}, w_asset: {w_asset}" + ENDC)
                                elif bhs_signal == 1:
                                    #hold
                                    r_t = 0.0
                                else:
                                    #sell
                                    w_asset = asset_allocs[domain_idx][asset_idx]
                                    r_t = w_asset * -1 * (frac - 1) #we bought, so reward is subtracted (into -1).
                                net_training_returns[domain_idx][asset_idx] += r_t
                                r_t_domain += r_t

                            asset_to_domain_sig_domain.append(ast_to_dom) #(assets, batch, ...) because the first dimension of every ast_to_dom
                    #is batch and we are appending various ast_to_doms in the list above.
                            asset_mems[domain_idx][asset_idx] += asset_mem_param * mem_update

                        #I JUST NEED TO STORE IN ROLLOUTS IN THE SAME FORMAT AS I HAD PASSED INTO ACT() BECAUSE THE UPDATE POLICY ONLY 
                        #RECREATES THE SCENARIO

                        if(len(asset_buffer)==0):
                            # print(CYAN + "First value in rollouts being stored" + ENDC)
                            asset_buffer["seq_data"] = seq_in.clone()
                            asset_buffer["non_seq_data"] = non_seq_in.clone()
                            asset_buffer["bhs"] = bhs.clone()
                            asset_buffer["ast_to_dom"] = ast_to_dom.clone()
                            asset_buffer["mem_update"] = mem_update.clone()
                            asset_buffer["trade_frac"] = trade_frac.clone()
                            asset_buffer["old_logprobs"] = total_logprob.clone()
                            asset_buffer["values"] = value.clone()
                            # print(CYAN + "r_t: ", r_t.unsqueeze(0), ENDC)
                            asset_buffer["rewards"] = r_t.clone().unsqueeze(0) #need to put this in a batch first format maybe
                            # print(CYAN + "First asset done" + ENDC)
                        else:
                            print(RED, "asset buffer before: ", asset_buffer["seq_data"].shape, ENDC)
                            asset_buffer["seq_data"] = torch.cat([asset_buffer["seq_data"], seq_in.clone()], dim=0) #seq_tensor has batch first (batch, window timesteps, columns)
                            # print(CYAN + "seq_in shape stored: ", seq_in, ENDC)
                            print(RED, "asset buffer after: ", asset_buffer["seq_data"].shape, ENDC)
                            asset_buffer["non_seq_data"] = torch.cat([asset_buffer["non_seq_data"], non_seq_in.clone()], dim=0)
                            asset_buffer["bhs"] = torch.cat([asset_buffer["bhs"], bhs.clone()], dim=0)
                            asset_buffer["ast_to_dom"] = torch.cat([asset_buffer["ast_to_dom"], ast_to_dom.clone()], dim=0)
                            asset_buffer["mem_update"] = torch.cat([asset_buffer["mem_update"], mem_update.clone()], dim=0)
                            asset_buffer["trade_frac"] = torch.cat([asset_buffer["trade_frac"], trade_frac.clone()], dim=0)
                            asset_buffer["old_logprobs"] = torch.cat([asset_buffer["old_logprobs"], total_logprob.clone()], dim=0)
                            asset_buffer["values"] = torch.cat([asset_buffer["values"], value.clone()], dim=0)
                            # print(CYAN + "r_t: ", r_t.unsqueeze(0), ENDC)
                            asset_buffer["rewards"] = torch.cat([asset_buffer["rewards"], r_t.clone().unsqueeze(0)], dim=0) #need to put this in a batch first format maybe
                            # print(CYAN + "Asset done" + ENDC)

                    # if isinstance(asset_to_domain_sig_domain, list):
                    #     if isinstance(asset_to_domain_sig_domain[0], torch.Tensor):
                    #         asset_to_domain_sig_domain = torch.stack(asset_to_domain_sig_domain)
                    #         #joins all the elements of the array along dimension 0 (stack) => batch stack
                    #     else:
                    #         asset_to_domain_sig_domain = torch.tensor(asset_to_domain_sig_domain, dtype=torch.float32)

                    #i am skipping this block because i want to keep everything batch first to ensure uniformity
                    with torch.no_grad():
                        asset_to_domain_sig_domain = (
                            torch.stack(asset_to_domain_sig_domain)    # (num_assets, batch, dim)
                            .permute(1, 0, 2)                          # → (batch, num_assets, dim)
                            .to(DomainAG.device)
                        )

                        # if not asset_to_domain_sig_domain:
                        #     # Make a dummy tensor if empty to prevent crash
                        #     asset_to_domain_sig_domain = torch.zeros(
                        #         (1, len(domain_wise_tickers[domain]), DomainAG.h_asset_dim),
                        #         device=DomainAG.device
                        #     )
                        # else:
                        #     asset_to_domain_sig_domain = (
                        #         torch.stack(asset_to_domain_sig_domain)
                        #         .permute(1, 0, 2)
                        #         .to(DomainAG.device)
                        #     )

                    # print(CYAN, "asset to domain sig domain stacked" , asset_to_domain_sig_domain, ENDC)
                    alloc_val = domain_allocs[0][domain_idx]  #  this is not batch first when printed.
                    # print(GREEN, "alloc val: ", alloc_val, ENDC)

                    if alloc_val.ndim == 1:
                        alloc_val = alloc_val.unsqueeze(0)  # (batch, 1)
                    
                    elif alloc_val.ndim == 0:
                        alloc_val = alloc_val.unsqueeze(0).unsqueeze(0) # (batch, 1) => adding 2 dims to match h_assets
                    elif alloc_val.ndim == 3:
                        alloc_val = alloc_val.squeeze(0)  #remove extra dim if already present
                    else:
                            pass
                    # print(GREEN, "alloc val: ", alloc_val, ENDC)
                    # print(BLUE + "h_assets: " , asset_to_domain_sig_domain , ENDC)
                    actions, total_logprob, value, alloc_distn = DomainAG.act(asset_to_domain_sig_domain, alloc_val, domain_mem[domain_idx])

                    with torch.no_grad():
                        allocations = actions[0].detach() #Keeping batch dimensions
                        print(GREEN , "all allocations (domain output)" , allocations, ENDC)    
                        dtom = actions[1].detach()
                        mem_update = actions[2].detach()

                        asset_allocs[domain_idx] = allocations.squeeze(0)  #remove batch dim
                        domain_mem[domain_idx] += domain_mem_param * mem_update

                        if(len(domain_buffer)==0):
                            # print(CYAN, "First value in domain rollouts being stored: h_assets ", asset_to_domain_sig_domain, ENDC)
                            domain_buffer["h_assets"] = asset_to_domain_sig_domain.detach()
                            domain_buffer["domain_memory"] = domain_mem[domain_idx].detach()
                            domain_buffer["master_alloc"] = alloc_val.detach()
                            domain_buffer["allocations"] = allocations.detach()
                            domain_buffer["dtom"] = dtom.detach()
                            domain_buffer["mem_update"] = mem_update.detach()
                            domain_buffer["returns"] = r_t_domain.detach().unsqueeze(0)
                            domain_buffer["old_logprobs"] = total_logprob.detach()
                            domain_buffer["advantages"] = value.detach()
                        else:
                            # print(CYAN, "Next value in domain rollouts being stored: h_assets ", asset_to_domain_sig_domain, ENDC)
                            domain_buffer["h_assets"] = torch.cat([domain_buffer["h_assets"], asset_to_domain_sig_domain.detach()], dim=0)
                            # print(CYAN + "domain h_assets shape stored: ", asset_to_domain_sig_domain, ENDC)
                            domain_buffer["domain_memory"] = torch.cat([domain_buffer["domain_memory"], domain_mem[domain_idx].detach()], dim=0)
                            domain_buffer["master_alloc"] = torch.cat([domain_buffer["master_alloc"], alloc_val.detach()], dim=0)
                            domain_buffer["allocations"] = torch.cat([domain_buffer["allocations"], allocations.detach()], dim=0)
                            domain_buffer["dtom"] = torch.cat([domain_buffer["dtom"], dtom.detach()], dim=0)
                            domain_buffer["mem_update"] = torch.cat([domain_buffer["mem_update"], mem_update.detach()], dim=0)
                            domain_buffer["returns"] = torch.cat([domain_buffer["returns"], torch.tensor(r_t_domain).detach().unsqueeze(0)], dim=0)
                            domain_buffer["old_logprobs"] = torch.cat([domain_buffer["old_logprobs"], total_logprob.detach()], dim=0)
                            domain_buffer["advantages"] = torch.cat([domain_buffer["advantages"], value.detach()], dim=0)


                        domain_to_master_signals.append(dtom)
                        master_returns += domain_allocs[0][domain_idx] * r_t_domain
                        print(MAGENTA + "domain done" + ENDC)
                
                with torch.no_grad():
                    # master
                    domain_to_master_signals = (
                            torch.stack(domain_to_master_signals)    # [num_domains,batch, dtom_dim] 
                            .permute(1, 0, 2)                          # → (batch, num_domains, dtom_dim)
                            .to(MasterAG.device)
                    )

                master_actions, total_logprob, value, alloc_distn = MasterAG.act(domain_to_master_signals, master_mem)

                with torch.no_grad():
                    allocations = master_actions[0].detach()  #keeping batch first => that's why not applying squeeze(0)
                    mem_update = master_actions[1].detach()
                    domain_allocs = allocations #keep as it is                     #DOUBT HERE ******
                    print(GREEN , "all allocations (master output)" , allocations, ENDC)
                    master_mem += master_mem_param * mem_update

                    if(len(master_buffer)==0):
                        # print(MAGENTA, "First value in master rollouts being stored: h_assets ", asset_to_domain_sig_domain, ENDC)
                        master_buffer["h_domains"] = domain_to_master_signals.detach()
                        master_buffer["master_memory"] = master_mem.detach()
                        master_buffer["allocations"] = allocations.detach()
                        master_buffer["mem_update"] = mem_update.detach()
                        master_buffer["returns"] = master_returns.detach().unsqueeze(0)
                        master_buffer["old_logprobs"] = total_logprob.detach()
                        master_buffer["advantages"] = value.detach()
                    else:
                        # print(MAGENTA, "Next value in master rollouts being stored: h_assets ", domain_to_master_signals, ENDC)
                        master_buffer["h_domains"] = torch.cat([master_buffer["h_domains"], domain_to_master_signals.detach()], dim=0)
                        master_buffer["master_memory"] = torch.cat([master_buffer["master_memory"], master_mem.detach()], dim=0)
                        master_buffer["allocations"] = torch.cat([master_buffer["allocations"], allocations.detach()], dim=0)
                        master_buffer["mem_update"] = torch.cat([master_buffer["mem_update"], mem_update.detach()], dim=0)
                        master_buffer["returns"] = torch.cat([master_buffer["returns"], torch.tensor(master_returns).detach().unsqueeze(0)], dim=0)
                        master_buffer["old_logprobs"] = torch.cat([master_buffer["old_logprobs"], total_logprob.detach()], dim=0)
                        master_buffer["advantages"] = torch.cat([master_buffer["advantages"], value.detach()], dim=0)

                print(WHITE + f"one asset-domain-master cycle (window: {window_idx}) done -----------------------------------------------------" + ENDC)


            print(BRIGHT_MAGENTA + "upadating all agents" + ENDC)
            AssetAG.update_policy(asset_buffer)
            print(MAGENTA + "asset agent updated" + ENDC)
            DomainAG.update_policy(domain_buffer)
            print(MAGENTA + "domain agent updated" + ENDC)
            MasterAG.update_policy(master_buffer)
            print(BRIGHT_MAGENTA + "updates done!" + ENDC)
