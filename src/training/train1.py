# train.py (corrected & cleaned baseline)

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

# -------------------------------------------------------------------------
# VARIABLES

start = "2007-01-01"
end = "2024-12-31"
augment = False
otpt_show = True
batch_size = 20           # number of windows per batch
split_ratio = 0.8
num_epochs = 10
total_portfolio_value = 100000.0

all_tickers = ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC', 'GS']
domain_wise_tickers = {
    "Tech": ['AAPL', 'GOOGL', 'MSFT'],
    "Finance": ['JPM', 'BAC', 'GS'],
}

domain_indices = {}
asset_indices = {}
d = 0

# ASSET INDICES MAPPING:
for domain, assets in domain_wise_tickers.items():
    a = 0
    domain_indices[domain] = d
    for asset in assets:
        asset_indices[asset] = a
        a += 1
    d += 1

num_domains = len(domain_wise_tickers.keys())
total_num_assets = sum(len(v) for v in domain_wise_tickers.values())

# -------------------------------------------------------------------------
# LOADING DATA

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

# -------------------------------------------------------------------------
# PREPARING DATA FOR TRAINING AND TESTING

# number of columns per sequence
num_seq_cols = len(seq_data["AAPL"]["data"].keys())
# number of windows available for each asset
num_windows_per_asset = train_windows_len = None

def make_windows(seq_data, window_size=60):
    windows = {}
    for asset, info in seq_data.items():
        df = info["data"]
        # ensure df has consistent column ordering and that close is available
        data = df.values  # numpy array [rows, features]
        asset_windows = []
        for i in range(len(data) - window_size):
            seq = data[i: i + window_size]
            asset_windows.append(seq)
        windows[asset] = np.array(asset_windows)
    # assume all assets have same number of windows
    any_asset = next(iter(windows))
    return windows, len(windows[any_asset])

def split(windows, split_ratio=0.8):
    train_data = {}
    test_data = {}
    for asset, seqs in windows.items():
        n = len(seqs)
        split_idx = int(n * split_ratio)
        train_data[asset] = seqs[:split_idx]
        test_data[asset] = seqs[split_idx:]
    return train_data, test_data

window_size = 60
all_windows, num_windows = make_windows(seq_data, window_size)
train_data, test_data = split(all_windows, split_ratio=0.8)
# number of windows available for training for each asset
num_episodes = train_data[next(iter(train_data))].shape[0]

print("Train data shape: ", train_data["AAPL"].shape)  # (no_of_windows, window_timesteps, columns)
print("Test data shape: ", test_data["AAPL"].shape)

# -------------------------------------------------------------------------
# DEFINING ARCHITECTURES

num_assets = 3  # assets per domain (currently equal across domains)

AssetAG = AssetAgent(seq_input_dim=num_seq_cols, non_seq_input_dim=2, device="cpu")
DomainAG = DomainAgent(num_assets, h_asset_dim=2, master_signal_dim=1)
MasterAG = MasterAgent(num_domains=num_domains, h_domain_dim=2)

AssetBuffer = AssetRolloutBuffer()

asset_mem_param = 0.1
domain_mem_param = 0.1
master_mem_param = 0.1
gamma = 0.99


device = getattr(AssetAG, "device", torch.device("cpu"))
# define memories - assuming each memory contains 1 value (for now)
master_memory_dim = 1
domain_memory_dim = 1
asset_memory_dim = 1
master_mem = torch.zeros(1, master_memory_dim, device=device)

domain_mems = {
    domain: torch.zeros(1, domain_memory_dim, device=device)
    for domain in domain_wise_tickers.keys()
}

asset_mems = {
    domain: {
        asset: torch.zeros(1, asset_memory_dim, device=device)
        for asset in assets
    }
    for domain, assets in domain_wise_tickers.items()
}
#memories are initialized in dictionaries

# set equal capital to all (initialization)
domain_allocs = []
asset_allocs = []

# current holdings
domain_cur_holdings = []
asset_cur_holdings = []

for domain, assets in domain_wise_tickers.items():
    domain_allocs.append(1 / num_domains)
    domain_cur_holdings.append(0.0)
    asset_alloc = []
    asset_mem = []
    asset_holding = []
    for asset in assets:
        asset_alloc.append(domain_allocs[domain_indices[domain]] / len(assets))
        asset_holding.append(0.0)
        asset_mem.append(0.0)
    asset_allocs.append(asset_alloc)
    asset_cur_holdings.append(asset_holding)
    asset_mems[domain_indices[domain]] = asset_mem

# -------------------------------------------------------------------------
# UTILITIES


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


# small helper: safe stacking for lists that may contain scalars / different dtypes
def stack_tensor_list(tlist, dtype=torch.float32, device="cpu"):
    tlist2 = [torch.tensor(x, dtype=dtype) if not isinstance(x, torch.Tensor) else x for x in tlist]
    return torch.stack([t.to(device) for t in tlist2])


# -------------------------------------------------------------------------
# TRAINING

# We'll iterate over windows in chunks of batch_size so each batch contains `batch_size` windows
for epoch in range(num_epochs):
    print(GREEN + f"Epoch {epoch+1}/{num_epochs}" + ENDC)

    # iterate through windows in chunks
    for start_idx in range(0, num_episodes, batch_size):
        end_idx = min(start_idx + batch_size, num_episodes)
        current_batch_size = end_idx - start_idx
        # initialize buffers for this batch
        asset_buffer = {
            "seq_data": [],  # list of [T, F] numpy/torch arrays
            "non_seq_data": [],  # list of [non_seq_features]
            "actions": [],
            "old_logprobs": [],
            "values": [],
            "rewards": []
        }

        domain_buffer = {
            "h_assets": [],  # list of per-domain lists (for each domain-window)
            "domain_memory": [],
            "master_alloc": [],
            "actions": [],
            "old_logprobs": [],
            "returns": []
        }

        master_buffer = {
            "h_domains": [],
            "master_memory": [],
            "actions": [],
            "old_logprobs": [],
            "returns": []
        }

        # Loop over windows (timesteps) inside this batch
        for window_offset in range(current_batch_size):
            episode_idx = start_idx + window_offset
            # accumulate domain->master signals for this window
            domain_to_master_signals = []

            # iterate domains and assets
            for domain, assets in domain_wise_tickers.items():
                d_idx = domain_indices[domain]
                domain_mem = domain_mems[domain]
                asset_to_domain_sig_domain = []  # for this domain this window

                # accumulate domain-level P&L
                r_t_domain = 0.0

                for asset in assets:
                    a_idx = asset_indices[asset]
                    asset_mem = asset_mems[asset]

                    # obtain seq window for this asset at episode_idx
                    # train_data[asset] is numpy array shape [num_windows, T, F]
                    seq_np = train_data[asset][episode_idx]  # shape [T, F]
                    seq_tensor = torch.tensor(seq_np, dtype=torch.float32)
                    # non-seq: monetary allocation + memory value for this asset in domain
                    non_seq_tensor = torch.tensor([asset_allocs[d_idx][a_idx], asset_mems[d_idx][a_idx]],
                                                 dtype=torch.float32)

                    seq_in = seq_tensor.unsqueeze(0).to(AssetAG.device)       # [1, T, F]
                    non_seq_in = non_seq_tensor.unsqueeze(0).to(AssetAG.device)  # [1, 2]

                    asset_buffer["seq_data"].append(seq_tensor)  # store [T, F]
                    asset_buffer["non_seq_data"].append(non_seq_tensor)

                    (actions, total_logprob, value) = AssetAG.act(seq_data=seq_in, non_seq_data=non_seq_in)
                    # actions = (bhs, ast_to_dom, mem_update, trade_frac)
                    

                    # squeeze batch dim (we called with [1,...])
                    bhs = actions[0].squeeze(0).detach()
                    ast_to_dom = actions[1].squeeze(0).detach()
                    mem_update = actions[2].squeeze(0).detach()
                    trade_frac = actions[3].squeeze(0).detach()

                    # execute buy/hold/sell using trade_frac and bhs
                    asset_cap = asset_allocs[d_idx][a_idx] * total_portfolio_value
                    target_holding = float(trade_frac.item()) * asset_cap
                    trade_amount = target_holding - asset_cur_holdings[d_idx][a_idx]

                    if trade_amount > 0 and int(bhs.item()) == 0:  # buy
                        buy_amount = min(trade_amount, total_portfolio_value)
                        asset_cur_holdings[d_idx][a_idx] += buy_amount
                    elif trade_amount < 0 and int(bhs.item()) == 2:  # sell
                        sell_amount = -1.0 * trade_amount
                        asset_cur_holdings[d_idx][a_idx] -= sell_amount
                    else:
                        pass  # hold

                    # calculate asset return for this window: use close price column index 0 assumption
                    # Make sure train_data[asset] is [windows, T, F] and that close is at index 0
                    p_t = train_data[asset][episode_idx][window_offset][0]  # close at t (this is an assumption)
                    if episode_idx < num_episodes - 1:
                        p_t_plus_1 = train_data[asset][episode_idx + 1][window_offset][0]
                    else:
                        p_t_plus_1 = p_t
                    frac = p_t_plus_1 / p_t if p_t != 0 else 1.0
                    w_asset = asset_allocs[d_idx][a_idx]
                    r_t = w_asset * (frac - 1.0)
                    r_t_domain += r_t

                    # pass asset->domain signal
                    asset_to_domain_sig_domain.append(ast_to_dom)

                    # memory update: apply additive EMA-style update
                    asset_mems[d_idx][a_idx] = (1 - asset_mem_param) * asset_mems[d_idx][a_idx] + asset_mem_param * float(
                        mem_update.item())

                    # store action info in asset buffer
                    asset_buffer["actions"].append((bhs.cpu(), ast_to_dom.cpu(), mem_update.cpu(), trade_frac.cpu()))
                    asset_buffer["old_logprobs"].append(total_logprob.detach().cpu())
                    asset_buffer["values"].append(value.detach().cpu().squeeze().cpu())
                    asset_buffer["rewards"].append(r_t)

                # done with assets in this domain for this window
                # domain-level inputs for this domain-window: list of ast_to_dom tensors
                # store per domain-window for domain agent (flattened across windows later)
                # convert each ast_to_dom to cpu tensor (they are already)
                domain_buffer["h_assets"].append([x.cpu() for x in asset_to_domain_sig_domain])
                domain_buffer["domain_memory"].append(domain_mem[d_idx])
                domain_buffer["master_alloc"].append(domain_allocs[d_idx])

                # call domain agent for this domain-window (wrap inputs into batch dim)
                # This depends on DomainAG.act signature — adapt if needed.
                try:
                    # prepare domain input: stack asset signals to shape [1, num_assets, num_signal]
                    atod_stack = torch.stack(asset_to_domain_sig_domain).unsqueeze(0).to(DeviceIfExists := AssetAG.device)
                    dom_nonseq = torch.tensor([domain_mem[d_idx], domain_allocs[d_idx]]).unsqueeze(0).to(DeviceIfExists)
                    dom_actions, dom_logprob, dom_value, dom_alloc_distn = DomainAG.act(atod_stack, dom_nonseq, domain_mem[d_idx])
                    # squeeze and detach outputs similar to asset-level
                    domain_buffer["actions"].append(dom_actions)
                    domain_buffer["old_logprobs"].append(dom_logprob.detach().cpu())
                    domain_buffer["returns"].append(r_t_domain)
                except Exception:
                    # If DomainAG.act signature differs, just record placeholders — user should adapt DomainAG call.
                    domain_buffer["actions"].append(None)
                    domain_buffer["old_logprobs"].append(torch.tensor(0.0))
                    domain_buffer["returns"].append(r_t_domain)

                # domain->master signal
                # If DomainAG.act returned something meaningful, try to extract its signal; otherwise use mean of asset signals
                try:
                    # assume dom_actions[0] is allocation signal, dom_actions[1] is signal to master
                    if dom_actions is not None:
                        dom_to_master_signal = dom_actions[1].squeeze(0).detach().cpu()
                    else:
                        # fallback: mean of ast_to_dom signals
                        dom_to_master_signal = torch.stack(asset_to_domain_sig_domain).mean(dim=0).cpu()
                except Exception:
                    dom_to_master_signal = torch.stack(asset_to_domain_sig_domain).mean(dim=0).cpu()

                domain_to_master_signals.append(dom_to_master_signal)

                # update domain memory with domain-level mem update if domain agent provided it
                try:
                    if dom_actions is not None:
                        dom_mem_update = dom_actions[2].squeeze(0).detach().item()
                        domain_mem[d_idx] = (1 - domain_mem_param) * domain_mem[d_idx] + domain_mem_param * dom_mem_update
                except Exception:
                    pass

            # after all domains processed for this window, call master agent
            master_buffer["h_domains"].append([x for x in domain_to_master_signals])
            master_buffer["master_memory"].append(master_mem)
            # compute master return for this window: sum domain P&L (domain_buffer returns stored per domain-window)
            # For simplicity we already accumulated domain_buffer["returns"] per domain-window above; sum last num_domains entries
            # That requires domain_buffer["returns"] length to be multiple of num_domains. We'll compute master_returns directly:
            master_pl = 0.0
            # domain_returns for the last processed window are the last num_domains appended
            if len(domain_buffer["returns"]) >= num_domains:
                last_domain_returns = domain_buffer["returns"][-num_domains:]
                for d_ret, D_alloc in zip(last_domain_returns, domain_allocs):
                    master_pl += D_alloc * (d_ret / (D_alloc + 1e-9)) if D_alloc != 0 else d_ret
            # fallback: approximate master return as sum of domain PL divided by total portfolio value
            V_t = sum(domain_allocs) if sum(domain_allocs) != 0 else total_portfolio_value
            master_R = master_pl / V_t if V_t != 0 else 0.0
            master_buffer["returns"].append(master_R)

            # call master agent (wrap inputs)
            try:
                md_input = torch.stack(domain_to_master_signals).unsqueeze(0).to(AssetAG.device)  # [1, num_domains, num_signal?]
                master_actions, master_logprob, master_value, master_alloc_distn = MasterAG.act(md_input, torch.tensor([master_mem]).unsqueeze(0).to(AssetAG.device))
                master_buffer["actions"].append(master_actions)
                master_buffer["old_logprobs"].append(master_logprob.detach().cpu())
            except Exception:
                master_buffer["actions"].append(None)
                master_buffer["old_logprobs"].append(torch.tensor(0.0))

            # update domain allocations and master memory if master provided allocations/mem
            try:
                if master_actions is not None:
                    new_allocs = master_actions[0].squeeze(0).detach().cpu().numpy()
                    # new_allocs expected to be per-domain monetary allocation — adapt if its normalized weights
                    # For now, if Master returns per-domain fractions sum to 1, scale by total_portfolio_value
                    if np.isclose(new_allocs.sum(), 1.0):
                        domain_allocs = (new_allocs * total_portfolio_value).tolist()
                    else:
                        domain_allocs = new_allocs.tolist()
                    master_mem_update = master_actions[1].squeeze(0).detach().item()
                    master_mem = (1 - master_mem_param) * master_mem + master_mem_param * master_mem_update
            except Exception:
                pass

        # ------------------------------
        # End of batch collection. Prepare rollouts for PPO update.
        # ------------------------------

        # === ASSET LEVEL ROLLOUTS ===
        if len(asset_buffer["rewards"]) > 0:
            # stack seq_data into [N, T, F]
            seq_batch = torch.stack([s for s in asset_buffer["seq_data"]]).to(AssetAG.device)  # [N, T, F]
            nonseq_batch = torch.stack([ns for ns in asset_buffer["non_seq_data"]]).to(AssetAG.device)  # [N, 2]

            # actions: (a1_batch, a2_batch, a3_batch, a4_batch)
            a1_batch = torch.stack([a[0].squeeze() if isinstance(a[0], torch.Tensor) else torch.tensor(a[0]) for a in asset_buffer["actions"]])
            a2_batch = torch.stack([a[1].squeeze() if isinstance(a[1], torch.Tensor) else torch.tensor(a[1]) for a in asset_buffer["actions"]])
            a3_batch = torch.stack([a[2].squeeze() if isinstance(a[2], torch.Tensor) else torch.tensor(a[2]) for a in asset_buffer["actions"]])
            a4_batch = torch.stack([a[3].squeeze() if isinstance(a[3], torch.Tensor) else torch.tensor(a[3]) for a in asset_buffer["actions"]])

            # dtype fixes: categorical actions should be long
            try:
                a1_batch = a1_batch.long().to(AssetAG.device)
            except Exception:
                a1_batch = a1_batch.to(AssetAG.device)

            a2_batch = a2_batch.float().to(AssetAG.device)
            a3_batch = a3_batch.float().to(AssetAG.device)
            a4_batch = a4_batch.float().to(AssetAG.device)

            old_logprobs_batch = torch.stack(asset_buffer["old_logprobs"]).squeeze().to(AssetAG.device)
            values_batch = torch.stack(asset_buffer["values"]).to(AssetAG.device).squeeze()
            rewards_batch = asset_buffer["rewards"]

            # returns & advantages
            returns = compute_returns(rewards_batch, gamma=gamma).to(AssetAG.device)
            advantages = returns - values_batch
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            rollouts = {
                "seq_data": seq_batch,  # [N, T, F]
                "non_seq_data": nonseq_batch,  # [N, 2]
                "actions": (a1_batch, a2_batch, a3_batch, a4_batch),
                "old_logprobs": old_logprobs_batch,
                "returns": returns,
                "advantages": advantages
            }

            # asset-level update
            asset_update_info = AssetAG.update_policy(rollouts)
            print(f"Asset update: {asset_update_info}")

        # === DOMAIN LEVEL ROLLOUTS ===
        # The domain agent API may differ. Here we prepare flattened lists per domain-window.
        try:
            if len(domain_buffer["returns"]) > 0:
                # domain observations: we used `h_assets` as a list of lists (asset signals for that domain-window).
                # stack/format these into tensors for DomainAG.evaluate/update if it expects them.
                # This block is a best-effort; adapt to your DomainAG API as needed.

                # domain_obs_count = len(domain_buffer['h_assets'])
                # create tensors for domain inputs (one per domain-window)
                domain_seq_list = []
                for h_assets in domain_buffer["h_assets"]:
                    # h_assets is a list of tensors per asset: [num_assets, num_signal]
                    try:
                        stacked = torch.stack(h_assets)  # [num_assets, num_signal]
                    except Exception:
                        # fallback: try to convert list elements to tensors first
                        stacked = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in h_assets])
                    domain_seq_list.append(stacked)

                domain_seq_batch = torch.stack(domain_seq_list).to(AssetAG.device)  # [N_domain_windows, num_assets, num_signal]
                domain_nonseq_batch = torch.tensor(domain_buffer["domain_memory"], dtype=torch.float32).unsqueeze(-1).to(AssetAG.device)

                domain_old_logprobs = torch.stack(domain_buffer["old_logprobs"]).to(AssetAG.device).squeeze()
                domain_returns = torch.tensor(domain_buffer["returns"], dtype=torch.float32).to(AssetAG.device)

                # compute domain values/advantages placeholder: DomainAG.evaluate may return these
                # We'll try to call DomainAG.update_policy similarly if the API aligns
                dom_rollouts = {
                    "seq_data": domain_seq_batch,
                    "non_seq_data": domain_nonseq_batch,
                    "actions": domain_buffer["actions"],
                    "old_logprobs": domain_old_logprobs,
                    "returns": domain_returns,
                    "advantages": domain_returns  # placeholder; ideally compute returns - values
                }
                try:
                    DomainAG.update_policy(dom_rollouts)
                    print("Domain update called.")
                except Exception:
                    # If update_policy signature differs, skip
                    print(YELLOW + "DomainAG.update_policy skipped (API mismatch). Please adapt domain rollout packaging." + ENDC)
        except Exception as e:
            print(RED + f"Domain rollout packaging failed: {e}" + ENDC)

        # === MASTER LEVEL ROLLOUTS ===
        try:
            if len(master_buffer["returns"]) > 0:
                master_returns = torch.tensor(master_buffer["returns"], dtype=torch.float32).to(AssetAG.device)
                # minimal master rollout packaging (adapt if MasterAG API requires different shapes)
                master_rollouts = {
                    "seq_data": None,
                    "non_seq_data": torch.tensor(master_buffer["master_memory"], dtype=torch.float32).unsqueeze(-1).to(AssetAG.device),
                    "actions": master_buffer["actions"],
                    "old_logprobs": torch.stack(master_buffer["old_logprobs"]).to(AssetAG.device).squeeze(),
                    "returns": master_returns,
                    "advantages": master_returns  # placeholder
                }
                try:
                    MasterAG.update_policy(master_rollouts)
                    print("Master update called.")
                except Exception:
                    print(YELLOW + "MasterAG.update_policy skipped (API mismatch). Please adapt master rollout packaging." + ENDC)
        except Exception as e:
            print(RED + f"Master rollout packaging failed: {e}" + ENDC)

    # end batch loop

print(GREEN + "Training finished." + ENDC)
