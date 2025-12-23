net_testing_returns = [ [ 0 for asset in range(num_assets)] for domain in range(num_domains) ]  

num_domains = len(domain_wise_tickers)
#re initialize previous values:

# master memory as tensor (batch dim 1)
mMem = torch.zeros(1, master_memory_dim, device=device_master, dtype=torch.float32)

# domain memories: list of tensors shaped (1, memory_dim)
dMem = [
    torch.zeros(1, domain_memory_dim, device=device_domain, dtype=torch.float32)
    for _ in range(num_domains)
]

# asset memories: list (per domain) of list (per asset) of tensors (1, memory_dim)
asset_mems = []
for domain, assets in domain_wise_tickers.items():
    am = [torch.zeros(1, asset_memory_dim, device=device_asset, dtype=torch.float32) for _ in assets]
    asset_mems.append(am)


dAllocs = []
dtemp = [torch.tensor(1 / num_domains) for _ in range(num_domains+1)]
dAllocs.append(dtemp)
#a tensor

print(BLUE, "domain allocs: ", dAllocs, ENDC)

# asset_allocs: per-domain list of per-asset 1-element tensors
aAllocs = []
for domain_idx, (domain, assets) in enumerate(domain_wise_tickers.items()):
    per_domain = [
        torch.tensor((1 / num_domains) / len(assets), dtype=torch.float32, device=device_asset)
        for _ in assets
    ]
    aAllocs.append(per_domain)
#an array of tensors

print(BLUE, "asset allocs initialized: ", aAllocs, ENDC)

# current holdings (keep as python numbers or tensors as you prefer)
domain_cur_holdings = [0 for _ in range(num_domains)]
asset_cur_holdings = [[0 for _ in assets] for assets in domain_wise_tickers.values()]


asset_mem_param = 0.7
domain_mem_param = 0.7
master_mem_param = 0.7


def test():

    global dAllocs, aAllocs, dMem, mMem
    print(GREEN + f"starting test..." + ENDC)
    num_windows = test_data["AAPL"].shape[0]
    for window_idx in range(num_windows):

        asset_to_domain_signals = []  # [num_domains][num_assets][ast_to_dom_dim]
        domain_to_master_signals = []  # [num_domains,batch, dtom_dim] 
        master_returns = 0

        for domain, assets in domain_wise_tickers.items():

            domain_idx = domain_indices[domain]
            asset_to_domain_sig_domain = []
            r_t_domain = 0.0

            for asset in assets:
                asset_idx = asset_indices[asset]

                seq_tensor = torch.tensor(test_data[asset][window_idx], dtype=torch.float32) #(window timesteps, columns)
                alloc_val = aAllocs[domain_idx][asset_idx]  #  this is not batch first when printed.
                mem_val = asset_mems[domain_idx][asset_idx]

                if alloc_val.ndim == 1:
                    alloc_val = alloc_val.unsqueeze(-1)  # (batch, 1)
                
                elif alloc_val.ndim == 0:
                    alloc_val = alloc_val.unsqueeze(0) .unsqueeze(0) # (batch, 1)
                else:
                    pass

                non_seq_tensor = torch.cat([alloc_val, mem_val], dim=-1)  # (batch, 2)

                seq_in = seq_tensor.unsqueeze(0).to(AssetAG.device)
                non_seq_in = non_seq_tensor.to(AssetAG.device)

                actions, total_logprob, value = AssetAG.act(seq_in, non_seq_in) #put in the form of batch first

                bhs = actions[0].detach() 
                ast_to_dom = actions[1].detach() #detach completely removes the new tensor from the current computational graph 
                mem_update = actions[2].detach()
                trade_frac = actions[3].detach() #not adding squeeze(0) so the first dimension is still batch.

                with torch.no_grad():

                    p_t = test_data[asset][window_idx+1][0][0] #close price of first timestep of next window
                    if window_idx < num_windows - 1:
                        p_t_plus_1 = test_data[asset][window_idx][0][0]
                        frac = p_t_plus_1 / p_t

                        diff = aAllocs[domain_idx][asset_idx] - asset_cur_holdings[domain_idx][asset_idx]
                        bhs_signal = bhs[0] #batch first
                        if(diff > 0):
                            #max allocation > current holding -> can b,h,s
                            if bhs_signal == 0:
                                #buy
                                w_asset = aAllocs[domain_idx][asset_idx]
                                r_t = w_asset * (frac - 1) #we bought, so reward is added. 
                            # print(YELLOW + f"Asset: {asset}, p_t: {p_t}, p_t+1: {p_t_plus_1}, frac: {frac}, w_asset: {w_asset}" + ENDC)
                            elif bhs_signal == 1:
                                #hold
                                r_t = 0.0
                            else:
                                #sell
                                w_asset = aAllocs[domain_idx][asset_idx]
                                r_t = w_asset * -1 * (frac - 1) #we bought, so reward is subtracted (into -1).
                        
                        elif(diff == 0):
                            #equal -> can only h,s
                            if bhs_signal == 0:
                                #buy -> no action possible
                                r_t = 0.0
                            elif bhs_signal == 1:
                                #hold
                                r_t = 0.0
                            else:
                                #sell
                                w_asset = aAllocs[domain_idx][asset_idx]
                                r_t = w_asset * -1 * (frac - 1) #we bought, so reward is subtracted (into -1).
                        
                        else:
                            #max allocation < current holding -> can only s
                            if bhs_signal == 0:
                                #buy -> no action possible
                                r_t = 0.0
                            elif bhs_signal == 1:
                                #hold
                                r_t = 0.0
                            else:
                                #sell
                                w_asset = aAllocs[domain_idx][asset_idx]
                                r_t = w_asset * -1 * (frac - 1) #we bought, so reward is subtracted (into -1).

                        r_t = torch.tensor(r_t, dtype=torch.float32, device=AssetAG.device)
                        net_testing_returns[domain_idx][asset_idx] += r_t
                        r_t_domain += r_t
                    
                    asset_to_domain_sig_domain.append(ast_to_dom) #(assets, batch, ...) because the first dimension of every ast_to_dom
                    #is batch and we are appending various ast_to_doms in the list above.
                    asset_mems[domain_idx][asset_idx] += asset_mem_param * mem_update


            with torch.no_grad():
                asset_to_domain_sig_domain = (
                    torch.stack(asset_to_domain_sig_domain)    # (num_assets, batch, dim)
                    .permute(1, 0, 2)                          # → (batch, num_assets, dim)
                    .to(DomainAG.device)
                )

            alloc_val = dAllocs[0][domain_idx]

            if alloc_val.ndim == 1:
                alloc_val = alloc_val.unsqueeze(0)  # (batch, 1)
            
            elif alloc_val.ndim == 0:
                alloc_val = alloc_val.unsqueeze(0).unsqueeze(0) # (batch, 1) => adding 2 dims to match h_assets
            elif alloc_val.ndim == 3:
                alloc_val = alloc_val.squeeze(0)  #remove extra dim if already present
            else:
                    pass
            
            actions, total_logprob, value, alloc_distn = DomainAG.act(asset_to_domain_sig_domain, alloc_val, domain_mem[domain_idx])

            with torch.no_grad():

                allocations = actions[0].detach() #Keeping batch dimensions
                print(GREEN , "all allocations (domain output)" , allocations, ENDC)    
                dtom = actions[1].detach()
                mem_update = actions[2].detach()

                aAllocs[domain_idx] = allocations.squeeze(0)  #remove batch dim
                domain_mem[domain_idx] += domain_mem_param * mem_update

                domain_to_master_signals.append(dtom)
                master_returns += dAllocs[0][domain_idx] * r_t_domain
                print(MAGENTA + "domain done" + ENDC)
        
        with torch.no_grad():
            # master
            domain_to_master_signals = (
                    torch.stack(domain_to_master_signals)    # [num_domains,batch, dtom_dim] 
                    .permute(1, 0, 2)                          # → (batch, num_domains, dtom_dim)
                    .to(MasterAG.device)
            )

        master_actions, total_logprob, value, alloc_distn = MasterAG.act(domain_to_master_signals, mMem)

        with torch.no_grad():

            allocations = master_actions[0].detach()  #keeping batch first => that's why not applying squeeze(0)
            mem_update = master_actions[1].detach()
            dAllocs = allocations #keep as it is                     #DOUBT HERE ******
            print(GREEN , "all allocations (master output)" , allocations, ENDC)
            mMem += master_mem_param * mem_update

    
    print(WHITE + f"Testing ended!" + ENDC)
    print(WHITE + f"net returns: ")
    print("Initial allocated value for each asset: ", aAllocs[0])
    

