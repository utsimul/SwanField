# SwanField - Prototype version

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

