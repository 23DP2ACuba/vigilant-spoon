# vigilant-spoon
```yaml
                  ┌─────────────┐
                  │   Features  │
                  └─────┬───────┘
                        ↓
         ┌────────────────────────────┐
         │ Gaussian HMM / KMeans      │←───────┐
         └────────────┬──────────────┘         │
                      ↓                        │
   ┌────────────── Multi-MLP Ensemble ─────────┘
   │     MLP1 ─┬─ MLP2 ─┬─ MLP3 (Regime-aware) │
   └────────────┬───────────────┬──────────────┘
                ↓               ↓
          Softmax outputs   + Features + Regime Labels
                         ↓
               Temporal Evaluation Module
                       (BERT)
                         ↓
               Trade Signal / Position Sizing
                         ↓
                   Execution Layer
```
