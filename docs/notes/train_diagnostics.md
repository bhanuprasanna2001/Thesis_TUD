# Train diagnostics

You have built a model, and have a large dataset to train. How do you actually verify if the model you have built actually learns the dataset.

These are the approaches that I have found:

1. Overfit a single batch

- We can try to overfit the model with 1 - 2 batches so that the loss goes very near to zero.
- If the model can't memorize a tiny dataset, it is not yet ready.

```python
# In your training script
from torch.utils.data import Subset

# Take only 2 batches (e.g., 2 * 32 = 64 samples)
train_subset = Subset(train_dataset, range(64))
val_subset = Subset(val_dataset, range(64))

# Train with high learning rate, many epochs
model = Diffusion(lr=0.001)  # Higher LR
trainer = L.Trainer(max_epochs=100, overfit_batches=64)
trainer.fit(model, train_dataloader)

# Loss should go to near-zero (~0.001 or less)
# If it doesn't, something is broken in the model/training
```

2. Gradient flow checks

- It is always important to check how the gradients propagate through the layers.
- It is for to check vanishing/exploding gradients.
- There might be some places where gradient is 0, then some layer's aren't training at all. 

```python
# Check per-layer gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: {grad_norm:.6f}")
    else:
        print(f"{name}: NO GRADIENT!")  # This is a problem!
```

3. Visual inspection

- We can plot attention maps (for attention layers) for when dealing with self-attention.
- We can also visualize the intermediate activations.
- We can visualize the architecture as well.

3 ways to do:

**torch info**

```python
from torchinfo import summary

model = UNetDiffusion(in_channels=1, out_channels=1, base_channels=64)
summary(
    model, 
    input_data=[
        torch.randn(1, 1, 28, 28),  # X
        torch.randint(0, 1000, (1,))  # t
    ],
    depth=4,
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"]
)
```

**torch summary**

```python
from torchsummary import summary
summary(model, [(1, 28, 28), (1,)], device="cpu")
```

**torch fx**

```python
# 1st appraoch
from torch.fx import symbolic_trace
import torch.fx as fx

# Trace the model
traced = symbolic_trace(model)
print(traced.graph)  # Shows computation graph

# Export to graphviz
# traced.graph.print_tabular()

# 2nd approach
# Export to ONNX, then view in Netron app
import torch.onnx
dummy_x = torch.randn(1, 1, 28, 28)
dummy_t = torch.randint(0, 1000, (1,))
torch.onnx.export(model, (dummy_x, dummy_t), "model.onnx")
# Then: netron model.onnx
```

