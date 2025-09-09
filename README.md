
# 🔄 Global Adaptive Recurrent Optimizer (GARO)

## Overview

This project explores a **theory of recurrent optimization**:
Instead of treating learning rate, momentum, and weight decay as *fixed constants*, we let them **emerge dynamically from the optimizer’s hidden state**.

The optimizer itself becomes a **recurrent neural system**, evolving online as training progresses.
At each step, it generates new hyperparameters (`lr_t`, `mom_t`, `wd_t`) conditioned on the gradient signal and its own hidden dynamics.

---

## ✨ Key Idea

> An optimizer can be modeled as a **recurrent dynamical system**, where its hidden state encodes optimization history and generates **adaptive hyperparameters** in real time.

* Traditional SGD:

  * `lr`, `momentum`, and `weight decay` are constants.
* Online Recurrent SGD (ours):

  * Hidden state evolves with training.
  * Hyperparameters are **functions of the hidden state**.
  * Updates are global, shared across all model parameters.

This bridges the gap between:

* **Static optimizers** (SGD, Adam, AdamW).
* **Learned optimizers** (meta-learned RNNs).

---

## 🔧 Usage

```python
model = MyModel()
opt = GlobalAdaptiveSGD(model.parameters())

for step in range(num_steps):
    loss = compute_loss(batch, model)

    opt.zero_grad()
    loss.backward()
    opt.step()
```

Just like `torch.optim.SGD`, but the hyperparameters now evolve automatically.

---

## ⚡ Implementation Details

* **Recurrent Cell**: A GRU maintains the optimizer’s hidden state.
* **Hyperparameter Heads**:

  * `lr_t = softplus(W_lr h_t + b_lr)`
  * `mom_t = tanh(W_mom h_t + b_mom)`
  * `wd_t = softplus(W_wd h_t + b_wd)`
* **Global Control**: One hidden state, shared across all parameters (cheaper + more stable).
* **Initialization**: Starts near vanilla SGD (`lr≈0.01, momentum≈0, wd≈0`).

---

## ✅ Behavior

* At **step 0**, behaves like plain SGD.
* Over time, the optimizer’s **hidden dynamics adapt**, leading to time-varying hyperparameters.
* Effectively acts as a built-in scheduler and controller without manual tuning.

---

## 📚 Theory

* The optimizer is treated as a **stateful system**:

  ```
  h_t = RNN(grad_t, h_{t-1})
  θ_{t+1} = θ_t - f(h_t) * update_t
  ```
* Hyperparameters are not externally chosen — they **emerge online**.
* This is equivalent to embedding the optimization process inside a **recurrent policy**.

---

## 🚀 Applications

* Training stability without hand-tuned schedules.
* Potential meta-learning foundation: optimizer can be *learned* rather than hand-designed.
* Could generalize across tasks, datasets, and architectures.

---

## ⚠️ Caveats

* Currently the recurrent weights are **frozen** (not meta-trained).
* Adds extra computation vs. vanilla SGD.
* Global hyperparameters → simpler but less expressive than per-layer approaches.

---

## 🧭 Next Steps

* Add logging hooks to observe `lr_t`, `mom_t`, `wd_t`.
* Explore **layer-wise recurrence**.
* Try meta-training the recurrent weights.
* Benchmark vs. Adam/AdamW on common tasks.
