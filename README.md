# GRAND

Welcome to the pre-release repository for the **GRAND** method, as introduced in our paper, *Machine Learning Vast Dynamic Conditional Covariance Matrices: the Spirit of "Divide and Conquer"*. You can access the paper on ***coming soon***.

## Environment

- **Main Settings:** Python 3.10.13 & Pytorch 2.5.1 & CUDA 12.5
- **Minor Settings:** To be completed.

## Reproduce the Results

```bash
# Step 1: Construct features and labels for quantile models.
python data_pipe/feature.py
python data_pipe/label.py

# Step 2: Train models
# Mean model 
python network/train.py --tau 0.0 --mse-loss
# Quantile models
python network/train.py --tau 0.005
python network/train.py --tau 0.01
...
python network/train.py --tau 0.99
python network/train.py --tau 0.995

# Step 3: Inference for quantiles and QCM learning
python network/infer.py

# Step 4: Construct features and labels for sigma models.
python data_pipe/sigma_feature.py
python data_pipe/sigma_label.py

# Step 5: Train variance model
python network/train_sigma.py

# Step 6: Train node fusion models
python network/train_mlp.py --tau 0.005
python network/train_mlp.py --tau 0.01
...
python network/train_mlp.py --tau 0.99
python network/train_mlp.py --tau 0.995

# Step 7: Estimate conditional covariance matrics
python network/estimate_variance.py 
```


Detailed usage instructions will be provided soon.