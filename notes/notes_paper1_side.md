The score is a term for "log probability density with respect to the data" - $\nabla_x \log~p(x)$. The score lives in this landscape of hills and valleys and tells which direction is the uphill (toward higher probability or more realistic data) and also how steep the hill is.

Score Matching is the technique used to train a neural network to estimate that score function. However, calculating the score of a clean, complex dataset is hard.

Denoising Score Matching (DSM) simplifies this by:

1. Taking a clean data point $x_0$.
2. Adding a bit of noise to it to get a "noisy" version $x_t$.
3. Training the model to predict the noise that was added (or, equivalently, the direction back to the clean data).

The DDPM paper proved that minimizing the difference between your model's prediction and the added noise is mathematically equivalent to matching the score of the noisy data distribution.