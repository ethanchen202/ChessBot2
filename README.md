# Deep Learning Chess Bot

## Methodology

Using similar data preprocessing techniques to Google's AlphaZero model. Architecture is a standard ViT with a policy head of output dimension $(4672,)$ and value head of a single scalar output. Model pretrained on publicly available CCRL40-15 data. Currently implementing a monte carlo tree search algorithm for self-play game generation used for reinforcement learning.