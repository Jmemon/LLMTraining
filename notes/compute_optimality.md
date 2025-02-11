### What is compute optimality?
Compute optimality is the notion that when training large language models (or other neural networks), you want to allocate your fixed compute budget in a balanced way between the model’s size (the number of parameters, often denoted by P) and the amount of data (training tokens, denoted by D) used during training. The basic intuition is that many training regimes follow a cost model roughly given by:
  Compute ≈ constant × P × D
If you have too few tokens for a given model size, the model might not see enough data to use its full capacity (leading to underfitting). Conversely, if you train on a huge amount of data but the model is too small, the model’s capacity becomes the bottleneck.

### How do I calculate optimal token count given a model size?
20 toks per parameter. So 
10M params = 200M toks. 
100M params = 2B toks.
1B params = 20B toks.

### Vice versa?
So then divide by 20.

### Thoughts
I'm slightly suspicious. I feel like there must be more than just volume of data. Does the way data is fed in matter? Does architecture matter? 
