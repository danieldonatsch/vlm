# VLM
Some Vision-Language (Model) Experiments


## CLIP

In `clip.py` is a very small example of a clip-style model.
The goal is to find a common feature space between a vision and a language encoder.
We train a total of four models at the same time: 
Vision encoder (CNN), language encoder (the mean of the embeddings) and two linear projections.
The loss is built such, that the encoded text and the encoded image of corresponding text-image pairs are similar vectors. 
