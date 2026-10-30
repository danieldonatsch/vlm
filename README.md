# VLM
Some Vision-Language (Model) Experiments


## CLIP

In `clip.py` is a very small example of a clip-style model.
The goal is to find a common feature space between a vision and a language encoder.
We train a total of four models at the same time: 
Vision encoder (CNN), language encoder (the mean of the embeddings) and two linear projections.
The loss is built such, that the encoded text and the encoded image of corresponding text-image pairs are similar vectors. 

## Image captioning

`train_captioner.py` trains a small decoder-only transformer (`CaptionDecoder` in `vlm_models.py`)
that generates captions such as "red circle".
The image embedding is projected to one "visual token" in front of the caption tokens,
and the decoder learns to predict the next word (teacher forcing, cross-entropy).
At test time, captions are generated greedily, word by word, until `<eos>`.

```
# Use the frozen image encoder of a trained CLIP checkpoint
python train_captioner.py --checkpoint-path out_clip/<run>/clip_epoch_020.pt
# Train image encoder and decoder from scratch
python train_captioner.py
```
