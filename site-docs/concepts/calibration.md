# Calibration

Each answer's probabilities come from the option scores divided by a **temperature** and turned into percentages. A
temperature above 1 flattens them (less sure), below 1 sharpens them (more sure). The checkpoint stores one
temperature per question type (`choice`, `score`, `noul`), fitted after training on its validation sets, so on
those sets "80% sure" is right about 80% of the time.

## Why it can be off on your data

On your photos and your questions it may not be: the model can be overconfident on a domain it has not seen. Even on
the checkpoint's own validation sets calibration varies by set: pooled over all of them, the calibrated expected
calibration error (ECE) of the recommended checkpoint is small, but it is 0.16 on A-OKVQA, 0.035 on ScienceQA and
0.077 on VQAv2 yes/no ([checkpoints](../reference/checkpoints.md)). If you act on a probability threshold,
[calibrate on your own data](../how-to/calibrate.md) first.

## Accuracy does not change

Dividing every option's score by the same positive number keeps their order. The chosen option, the `score` level
with the highest probability and which side of 0.5 a `noul` falls on all stay the same; only how sure the model says
it is moves. The `score` field (the expected level) does move a little, since it averages over the probabilities.

## A probability is conditional on your options

The probabilities are a softmax over the options you list. A question whose right answer is not among them still
gets a confident-looking distribution. On a photo of a washing machine, a `choice` between electronics, clothing,
furniture, food and other put "food" first (32%); with "home appliance" added it was 90%. Add an option such as
"other" or "none of these" when the answer may not be listed.
