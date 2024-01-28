# Capstone 

# MultiModal Phi2

<br>

# Task    

You are going to make a multi-modal LLM that can take these inputs:
1. Text
2. Image
3. Audio 

The output remains text for us. Here are the guidelines:   

### Training:   

1. Image:
    - Use the original Instruct 150k dataset, and use CLIP to get the image embeddings:   
    - either run it in real-time (need more GPU), or   
    - store the embeddings (basically preprocess and store)      
    - Add your projection layer from this CLIP embeddings to something that can be fed to your Phi Model (do not apply QLoRa to this payer)   
    - Add an adapter that you'll train (QLoRa) on the instruct 150k dataset

<br>

2. Audio:   
    - You need to use Whisper to perform ASR. 
    - Add a projection layer for whisper output (which is text only)
    - This audio part "should" not require any training, but a pipeline from your end to link it properly to your model

<br>

3. Text:   
    - You are going to use Microsoft's Phi-2 or any other model and generate data. Recommend you generate this data in parallel, don't generate and store everything as that would be a very very large dataset
    - You are going to collect "some" clean data (100MB when zipped). This data CAN be generated from Phi-2 and stored.
    - You are going to use the same tokenizer and other data structures 
    - You are going to use AWS (or an equvalent system) where you are going to train YOUR model. 
    - You are going to train YOUR model. Train it somehow to reach the "initial loss - 1" value. Compare it with the final Microsoft's Phi 2's value and see how much more you have to train!!!
    - Then you're going to take the default Phi-2 model (Microsoft's version and move to the next step)
    - You'll start fine-tuning the model with QLORA (Fine-tuning & Multimodality)

<br>

# Solution

## Areas of Improvement

1. More datasets can be used for both pre-training and fine-tuning.
2. Number of layers in the Projection model can be experimented for better predictions.
3. More Audio libraries can be implemented for better audio extraction and transcription.
4. Lighter version of CLIP can be tried.

<br>

# Pre-training Logs

```python
iter 0 step 0: loss 10.7340, LR: 0.000000, iter time: 1210.67ms
iter 100 step 25: loss 7.9553, LR: 0.000300, iter time: 100.06ms
iter 200 step 50: loss 7.9136, LR: 0.000600, iter time: 95.83ms
iter 300 step 75: loss 8.9131, LR: 0.000900, iter time: 95.73ms
iter 400 step 100: loss 9.0141, LR: 0.001200, iter time: 97.15ms
iter 500 step 125: loss 8.8424, LR: 0.001500, iter time: 97.29ms
iter 600 step 150: loss 9.1321, LR: 0.001800, iter time: 97.54ms
iter 700 step 175: loss 7.5628, LR: 0.002100, iter time: 96.22ms
iter 800 step 200: loss 8.1672, LR: 0.002400, iter time: 98.89ms
iter 900 step 225: loss 7.8912, LR: 0.002700, iter time: 95.66ms
iter 1000 step 250: loss 8.5327, LR: 0.003000, iter time: 97.23ms
iter 1100 step 275: loss 6.1776, LR: 0.003300, iter time: 96.62ms
iter 1200 step 300: loss 8.9131, LR: 0.003600, iter time: 93.58ms
iter 1300 step 325: loss 7.9728, LR: 0.003900, iter time: 96.73ms
iter 1400 step 350: loss 8.6512, LR: 0.004200, iter time: 95.93ms
iter 1500 step 375: loss 8.5109, LR: 0.004500, iter time: 96.44ms
iter 1600 step 400: loss 8.1521, LR: 0.004800, iter time: 94.96ms
iter 1700 step 425: loss 8.5120, LR: 0.005100, iter time: 94.43ms
iter 1800 step 450: loss 7.7462, LR: 0.005400, iter time: 96.84ms
iter 1900 step 475: loss 7.4216, LR: 0.005700, iter time: 95.84ms
iter 2000 step 500: loss 7.2980, LR: 0.006000, iter time: 91.55ms
iter 2100 step 525: loss 8.5015, LR: 0.005998, iter time: 95.05ms
iter 2200 step 550: loss 7.4835, LR: 0.005991, iter time: 96.98ms
iter 2300 step 575: loss 7.5653, LR: 0.005979, iter time: 97.88ms
iter 2400 step 600: loss 7.6941, LR: 0.005963, iter time: 96.96ms
iter 2500 step 625: loss 8.0123, LR: 0.005942, iter time: 95.29ms
iter 2600 step 650: loss 8.1154, LR: 0.005917, iter time: 96.71ms
iter 2700 step 675: loss 6.5120, LR: 0.005887, iter time: 96.51ms
iter 2800 step 700: loss 7.7218, LR: 0.005853, iter time: 96.25ms
iter 2900 step 725: loss 7.4919, LR: 0.005815, iter time: 97.89ms
iter 3000 step 750: loss 7.1346, LR: 0.005772, iter time: 97.59ms 
```

<br>

# Projection Layer Training Logs

```python
Epoch : 1/15
Loss  : 7.161831855773926
Caption    :  ['A bedroom with a bed and small table near by.']
Prediction :  ['_, to\n\n, a view, a table. the.\nThe']
==============================
Epoch : 2/15
Loss  : 6.663324356079102
Caption    :  ['A man riding a brown horse in uniform next to tall green trees.']
Prediction :  [',_\n\n\n who a bike horse is a. to a buildings trees.\nA\nIN']
==============================
Epoch : 3/15
Loss  : 6.508380889892578
Caption    :  ['A couple of computer monitors sitting on top of a wooden desk.']
Prediction :  ['_\n\n otherux of years science are on a of each desk desk.\nA#']
==============================
Epoch : 4/15
Loss  : 6.26915168762207
Caption    :  ['A woman in a hat sitting next to luggage.']
Prediction :  [",_.ayactions's the red Online on to a.\nAThe\n\n\n\nThe"]
==============================
Epoch : 5/15
Loss  : 6.324067115783691
Caption    :  ['A child holding chocolate donut with both hands.']
Prediction :  ['_",\',.men\'s a\n\'t\n a hands.\nTheTheTheTheThe']
==============================
Epoch : 6/15
Loss  : 6.472545146942139
Caption    :  ['two zebras are standing together in the woods']
Prediction :  [",\n')'\neraas twenty a in in a jungle.TheThe"]
==============================
Epoch : 7/15
Loss  : 6.051589488983154
Caption    :  ['An orange truck driving down a street full of men in the back.']
Prediction :  ['."\n otherxiety orange orangeA the road" of people walking suits middle of\nTheTheThe']
==============================
Epoch : 8/15
Loss  : 5.945560932159424
Caption    :  ['A couple of computer monitors sitting on top of a wooden desk.']
Prediction :  ["_',\n yetux of weeks science the on a of each desk desk with\nTheThe\n\n\nThe"]
==============================
Epoch : 9/15
Loss  : 5.876091480255127
Caption    :  ['A man standing in front of a clock.']
Prediction :  [',_\n"ffect woman on a of a building a']
==============================
Epoch : 10/15
Loss  : 6.132306098937988
Caption    :  ['A woman taking pictures on a busy street.']
Prediction :  ['__" orangeuxAAA the beach street in\nTheThe\nThe']
==============================
Epoch : 11/15
Loss  : 6.142520904541016
Caption    :  ['A bedroom with a bed and small table near by.']
Prediction :  ['."\n ofwardB a bathroom a a desk and the the TheTheThe']
==============================
Epoch : 12/15
Loss  : 5.425280570983887
Caption    :  ['a person walking on a snow covered field.']
Prediction :  ["\n''. Output'.\n\n the pathman ground the"]
==============================
Epoch : 13/15
Loss  : 5.1739821434021
Caption    :  ['a woman siting at a restaurant table with a plate of mexican food on it ']
Prediction :  ['_"\n\' sad\'. study\n the table. with a man of foodxican food and it andiph']
==============================
Epoch : 14/15
Loss  : 4.745423316955566
Caption    :  ['Two Zebras grazing together in a grassy area.']
Prediction :  ['_"\n\') Threebrina Three\')? the fieldy me" Year']
==============================
Epoch : 15/15
Loss  : 4.2316107749938965
Caption    :  ['A child holding chocolate donut with both hands.']
Prediction :  [',"\n.xeA rabbit paper\'t at a hands.\nTheThe']
==============================
```

<br>

# QLoRA Fine-tuning Logs

```python
[500/500 06:35, Epoch 0/1]
Step	Training Loss
10	1.374600
20	1.488600
30	1.827800
40	1.869200
50	2.043000
60	1.438300
70	1.382600
80	1.726100
90	2.141600
100	2.376900
110	1.849200
120	1.272500
130	1.426700
140	1.907800
150	1.782700
160	1.105000
170	1.401800
180	1.518600
190	2.079800
200	1.901300
210	1.453600
220	1.542800
230	1.243900
240	1.863700
250	2.087200
260	1.168400
270	1.124000
280	1.380200
290	1.877700
300	2.130400
310	1.745800
320	1.550200
330	1.496400
340	2.263400
350	2.353900
360	1.185700
370	1.708900
380	1.941400
390	1.953500
400	1.940400
410	1.380900
420	1.494100
430	1.940900
440	2.140200
450	2.075200
460	1.225800
470	1.320300
480	1.272100
490	1.821100
500	2.140000

TrainOutput(global_step=500, training_loss=1.6947186126708984, metrics={'train_runtime': 441.0288, 'train_samples_per_second': 1.134, 'train_steps_per_second': 1.134, 'total_flos': 1949189771612160.0, 'train_loss': 1.6947186126708984, 'epoch': 0.06})
```