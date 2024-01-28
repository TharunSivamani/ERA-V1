# ERA V1 - Capstone 

# MultiModal Phi2

<br>

# Task    

You are going to modify phi-2 into a multi-modal LLM that can take these inputs:
1. Text
2. Image
3. Audio 

The output will be the text format of all those inputs.

<br>

### Training:   

1. **Image**:
    - Use the original Instruct 150k dataset, and use CLIP to get the image embeddings:   
    - either run it in real-time (need more GPU), or   
    - store the embeddings (basically preprocess and store)      
    - Add your projection layer from this CLIP embeddings to something that can be fed to your Phi Model (do not apply QLoRa to this payer)   
    - Add an adapter that you'll train (QLoRa) on the instruct 150k dataset

<br>

2. **Audio**:   
    - You need to use Whisper to perform ASR. 
    - Add a projection layer for whisper output (which is text only)
    - This audio part "should" not require any training, but a pipeline from your end to link it properly to your model

<br>

3. **Text**:   
    - You are going to use Microsoft's Phi-2 or any other model and generate data. Recommend you generate this data in parallel, don't generate and store everything as that would be a very very large dataset
    - You are going to collect "some" clean data (100MB when zipped). This data CAN be generated from Phi-2 and stored.
    - You are going to use the same tokenizer and other data structures 
    - You are going to use AWS (or an equvalent system) where you are going to train YOUR model. 
    - You are going to train YOUR model. Train it somehow to reach the "initial loss - 1" value. Compare it with the final Microsoft's Phi 2's value and see how much more you have to train!!!
    - Then you're going to take the default Phi-2 model (Microsoft's version and move to the next step)
    - You'll start fine-tuning the model with QLORA (Fine-tuning & Multimodality)

<br>

# Solution

<br>

## Areas of Improvement

1. Combination / Collection of various datasets can be used for both pre-training and fine-tuning.
2. Layer Count in the projection layer can be experimented and documented to find a better select.
3. More modelling for Audio can implemented for improved quality extraction and transcription.
4. Lighter version of CLIP can be trained and tried.

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
Epoch : 1 | Loss  : 7.007693767547607

Predicted :  [',,.,throp, is through a road. of people and suits middle.\nTheThe\n\n\n\n\n\nThe']
Caption   :  ['An orange truck driving down a street full of men in the back.']
----------------------------------------------
Epoch : 2 | Loss  : 6.2756452560424805

Predicted :  [',\n\n\n\n of years have a history and home table in\nThe']
Caption   :  ['A couple of people with a laptop at a table.']
----------------------------------------------
Epoch : 3 | Loss  : 6.2440924644470215

Predicted :  ['ed\n\n\n\n of years have a lot and a conference in\nTheThe\n\n\n\n\n\n\n\nThe']
Caption   :  ['A couple of people with a laptop at a table.']
----------------------------------------------
Epoch : 4 | Loss  : 6.318299293518066

Predicted :  [',,\n\n\n of years scientists are on a of a desk table.\nTheThe']
Caption   :  ['A couple of computer monitors sitting on top of a wooden desk.']
----------------------------------------------
Epoch : 5 | Loss  : 5.873800277709961

Predicted :  [',=,\n-hendas, not in. a field.The']
Caption   :  ['two zebras are standing together in the woods']
----------------------------------------------
Epoch : 6 | Loss  : 5.042659282684326

Predicted :  ['\n\n\n\n Dollarhas in in in the fieldy field.']
Caption   :  ['Two Zebras grazing together in a grassy area.']
----------------------------------------------
Epoch : 7 | Loss  : 4.50493049621582

Predicted :  ["._,\n =', a single, a table. the."]
Caption   :  ['A bedroom with a bed and small table near by.']
----------------------------------------------
Epoch : 8 | Loss  : 3.7070505619049072

Predicted :  ['..(\n\na through the tight- path.']
Caption   :  ['a person walking on a snow covered field.']
----------------------------------------------
Epoch : 9 | Loss  : 3.745819091796875

Predicted :  [' of.(\n =\'s the" and a man on a a box on']
Caption   :  ['A woman in black jacket watching a cat eating from pizza box.']
----------------------------------------------
Epoch : 10 | Loss  : 3.5547196865081787

Predicted :  ['\n.\n\n, of a few, appuccino. a to a desk case']
Caption   :  ['A tray holding a sandwich and cappuccino, next to the pastry.']
----------------------------------------------
Epoch : 11 | Loss  : 2.904674530029297

Predicted :  ['..end\nend in the red" on to a"']
Caption   :  ['A woman in a hat sitting next to luggage.']
----------------------------------------------
Epoch : 12 | Loss  : 2.89656925201416

Predicted :  ["._\n')\n who to do the road. a crosswalk."]
Caption   :  ['A man prepares to cross the street at a crosswalk']
----------------------------------------------
Epoch : 13 | Loss  : 2.564776659011841

Predicted :  [".(\n\n'), to make the bridge. a greenwalk."]
Caption   :  ['A man prepares to cross the street at a crosswalk']
----------------------------------------------
Epoch : 14 | Loss  : 2.400994062423706

Predicted :  ['..\n\nanOrangeload towards the road, of people and suits middle of\n\n\n\n\n\n\n\n<|endoftext|>\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n<|endoftext|>\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n<|endoftext|>\n\n\n\n\n\n<|endoftext|>']
Caption   :  ['An orange truck driving down a street full of men in the back.']
----------------------------------------------
Epoch : 15 | Loss  : 2.8281171321868896

Predicted :  ["\n.(\n\n's on on the table,, a man of foodzican food in it.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n<|endoftext|>\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n<|endoftext|>\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n<|endoftext|><|endoftext|>\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n<|endoftext|>"]
Caption   :  ['a woman siting at a restaurant table with a plate of mexican food on it ']
----------------------------------------------
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