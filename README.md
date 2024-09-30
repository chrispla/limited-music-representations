# Learning music audio representations with limited data
Code for “Learning Music Audio Representations with Limited Data”.

### Overview
***What happens when we train music audio representation models with very limited data?***
![image][(https://github.com/chrispla/limited-music-representations/blob/main/figures/overview.png)

We train
* MusiCNN with tagging
- VGGish with tagging
- AST with tagging
-  CLMR with self-supervised contrastive learning
- TMAE, a transformer-based masked autoencoder
on subsets of the MagnaTagATune music dataset, ranging from 5 to ~8000 minutes.

We extract representations from each, along with untrained models, and train downstream models on
* music tagging
- monophonic pitch detection
- monophonic instrument recognition

We show that, in certain cases,
* the representations from untrained and minimally-trained models perform comparatively to those from “fully-trained” models
- larger downstream models are able to "recover" performance from untrained and minimally-trained representations
- the inherent robustness of representations to noise is bad accross the board
- the performance gap to "hand-crafted" features is still significant in pitch and instrument recognition

### Reproduction
#### 1. Requirement installation:
```bash
pip install -r requirements.txt
```

#### 2. Pretraining:
MagnaTagATune will be downloaded automatically if it's not already present in `data/MTAT`. Each model has a training script, which can be run with:
```bash
python model_name/train.py
```
where `model_name` is one of `musicnn`, `vggish`, `ast`, `clmr`, or `tmae`.

#### 3. Feature extraction
MagnaTagATune, TinySOL, and Beatport will be downloaded automatically if they're not already present in `data/`.
```bash
python extract_features.py --model model_name --task task_name
```
where `model_name` is one of `musicnn`, `vggish`, `ast`, `clmr`, or `tmae`, and `task_name` is one of `tagging`, `pitch`, or `instrument`.

#### 4. Downstream training and evaluation
```bash
python downstream.py --model model_name --task task_name
```
where `model_name` is one of `musicnn`, `vggish`, `ast`, `clmr`, or `tmae`, and `task_name` is one of `tagging`, `pitch`, or `instrument`.

#### 5. Visualization
See visualization notebooks.
