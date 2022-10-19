# NLP-GPT-Fairytale
This repository is a small project to generate fairytales using the GPT architecture. The dataset can be obtained from [here]( https://raw.githubusercontent.com/SuzannaWentzel/FairyTale-Generator/main/data/Fairytales_db/merged_clean.txt). Alternatively, run the command
```
wget https://raw.githubusercontent.com/SuzannaWentzel/FairyTale-Generator/main/data/Fairytales_db/merged_clean.txt
```
to get the data. For convenience, the data is also included in the repository's release. The model's default settings uses a window of 50 words, 6 decoder layers and a hidden size of 512 with 8 heads.

## Model Training and Inference
To train the model, first generate the word tokens via
```
python clean_fairytale_data.py
```
followed by training the model with the command
```
python train_fairytale_gpt.py
```
Note that the model aggregates weight updates across 4 sub-batches of 128 samples before updating the weights to stabilize the model training. After the model is trained, run
```
python infer_fairytale_gpt.py
```
to generate samples of fairytales. 

## Generated Sample

