# NLP-GPT-Fairytale
This repository is a small project to generate fairytales using the GPT architecture. The data can be obtained from [here]( https://raw.githubusercontent.com/SuzannaWentzel/FairyTale-Generator/main/data/Fairytales_db/merged_clean.txt). 

To train the model, first generate the word tokens via
```
python clean_fairytale_data.py
```
followed by training the model with the command
```
python train_fairytale_gpt.py
```
After the model is trained, run
```
python infer_fairytale_gpt.py
```
to generate samples of fairytales.
