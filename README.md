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
The fairytale text generated at the 16,500 iteration is:
```
Input Phrase:
[SOS] once
```
`[SOS]` once upon a time there lived a king who had a wife and three sons , and they were all well fond of their mother , and they loved each other dearly , and the elder called one of his sons , and said to him , ' i am going to marry the witch that she may choose .' this made the girl look rather pale , but she would blush away . ' i never ,' she said , ' i am afraid you are a dead man .' ' why did you believe me ?' ' because i did not know you were here , though i was , to see if i had not known anything to you . but i have never seen you before .' ' oh , yes , i am ,' answered the princess , and a little feeble voice pressed forward . ' i am the king of the gnomes ,' she said to the other messengers , ' but if you will only bring me my golden sword and your head your head will no longer sit here , and then mount the colt which is taller than i .' on each side of the mountain they were all sitting a huge black horse , with long ears and long black hair , and his eyes were as large as saucers . ' now i am tired of waiting for my brother ,' said he , ' and i shall...
