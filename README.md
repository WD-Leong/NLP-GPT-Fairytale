# NLP-GPT-Fairytale
This repository is a small project to generate fairytales using the GPT architecture. The dataset can be obtained from [here]( https://raw.githubusercontent.com/SuzannaWentzel/FairyTale-Generator/main/data/Fairytales_db/merged_clean.txt). Alternatively, run the command
```
wget https://raw.githubusercontent.com/SuzannaWentzel/FairyTale-Generator/main/data/Fairytales_db/merged_clean.txt
```
to get the data. For convenience, the data is also included in the repository's release. The model's default settings uses a window of 50 words, 6 decoder layers and a hidden size of 512 with 8 heads. A `[SOS]` token is added to the beginning of each fairytale and an `[EOS]` token is added at the end. 

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
to generate samples of fairytales. This model has approximately 41M parameters.

## Generated Samples
The fairytale text generated at the 55,000 iteration is:
```
Input Phrase:
[SOS] once
```
`[SOS]` once upon a time there lived a king who had a son who was a very wicked and wicked , and he was determined to marry . he took his hazel aside , and tried to [UNK] his arms , but it was impossible to do him good to do so much . at last he grew angry , and his eyes began to [UNK] and he gave two strokes of his wand upon the golden branch , and with it the golden star on his breast , and the leaves turned to the air . the earth trembled as if with a sigh , and the mother saw the body ready to fall . " i' ll have you ," she said , " because i have done the work of the church ." the girl was taken out of the house and fed them ; but no one knew how to manage it , and did not know the way . he went into the yard . he got up and began to dance , and the mangot up and stretched himself . and then there was a great crowd assembled to witness the ceremony , for the young people were especially fond of the great warrior , and would avenge him on the other king , his enemy . and he bade her keep the broken pieces of the sword , to make a new sword for his son , and that blade should be...

```
Input Phrase:
[SOS] this is a
```
`[SOS]` this is a story which happened in the old time , and it was said that the king had killed the crab , who was then a few days after , heard the old man ' s summons , and was wiping away in a moment the creature had disappeared with the softest voice . ' i have taken the wrong basket -- by mistake , of course ,' said he . ' here is a purse ; take it and say to it , " dear purse , give me some money ," and you will get as much as you can want but the charm will only work if you promise to remain three years ." the old man listened in dismay to these words , but with an effort he thanked the emperor for his kindness and left the palace , wondering how he was to fulfil the task allotted to him . luckily for him , the emperor ' s daughter had overheard everything her father had said , and peeping through a curtain had seen the youth , and thought him handsomer than anyone she had ever beheld . at the sight of a man she seemed delighted that he had been allowed to go out to play once in the morning , and when she had finished she said to her : ' i am too tired when i come home in the evening to clean up the house .' and the two friends...
