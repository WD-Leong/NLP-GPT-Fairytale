import time
import numpy as np
import pandas as pd
import pickle as pkl
from nltk import wordpunct_tokenize

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_gpt_keras_v2 as tf_gpt

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_encode, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = x_encode[id_st:id_en, :]
        tmp_output = x_output[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            output_logits = model(tmp_encode, training=True)
            
            tmp_losses = tf.reduce_sum(tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=output_logits), axis=1))
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clipped_gradients, _ = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_losses

# Model Parameters. #
prob_keep  = 0.9
batch_size = 512
sub_batch  = 128
num_heads  = 8
num_layers = 6
seq_length = 50

gradient_clip = 1.00
maximum_iter  = 100000
restore_flag  = True
save_step     = 250
warmup_steps  = 25000
display_step  = 50
anneal_step   = 2500
anneal_rate   = 0.75

hidden_size = 512
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = maximum_iter

model_ckpt_dir  = "TF_Models/fairytale_gpt"
train_loss_file = "train_loss_fairytale_gpt.csv"

# Load the data. #
tmp_pkl_file = "data/fairytale_words_"
tmp_pkl_file += "data_window_50.pkl"
with open(tmp_pkl_file, "rb") as tmp_load:
    fairytale_data = pkl.load(tmp_load)

# Extract the data and its assets. #
word_vocab = fairytale_data["word_vocab"]
word_2_idx = fairytale_data["word_2_idx"]
idx_2_word = fairytale_data["idx_2_word"]
vocab_size = len(word_vocab)

# Define the special tokens. #
SOS_token = word_2_idx["[SOS]"]
EOS_token = word_2_idx["[EOS]"]
PAD_token = word_2_idx["[PAD]"]
UNK_token = word_2_idx["[UNK]"]
print("Vocabulary Size:", str(vocab_size))

num_data = len(fairytale_data["input_array"])
input_array = fairytale_data["input_array"]
label_array = fairytale_data["label_array"]
print("Total of", num_data, "rows loaded.")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the Transformer. #
print("Building the GPT Keras Model.")
start_time = time.time()

gpt_model = tf_gpt.GPTDecoder(
    num_layers, num_heads, 
    hidden_size, ffwd_size, vocab_size, 
    seq_length, rate1=0.0, rate2=1.0-prob_keep)
gpt_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

elapsed_time = (time.time()-start_time) / 60
print("GPT Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optimizer=gpt_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(
            manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Train the GPT model. #
tmp_out_seq = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_length], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.005
    anneal_pow = int(n_iter / anneal_step)
    learning_rate = max(np.power(
        anneal_rate, anneal_pow)*initial_lr, 2.5e-5)

print("-" * 50)
print("Training the GPT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            anneal_pow = int(n_iter / anneal_step)
            learning_rate = max(np.power(
                anneal_rate, anneal_pow)*initial_lr, 2.5e-5)
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    # Set the training data. #
    tmp_input  = input_array[batch_sample, :]
    tmp_output = label_array[batch_sample, :]
    
    tmp_loss = sub_batch_train_step(
        gpt_model, sub_batch, tmp_input, tmp_output, 
        gpt_optimizer, learning_rate=learning_rate)
    
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_loss = tot_loss / display_step
        avg_ppl  = np.log2(avg_loss)
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        tmp_prompt = "once"
        tmp_tok_id = [SOS_token]
        tmp_tok_id += [
            word_2_idx.get(x, UNK_token) for \
                x in wordpunct_tokenize(tmp_prompt)]
        tmp_tokens = [idx_2_word[x] for x in tmp_tok_id]
        num_tokens = len(tmp_tok_id)
        num_sample = np.random.choice(num_tokens)
        
        tmp_test_in = np.array(
            tmp_tok_id, dtype=np.int32)
        tmp_in_phrase  = " ".join(
            [idx_2_word[x] for x in tmp_test_in])
        tmp_out_phrase = " ".join(tmp_tokens)
        
        gen_tokens = gpt_model.gen_text(
            tmp_test_in.reshape((1, -1)), 250).numpy()[0]
        gen_phrase = [idx_2_word[x] for x in gen_tokens]
        gen_phrase = " ".join(gen_phrase)
        
        print("Iteration", str(n_iter) + ".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")
        print("Average Loss:", str(avg_loss) + ".")
        print("Average Perplexity:", str(avg_ppl) + ".")
        
        print("")
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        
        train_loss_list.append((n_iter, avg_loss, avg_ppl))
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_loss_cols = ["n_iter", "xent_loss", "perplexity"]
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=tmp_loss_cols)
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)

