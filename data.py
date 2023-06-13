import io
import re
import tensorflow as tf
from constant import *

def preprocess_sentence(w, max_length):
    w = w.lower().strip()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.strip()

    # Truncate Length up to ideal_length
    w = " ".join(w.split()[:max_length+1])
    
    # Add start and end token 
    w = '{} {} {}'.format(BOS, w, EOS)
    # w = '{} {}'.format(w, EOS)
    return w

def process_data(inp_path, targ_path, num_examples, max_length, inp_tokenizer=None, targ_tokenizer=None):
    
    inp_lines = io.open(inp_path, encoding=UTF_8).read().strip().split('\n')[:num_examples]
    targ_lines = io.open(targ_path, encoding=UTF_8).read().strip().split('\n')[:num_examples]
    
    print('=============Data================')
    print('----------------Begin--------------------')
    print('----------> Sample:')
    print('Input: ', inp_lines[3])
    print('Target: ', targ_lines[3])
    print('----------------End--------------------\n')
    
    inp_lines = [preprocess_sentence(inp, max_length) for inp in inp_lines]
    targ_lines = [preprocess_sentence(targ, max_length) for targ in targ_lines]
    
    print('=============Data================')
    print('----------------Begin--------------------')
    print('----------> Sample:')
    print('Input: ', inp_lines[3])
    print('Target: ', targ_lines[3])
    print('----------------End--------------------\n')
    
    if inp_tokenizer == None and targ_tokenizer == None:
        inp_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' !"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
        targ_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' !"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
        
        inp_tokenizer.fit_on_texts(inp_lines)
        targ_tokenizer.fit_on_texts(targ_lines)
        
        inp_to_index = inp_tokenizer.word_index
        targ_to_index = targ_tokenizer.word_index
        
    inp_tensor = inp_tokenizer.texts_to_sequences(inp_lines)
    targ_tensor = targ_tokenizer.texts_to_sequences(targ_lines)
    
    print('----------------Begin--------------------')
    print('----------> Sample:')
    print('Input: ', inp_tensor[3])
    print('Target: ', targ_tensor[3])
    print('----------------End--------------------\n')
    
    inp_tensor = tf.keras.preprocessing.sequence.pad_sequences(inp_tensor, padding='post', maxlen=max_length)
    targ_tensor = tf.keras.preprocessing.sequence.pad_sequences(targ_tensor, padding='post', maxlen=max_length)
    
    inp_tensor = tf.convert_to_tensor(inp_tensor, dtype=tf.int64)
    targ_tensor = tf.convert_to_tensor(targ_tensor, dtype=tf.int64)
    
    return inp_tensor, targ_tensor, inp_tokenizer, targ_tokenizer

def texts_to_sequences(tokenizer, sentences):
    tar = []
    for word in sentences.split():
        if tokenizer.word_index.get(word) != None :
            tar.append(tokenizer.word_index.get(word))
    return tar

def creat_dataset(inp_path, targ_path, num_examples, max_length, batch_size=30, buffer_size=8, inp_tokenizer=None, targ_tokenizer=None):
    inp_tensor, targ_tensor, inp_tokenizer, targ_tokenizer = process_data(inp_path, targ_path, num_examples, max_length, inp_tokenizer, targ_tokenizer)
    # Tạo dataset từ inp_tensor và targ_tensor
    dataset = tf.data.Dataset.from_tensor_slices((inp_tensor, targ_tensor))

    # Pha trộn và phân lô dữ liệu
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    # Đặt kiểu dữ liệu của batch
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.int64), tf.cast(y, tf.int64)))

    # Đặt prefetch để tải dữ liệu mẫu tiếp theo trong quá trình huấn luyện
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset, inp_tokenizer, targ_tokenizer
