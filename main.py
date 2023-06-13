import os
from argparse import ArgumentParser
import tensorflow as tf
import logging
from data import creat_dataset
from transformer import Transformer
from optimizer import CustomSchedule
from loss import loss_function
from train import Train

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    parser = ArgumentParser()
    home_dir = os.getcwd()
    parser.add_argument("--input-path", default='{}/data/train/train.vi'.format(home_dir), type=str)
    parser.add_argument("--target-path", default='{}/data/train/train.en'.format(home_dir), type=str)
    parser.add_argument("--validation-input-path", default='{}/data/validation/validation.vi'.format(home_dir), type=str)
    parser.add_argument("--validation-target-path", default='{}/data/validation/validation.en'.format(home_dir), type=str)
    parser.add_argument("--checkpoint-folder", default='{}/checkpoints_vi_en/train'.format(home_dir), type=str)
    parser.add_argument("--buffer-size", default=8, type=str)
    parser.add_argument("--batch-size", default=30, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--max-length", default=100, type=int)
    parser.add_argument("--num-examples", default=100000, type=int)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--num-heads", default=16, type=int)
    parser.add_argument("--dff", default=128, type=int)
    parser.add_argument("--dropout-rate", default=0.1, type=float)

    args = parser.parse_args()

    print('---------------------Hello World-------------------')
    print('Github: nguyenvannghiem0312')
    print('Email: nguyenvannghiem0312@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Transfomer model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')


    dataset, inp_tokenizer, targ_tokenizer = creat_dataset(args.input_path, args.target_path, args.num_examples, args.max_length, args.batch_size, args.buffer_size)
    val_dataset = creat_dataset(args.validation_input_path, args.validation_target_path, args.num_examples, args.max_length, args.batch_size, args.buffer_size, inp_tokenizer, targ_tokenizer)

    learning_rate = CustomSchedule(args.d_model, 4000)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

    # Set checkpoint

    checkpoint_path = args.checkpoint_folder
    
    transformer = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        input_vocab_size=len(inp_tokenizer.word_counts) + 1,
        target_vocab_size=len(targ_tokenizer.word_counts) + 1,
        pe_input=1000,
        pe_target=1000,
        rate=args.dropout_rate
    )
    
    trainer = Train(transformer, optimizer, inp_tokenizer, targ_tokenizer, args.epochs, checkpoint_path)
    trainer.fit(dataset, val_dataset)

    
    