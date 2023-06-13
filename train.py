import tensorflow as tf
import time
from loss import loss_function, accuracy_function
from padding import *
import nltk
from nltk.translate.bleu_score import SmoothingFunction

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

class Train:
    def __init__(self, model, optimizer, inp_tokenizer, targ_tokenizer, epochs, checkpoint_folder):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.inp_tokenizer = inp_tokenizer
        self.targ_tokenizer = targ_tokenizer
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')
        self.checkpoint = tf.train.Checkpoint(model = self.model, optimizer = self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_folder, max_to_keep=5)


    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self.model([inp, tar_inp],
                                        training = True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(tar_real, predictions))
    
    def calculate_bleu_scores(self, validation_dataset):
        smoother = SmoothingFunction().method1
        bleu_scores = []
        for (inp, tar) in validation_dataset:
            predictions, _ = self.model([inp, tar[:, :-1]], training=False)
            predictions = tf.argmax(predictions, axis=-1).numpy()
            for i in range(len(predictions)):
                pred_sentence = [self.targ_tokenizer.index_word[idx] for idx in predictions[i] if idx != 0]
                targ_sentence = [self.targ_tokenizer.index_word[idx] for idx in tar.numpy()[i, 1:] if idx != 0]
                bleu_score = nltk.translate.bleu_score.sentence_bleu([targ_sentence], pred_sentence, smoothing_function=smoother)
                bleu_scores.append(bleu_score)
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        return avg_bleu_score

    def fit(self, dataset, validation):
        print('=============Training Progress================')
        print('----------------Begin--------------------')
        # Loading checkpoint
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Restored checkpoint manager !')

        for epoch in range(self.epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            
            for (batch, (inp, tar)) in enumerate(dataset):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')

            if (epoch + 1) >= 1:
                ckpt_save_path = self.checkpoint_manager.save()
                print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

            loss_val, acc_val, bleu = self.validiate(validation)
            
            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} - Accuracy {self.train_accuracy.result():.4f} \nValidation Loss {loss_val :.4f} - Validation Accuracy {acc_val :.4f} - BLEU {bleu :.4f}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
