# Translation Viet-Eng

## The requirements

Make sure you have installed the required library:

```bash
tensorflow==2.10.0
tensorflow-gpu==2.10.0
```
## Dataset
Dataset with 4 files:
- train.vi
- train.en
- validation.vi
- validation.en

For example: 

| train.vi   |   train.en      |
|----------|:-------------:|
| Tôi là ai?      |  Who am I?|
| ...              |    .... |

## Command

Run training:

```bash
python main.py
```
There are some arguments for the script you should consider when running it:

- `input-path`: The path of the input text file (E.g. ./data/train/train.vi)
- `target-path`: The path of the output text file (E.g. ./data/train/train.en)
- `validation-input-path`: The path of the validation input text file (E.g. ./data/validation/validation.vi)
- `validation-target-path`: The path of the validation output text file (E.g. ./data/validation/validation.en)
- `checkpoint-folder`: Saved model path
- `epochs` : epochs
- `batch-size`: The batch size of the dataset
- `max-length`: The maximum length of a sentence you want to keep when preprocessing
- `num-examples`: The number of lines you want to train. It was set small if you want to experiment with this library quickly.
- `d-model`: The dimension of linear projection for all sentence.
- `num-layers`: The number of Encoder/Decoder Layers. 
- `num-heads`: The number of Multi-Head Attention. 
- `dff`: The hidden size of Position-wise Feed-Forward Networks.
- `dropout-rate`. Dropout rate of any Layer. 

After training, you can test the model. My model has been trained with 11M parameters in the dataset.: https://huggingface.co/datasets/mt_eng_vietnamese, the accuracy 56%, BLEU 0.18 (detail in translationVi_En.ipynb). Some example:

```bash
Input:         : Hôm qua , tôi đi học
Prediction     : <start> yesterday , i went to school . <end> 
Truth          : Yesterday, I went to school
```
```bash
Input:         : Chính vì lượng khí thải rất lớn , nó có ý nghĩa quan trọng với hệ thống khí quyển .
Prediction     : <start> it is because of the emissions is great , it has a meaning to the atmosphere . <end> 
Truth          : Because of the huge emissions, it is important for the atmosphere.
```
```bash
Input:         : Để tôi nói bạn biết một bí mật .
Prediction     : <start> let me tell you a secret . <end> 
Truth          : Let me tell you a secret .
```
```bash
Input:         : Môn toán nằm trong các môn khoa học .
Prediction     : <start> the math is in science . <end> 
Truth          : Mathematics is in science subjects .
```
```bash
Input:         : Ngày mai là ngày cuối cùng của kỳ thi , tôi cần học thật chăm chỉ để đạt điểm cao .
Prediction     : <start> tomorrow is day the final day of exam , i need to learn to be hard to achieve the height . <end> 
Truth          : Tomorrow is the last day of the exam, I need to study hard to achieve a high score.
```

                    
