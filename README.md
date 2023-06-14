# Translation Viet-Eng

## The requirements

Make sure you have install library:

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

After training, you can test the model. My model has been trained with 12M parameters in the dataset.: https://huggingface.co/datasets/mt_eng_vietnamese, the accuracy 50% (detail in translationVi_En.ipynb). Some example:

```bash
Input:         : Bắt chước những gì bạn nhìn thấy .
Prediction     : <start> so what do you see <end> 
Truth          : You can mimic what you can see .
```

```bash
Input:         : Hôm nay tôi đi học
Prediction     : <start> today i go to school <end> 
Truth          : Today, I go to school
```

```bash
Input:         : Mô hình của chúng tôi gồm hàng trăm ngàn thùng xếp chồng tính toán với hàng trăm biến số trong thời gian cực ngắn
Prediction     : <start> our model including hundreds of thousands of boxes calculate hundreds of hundreds of times in time <end> 
Truth          : Our models have hundreds of thousands of grid boxes calculating hundreds of variables each , on minute timescales .
```


                    
