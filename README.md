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

After training, you can test the model. My model has been trained with 11M parameters in the dataset.: https://huggingface.co/datasets/mt_eng_vietnamese, the BLEU 0.18 (detail in translationVi_En.ipynb). Some example:

```bash
Thật không thể tin được ! Mọi thứ đều trở nên tuyệt vời hơn mọi mong đợi . 
it is impossible to believe ! everything is been more wonderful than the expectations .

Đừng bao giờ nói với tôi những điều đó nữa ! Tôi đã đủ nghe rồi . 
never told me that . i heard enough .

Hãy giữ lấy giấc mơ của bạn và không bao giờ từ bỏ nó  
keep your dreams and never giving it away .

Sao không dừng lại một chút ? Chúng ta cần thời gian để thưởng thức cuộc sống . 
why dont we stop ? we need time to enjoy life .

Có chăng đó là một lời đề nghị không thể từ chối ? Tôi không thể chờ đợi để tham gia ! 
it is a question that is not from the denied ? i cant wait to be waiting for the next .

Chúng ta đã đi qua bao nhiêu khó khăn và cuối cùng , chúng ta đã thành công ! 
we have been through how much difficulty , and we have been successful !

Xin lỗi , tôi không thể đồng ý với ý kiến của bạn . Tôi có quyền tự do biểu đạt suy nghĩ của mình . 
sorry , i cant agree with your idea . i have freedom to express my thoughts .

Hãy đến đây ngay lập tức ! Tôi cần bạn ở bên tôi trong khoảnh khắc này . 
let is go right here ! i need you to stay inside this moment .

Điều này liệu có thực sự xảy ra ? Tôi không thể tin nổi mắt mình . 
is this real ? i cant believe my eyes .

Xin hãy cho tôi một lời giải thích thỏa đáng ! Tôi muốn hiểu rõ vì sao điều đó lại xảy ra . 
let me give me a great explanation ! i want to understand why that happened .
```

                    
