# quantize simple
简易量化框架搭建，包括PTQ和QAT

# Prerequisites
Python 3.6+
PyTorch >= 1.6

# Training Fp32 Model
```
python main.py --type fp32 
```

# PTQ
```
python main.py --type PTQ 
```

# Training QAT Model
```
python main.py --type QAT 
```

# Resume
```
python main.py --resume
```

# Accuracy
| Model                                                | Acc.(fp32) | Acc.(PTQ) | Acc.(QAT) |
| ---------------------------------------------------- | ------ | ------ | ------ |
| VGG_s                                              | 98.8% | 97.8% | 98.7% |