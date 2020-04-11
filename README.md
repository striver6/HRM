# HRM
High resolution esophageal manometry
# Architecture
![image](https://github.com/striver6/HRM/blob/master/hrm%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E5%9B%BE%20.png)

# 一、First Model Performance
- inceptionresnetv2_normal:
Training:  0.2221 0.1104 0.9161 0.0451

Validation: 0.2430 0.0524 0.9148 0.0167

- inceptionv3_normal:  
Training:  0.3620 0.0998 0.8592 0.0430

Validation: 0.3558 0.3040 0.8778 0.0731

- resnet50_normal:  
Training:  0.2950 0.0907 0.8865 0.0357

Validation: 0.2749 0.1399 0.9007 0.0455

- smallervgg_normal:  
Training:  0.2596 0.0718 0.8970 0.0308

Validation: 0.2357 0.0335 0.9123 0.0158

- xception_normal:  *****
Training:  0.2171 0.0954 0.9175 0.0388

Validation: 0.2503 0.0978 0.9142 0.0262

# 二、First Model Performance

Conv2DLSTMNet:

3*16  || Training:  0.2396 0.1546 0.9125 0.0625 || Validation: 0.3487 0.0612 0.8840 0.0243

3*8  || Training:  0.2329 0.1236 0.9138 0.0556  || Validation: 0.3670 0.0705 0.8792 0.0220

8*16*32  || Training:  0.2163 0.1374 0.9221 0.0581 || Validation: 0.4088 0.0950 0.8723 0.0203
