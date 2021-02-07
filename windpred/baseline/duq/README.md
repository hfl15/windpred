# Application DUQ

Official website: `https://github.com/BruceBinBoxing/Deep_Learning_Weather_Forecasting`

Our implementation: 

```
├── README.md
├── __init__.py
├── config.py <- set our parameters according to `/blob/master/src/models/parameter_config_class.py`.
├── helper.py <- copy from `/blob/master/src/data/helper.py`.
├── main.py <- the implementation for our case. 
├── main_test_on_primitive_data.py <- sanity check.
├── requirements.txt <- the required libraries.
├── seq2seq_class.py <- copy from `/blob/master/src/models/seq2seq_class.py`.
└── weather_model.py <- copy from `/blob/master/src/models/weather_model.py`.
```

It should be noted that we properly reformatted the source code for readability and usability. We have confirmed that the resulted implementation did not degrade the performance of DUQ on its own dataset.

## Some problems to set up 

### We recommend keras=2.2.4 and tensorflow=1.7.0

The primitive paper used keras=2.2.4 and tensorflow=1.8.0. With such a setting, we meet a segmentation fault when calling 'clear_session()'. We follow a solution below (`https://github.com/keras-team/keras/issues/10399`), the quizzer said: 
> Not calling clear_session() prevents the segfault. This only happens with keras 2.2 and tensorflow 1.8 (both CPU and GPU version). All other combinations of keras (2.1.x) and tensorflow (<1.8 and also 1.9.0rc0) don't result in a segfault.

Motivated by the above recommendation, we tried:

- [fail]  keras < 2.2 and tensorflow=1.8.0
- [success] keras=2.2.4 and tensorflow=1.7.0