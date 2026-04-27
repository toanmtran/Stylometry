# MLP Classifier Evaluation Results (Subsets & Full Features)

## EVALUATING TOP 15 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 0.9990
- **Mean Testing Accuracy:** 0.8506

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.906754 | 0.88     |   0.891457 |      30   |
| Johnswentworth    |    0.89204  | 0.906667 |   0.897502 |      30   |
| Raemon            |    0.847213 | 0.826154 |   0.831982 |      25.2 |
| Scottalexander    |    0.720863 | 0.734943 |   0.725469 |      29.4 |
| Zvi               |    0.908401 | 0.9      |   0.903205 |      30   |
| macro avg         |    0.855054 | 0.849553 |   0.849923 |     144.6 |
| weighted avg      |    0.856017 | 0.850632 |   0.85102  |     144.6 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.8589

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.922814 | 0.906667 |   0.91319  |      30   |
| Johnswentworth    |    0.903752 | 0.913333 |   0.906219 |      30   |
| Raemon            |    0.843179 | 0.857231 |   0.848455 |      25.2 |
| Scottalexander    |    0.724862 | 0.714023 |   0.718933 |      29.4 |
| Zvi               |    0.908273 | 0.9      |   0.90305  |      30   |
| macro avg         |    0.860576 | 0.858251 |   0.857969 |     144.6 |
| weighted avg      |    0.861913 | 0.858918 |   0.858977 |     144.6 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.8493

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.913972 | 0.893333 |   0.902253 |      30   |
| Johnswentworth    |    0.913137 | 0.92     |   0.913014 |      30   |
| Raemon            |    0.803841 | 0.833538 |   0.817213 |      25.2 |
| Scottalexander    |    0.727418 | 0.707356 |   0.71676  |      29.4 |
| Zvi               |    0.890303 | 0.886667 |   0.887569 |      30   |
| macro avg         |    0.849734 | 0.848179 |   0.847362 |     144.6 |
| weighted avg      |    0.851932 | 0.849262 |   0.848992 |     144.6 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.8258

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.885126 | 0.873333 |   0.878314 |      30   |
| Johnswentworth    |    0.911246 | 0.9      |   0.903375 |      30   |
| Raemon            |    0.791333 | 0.802154 |   0.793098 |      25.2 |
| Scottalexander    |    0.681178 | 0.66046  |   0.668    |      29.4 |
| Zvi               |    0.864424 | 0.886667 |   0.874909 |      30   |
| macro avg         |    0.826661 | 0.824523 |   0.823539 |     144.6 |
| weighted avg      |    0.828666 | 0.825757 |   0.825205 |     144.6 |

### 5-Fold CV: Depth 50

- **Mean Training Accuracy:** 0.2075
- **Mean Testing Accuracy:** 0.2075

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |   0         | 0        |  0         |      30   |
| Johnswentworth    |   0         | 0        |  0         |      30   |
| Raemon            |   0         | 0        |  0         |      25.2 |
| Scottalexander    |   0         | 0        |  0         |      29.4 |
| Zvi               |   0.207471  | 1        |  0.343645  |      30   |
| macro avg         |   0.0414943 | 0.2      |  0.0687291 |     144.6 |
| weighted avg      |   0.0430448 | 0.207471 |  0.0712972 |     144.6 |

## EVALUATING TOP 30 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9199

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.946437 | 0.94     |   0.943164 |      30   |
| Johnswentworth    |    0.946141 | 0.953333 |   0.947432 |      30   |
| Raemon            |    0.898093 | 0.881538 |   0.885191 |      25.2 |
| Scottalexander    |    0.839065 | 0.850575 |   0.843578 |      29.4 |
| Zvi               |    0.979985 | 0.966667 |   0.972989 |      30   |
| macro avg         |    0.921944 | 0.918423 |   0.918471 |     144.6 |
| weighted avg      |    0.923064 | 0.919885 |   0.919861 |     144.6 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9295

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.93977  | 0.933333 |   0.936497 |      30   |
| Johnswentworth    |    0.964286 | 0.966667 |   0.963926 |      30   |
| Raemon            |    0.885397 | 0.913538 |   0.895302 |      25.2 |
| Scottalexander    |    0.877488 | 0.871034 |   0.872732 |      29.4 |
| Zvi               |    0.986667 | 0.96     |   0.972874 |      30   |
| macro avg         |    0.930722 | 0.928915 |   0.928266 |     144.6 |
| weighted avg      |    0.932469 | 0.92954  |   0.929546 |     144.6 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9198

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.939494 | 0.92     |   0.929365 |      30   |
| Johnswentworth    |    0.946141 | 0.953333 |   0.947432 |      30   |
| Raemon            |    0.899942 | 0.889846 |   0.888883 |      25.2 |
| Scottalexander    |    0.858593 | 0.857011 |   0.856851 |      29.4 |
| Zvi               |    0.967082 | 0.973333 |   0.970049 |      30   |
| macro avg         |    0.922251 | 0.918705 |   0.918516 |     144.6 |
| weighted avg      |    0.923376 | 0.919847 |   0.919778 |     144.6 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.8950

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.934363 | 0.92     |   0.926633 |      30   |
| Johnswentworth    |    0.959511 | 0.926667 |   0.942473 |      30   |
| Raemon            |    0.870804 | 0.857538 |   0.863035 |      25.2 |
| Scottalexander    |    0.790014 | 0.809425 |   0.797486 |      29.4 |
| Zvi               |    0.930244 | 0.953333 |   0.940459 |      30   |
| macro avg         |    0.896987 | 0.893393 |   0.894017 |     144.6 |
| weighted avg      |    0.898161 | 0.894962 |   0.895388 |     144.6 |

### 5-Fold CV: Depth 50

- **Mean Training Accuracy:** 0.2075
- **Mean Testing Accuracy:** 0.2075

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |   0         | 0        |  0         |      30   |
| Johnswentworth    |   0.124138  | 0.6      |  0.205714  |      30   |
| Raemon            |   0         | 0        |  0         |      25.2 |
| Scottalexander    |   0         | 0        |  0         |      29.4 |
| Zvi               |   0.0833333 | 0.4      |  0.137931  |      30   |
| macro avg         |   0.0414943 | 0.2      |  0.0687291 |     144.6 |
| weighted avg      |   0.0430448 | 0.207471 |  0.0712972 |     144.6 |

## EVALUATING TOP 50 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9461

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.973319 | 0.966667 |   0.969829 |      30   |
| Johnswentworth    |    0.951392 | 0.98     |   0.964417 |      30   |
| Raemon            |    0.900479 | 0.928923 |   0.914154 |      25.2 |
| Scottalexander    |    0.910592 | 0.877931 |   0.892951 |      29.4 |
| Zvi               |    0.993333 | 0.973333 |   0.983047 |      30   |
| macro avg         |    0.945823 | 0.945371 |   0.944879 |     144.6 |
| weighted avg      |    0.947476 | 0.946121 |   0.946079 |     144.6 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9489

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.973548 | 0.98     |   0.976721 |      30   |
| Johnswentworth    |    0.951192 | 0.973333 |   0.961239 |      30   |
| Raemon            |    0.92454  | 0.913231 |   0.916356 |      25.2 |
| Scottalexander    |    0.913621 | 0.897701 |   0.904451 |      29.4 |
| Zvi               |    0.986882 | 0.973333 |   0.979768 |      30   |
| macro avg         |    0.949956 | 0.94752  |   0.947707 |     144.6 |
| weighted avg      |    0.950912 | 0.94887  |   0.948914 |     144.6 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9350

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.966422 | 0.96     |   0.963049 |      30   |
| Johnswentworth    |    0.957197 | 0.946667 |   0.950261 |      30   |
| Raemon            |    0.882528 | 0.897231 |   0.886854 |      25.2 |
| Scottalexander    |    0.87977  | 0.884598 |   0.881537 |      29.4 |
| Zvi               |    0.993548 | 0.98     |   0.986435 |      30   |
| macro avg         |    0.935893 | 0.933699 |   0.933627 |     144.6 |
| weighted avg      |    0.938035 | 0.935038 |   0.935431 |     144.6 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9157

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.973333 | 0.933333 |   0.952281 |      30   |
| Johnswentworth    |    0.947297 | 0.933333 |   0.939983 |      30   |
| Raemon            |    0.884925 | 0.905231 |   0.891855 |      25.2 |
| Scottalexander    |    0.810319 | 0.857471 |   0.829992 |      29.4 |
| Zvi               |    0.979985 | 0.946667 |   0.962462 |      30   |
| macro avg         |    0.919172 | 0.915207 |   0.915315 |     144.6 |
| weighted avg      |    0.920944 | 0.915661 |   0.916487 |     144.6 |

### 5-Fold CV: Depth 50

- **Mean Training Accuracy:** 0.2075
- **Mean Testing Accuracy:** 0.2075

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |   0.0833333 | 0.4      |  0.137931  |      30   |
| Johnswentworth    |   0         | 0        |  0         |      30   |
| Raemon            |   0         | 0        |  0         |      25.2 |
| Scottalexander    |   0         | 0        |  0         |      29.4 |
| Zvi               |   0.124138  | 0.6      |  0.205714  |      30   |
| macro avg         |   0.0414943 | 0.2      |  0.0687291 |     144.6 |
| weighted avg      |   0.0430448 | 0.207471 |  0.0712972 |     144.6 |

## EVALUATING TOP 74 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9516

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.987097 | 0.966667 |   0.976488 |      30   |
| Johnswentworth    |    0.949291 | 0.966667 |   0.957455 |      30   |
| Raemon            |    0.953056 | 0.913231 |   0.930649 |      25.2 |
| Scottalexander    |    0.892742 | 0.931494 |   0.910524 |      29.4 |
| Zvi               |    0.986652 | 0.973333 |   0.979655 |      30   |
| macro avg         |    0.953768 | 0.950278 |   0.950954 |     144.6 |
| weighted avg      |    0.954018 | 0.951638 |   0.951805 |     144.6 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9461

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.97977  | 0.96     |   0.969601 |      30   |
| Johnswentworth    |    0.961772 | 0.973333 |   0.966736 |      30   |
| Raemon            |    0.923417 | 0.897846 |   0.905937 |      25.2 |
| Scottalexander    |    0.88372  | 0.917931 |   0.896871 |      29.4 |
| Zvi               |    1        | 0.973333 |   0.986207 |      30   |
| macro avg         |    0.949736 | 0.944489 |   0.94507  |     144.6 |
| weighted avg      |    0.950741 | 0.94613  |   0.946486 |     144.6 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9254

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.942095 | 0.946667 |   0.943801 |      30   |
| Johnswentworth    |    0.940774 | 0.933333 |   0.936581 |      30   |
| Raemon            |    0.876573 | 0.881538 |   0.877668 |      25.2 |
| Scottalexander    |    0.887774 | 0.884138 |   0.885268 |      29.4 |
| Zvi               |    0.97977  | 0.973333 |   0.976497 |      30   |
| macro avg         |    0.925397 | 0.923802 |   0.923963 |     144.6 |
| weighted avg      |    0.927138 | 0.925364 |   0.92564  |     144.6 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9087

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.973103 | 0.94     |   0.955545 |      30   |
| Johnswentworth    |    0.953333 | 0.913333 |   0.932151 |      30   |
| Raemon            |    0.873863 | 0.889846 |   0.878392 |      25.2 |
| Scottalexander    |    0.797291 | 0.836782 |   0.814624 |      29.4 |
| Zvi               |    0.959954 | 0.96     |   0.95965  |      30   |
| macro avg         |    0.911509 | 0.907992 |   0.908072 |     144.6 |
| weighted avg      |    0.913226 | 0.908736 |   0.909364 |     144.6 |

### 5-Fold CV: Depth 50

- **Mean Training Accuracy:** 0.2075
- **Mean Testing Accuracy:** 0.2075

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |   0         | 0        |  0         |      30   |
| Johnswentworth    |   0         | 0        |  0         |      30   |
| Raemon            |   0         | 0        |  0         |      25.2 |
| Scottalexander    |   0         | 0        |  0         |      29.4 |
| Zvi               |   0.207471  | 1        |  0.343645  |      30   |
| macro avg         |   0.0414943 | 0.2      |  0.0687291 |     144.6 |
| weighted avg      |   0.0430448 | 0.207471 |  0.0712972 |     144.6 |

## EVALUATING ALL 107 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9503

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.973057 | 0.946667 |   0.959305 |      30   |
| Johnswentworth    |    0.980833 | 0.966667 |   0.973206 |      30   |
| Raemon            |    0.922005 | 0.929231 |   0.924811 |      25.2 |
| Scottalexander    |    0.892874 | 0.932184 |   0.911293 |      29.4 |
| Zvi               |    0.986652 | 0.973333 |   0.979655 |      30   |
| macro avg         |    0.951084 | 0.949616 |   0.949654 |     144.6 |
| weighted avg      |    0.952282 | 0.950268 |   0.950587 |     144.6 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9517

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.979954 | 0.96     |   0.969474 |      30   |
| Johnswentworth    |    0.966407 | 0.946667 |   0.956266 |      30   |
| Raemon            |    0.920826 | 0.929231 |   0.924527 |      25.2 |
| Scottalexander    |    0.911949 | 0.945977 |   0.927919 |      29.4 |
| Zvi               |    0.980603 | 0.973333 |   0.976482 |      30   |
| macro avg         |    0.951948 | 0.951042 |   0.950934 |     144.6 |
| weighted avg      |    0.953077 | 0.951686 |   0.951823 |     144.6 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9364

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.954307 | 0.946667 |   0.949908 |      30   |
| Johnswentworth    |    0.947227 | 0.933333 |   0.939748 |      30   |
| Raemon            |    0.935814 | 0.905231 |   0.919308 |      25.2 |
| Scottalexander    |    0.86433  | 0.904368 |   0.883364 |      29.4 |
| Zvi               |    0.987097 | 0.986667 |   0.986546 |      30   |
| macro avg         |    0.937755 | 0.935253 |   0.935775 |     144.6 |
| weighted avg      |    0.938185 | 0.936379 |   0.936568 |     144.6 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9129

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.960509 | 0.92     |   0.938797 |      30   |
| Johnswentworth    |    0.927642 | 0.906667 |   0.915705 |      30   |
| Raemon            |    0.921824 | 0.921231 |   0.920098 |      25.2 |
| Scottalexander    |    0.793312 | 0.856782 |   0.821422 |      29.4 |
| Zvi               |    0.986667 | 0.96     |   0.972751 |      30   |
| macro avg         |    0.917991 | 0.912936 |   0.913755 |     144.6 |
| weighted avg      |    0.918346 | 0.912921 |   0.913917 |     144.6 |

### 5-Fold CV: Depth 50

- **Mean Training Accuracy:** 0.2075
- **Mean Testing Accuracy:** 0.2075

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |   0         | 0        |  0         |      30   |
| Johnswentworth    |   0.0833333 | 0.4      |  0.137931  |      30   |
| Raemon            |   0         | 0        |  0         |      25.2 |
| Scottalexander    |   0         | 0        |  0         |      29.4 |
| Zvi               |   0.124138  | 0.6      |  0.205714  |      30   |
| macro avg         |   0.0414943 | 0.2      |  0.0687291 |     144.6 |
| weighted avg      |   0.0430448 | 0.207471 |  0.0712972 |     144.6 |

