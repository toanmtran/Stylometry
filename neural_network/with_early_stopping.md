# MLP Classifier Evaluation Results (Subsets & Full Features)

## EVALUATING TOP 15 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 0.7577
- **Mean Testing Accuracy:** 0.7419

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.844004 | 0.879803 |   0.856495 |      28.2 |
| Johnswentworth    |    0.73978  | 0.945287 |   0.826488 |      29.4 |
| Raemon            |    0.782858 | 0.691667 |   0.725108 |      24   |
| Scottalexander    |    0.643452 | 0.232512 |   0.307    |      28.2 |
| Zvi               |    0.709463 | 0.948942 |   0.808583 |      27.4 |
| macro avg         |    0.743911 | 0.739642 |   0.704735 |     137.2 |
| weighted avg      |    0.742821 | 0.741934 |   0.704762 |     137.2 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 0.8200
- **Mean Testing Accuracy:** 0.8003

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.899932 | 0.865764 |   0.879751 |      28.2 |
| Johnswentworth    |    0.841602 | 0.951954 |   0.890645 |      29.4 |
| Raemon            |    0.719633 | 0.808333 |   0.755961 |      24   |
| Scottalexander    |    0.741457 | 0.439655 |   0.534176 |      28.2 |
| Zvi               |    0.810387 | 0.934127 |   0.865274 |      27.4 |
| macro avg         |    0.802602 | 0.799967 |   0.785162 |     137.2 |
| weighted avg      |    0.805508 | 0.800328 |   0.786605 |     137.2 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 0.8939
- **Mean Testing Accuracy:** 0.8542

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.947847 | 0.886946 |   0.914642 |      28.2 |
| Johnswentworth    |    0.932397 | 0.925057 |   0.928457 |      29.4 |
| Raemon            |    0.843761 | 0.791667 |   0.810257 |      24   |
| Scottalexander    |    0.713192 | 0.736453 |   0.718426 |      28.2 |
| Zvi               |    0.871258 | 0.919312 |   0.891095 |      27.4 |
| macro avg         |    0.861691 | 0.851887 |   0.852575 |     137.2 |
| weighted avg      |    0.862755 | 0.854237 |   0.854381 |     137.2 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 0.8732
- **Mean Testing Accuracy:** 0.8281

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.933802 | 0.886946 |   0.908122 |      28.2 |
| Johnswentworth    |    0.897161 | 0.918391 |   0.903105 |      29.4 |
| Raemon            |    0.82266  | 0.775    |   0.78916  |      24   |
| Scottalexander    |    0.663566 | 0.632759 |   0.646393 |      28.2 |
| Zvi               |    0.841225 | 0.919312 |   0.877756 |      27.4 |
| macro avg         |    0.831683 | 0.826481 |   0.824907 |     137.2 |
| weighted avg      |    0.832314 | 0.82815  |   0.826229 |     137.2 |

### 5-Fold CV: Depth 50

- **Mean Training Accuracy:** 0.2121
- **Mean Testing Accuracy:** 0.2143

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |   0.042029  | 0.2      |  0.0694611 |      28.2 |
| Johnswentworth    |   0.172263  | 0.8      |  0.283472  |      29.4 |
| Raemon            |   0         | 0        |  0         |      24   |
| Scottalexander    |   0         | 0        |  0         |      28.2 |
| Zvi               |   0         | 0        |  0         |      27.4 |
| macro avg         |   0.0428584 | 0.2      |  0.0705865 |     137.2 |
| weighted avg      |   0.0459359 | 0.214292 |  0.0756508 |     137.2 |

## EVALUATING TOP 30 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 0.8921
- **Mean Testing Accuracy:** 0.8703

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.931414 | 0.887192 |   0.90722  |      28.2 |
| Johnswentworth    |    0.944974 | 0.918391 |   0.931104 |      29.4 |
| Raemon            |    0.870074 | 0.791667 |   0.824153 |      24   |
| Scottalexander    |    0.72703  | 0.780296 |   0.750199 |      28.2 |
| Zvi               |    0.907507 | 0.963757 |   0.933474 |      27.4 |
| macro avg         |    0.876199 | 0.86826  |   0.86923  |     137.2 |
| weighted avg      |    0.876597 | 0.870306 |   0.870573 |     137.2 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 0.9165
- **Mean Testing Accuracy:** 0.8922

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.963038 | 0.915271 |   0.937436 |      28.2 |
| Johnswentworth    |    0.942954 | 0.952184 |   0.945992 |      29.4 |
| Raemon            |    0.852747 | 0.816667 |   0.832215 |      24   |
| Scottalexander    |    0.793128 | 0.787192 |   0.789245 |      28.2 |
| Zvi               |    0.913644 | 0.978042 |   0.943904 |      27.4 |
| macro avg         |    0.893102 | 0.889871 |   0.889758 |     137.2 |
| weighted avg      |    0.894673 | 0.892151 |   0.891715 |     137.2 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 0.9063
- **Mean Testing Accuracy:** 0.8791

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.950128 | 0.936453 |   0.942648 |      28.2 |
| Johnswentworth    |    0.916703 | 0.931724 |   0.923016 |      29.4 |
| Raemon            |    0.861341 | 0.766667 |   0.806725 |      24   |
| Scottalexander    |    0.752617 | 0.76601  |   0.75537  |      28.2 |
| Zvi               |    0.928915 | 0.978042 |   0.951254 |      27.4 |
| macro avg         |    0.881941 | 0.875779 |   0.875803 |     137.2 |
| weighted avg      |    0.882398 | 0.879075 |   0.877819 |     137.2 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 0.9821
- **Mean Testing Accuracy:** 0.8878

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.947903 | 0.915025 |   0.930184 |      28.2 |
| Johnswentworth    |    0.952993 | 0.938621 |   0.945258 |      29.4 |
| Raemon            |    0.825315 | 0.841667 |   0.827584 |      24   |
| Scottalexander    |    0.769691 | 0.779803 |   0.771338 |      28.2 |
| Zvi               |    0.966786 | 0.956349 |   0.960544 |      27.4 |
| macro avg         |    0.892538 | 0.886293 |   0.886982 |     137.2 |
| weighted avg      |    0.894647 | 0.887771 |   0.888883 |     137.2 |

### 5-Fold CV: Depth 50

- **Mean Training Accuracy:** 0.1830
- **Mean Testing Accuracy:** 0.1822

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |   0         | 0        |  0         |      28.2 |
| Johnswentworth    |   0.042029  | 0.2      |  0.0694611 |      29.4 |
| Raemon            |   0.140146  | 0.8      |  0.238509  |      24   |
| Scottalexander    |   0         | 0        |  0         |      28.2 |
| Zvi               |   0         | 0        |  0         |      27.4 |
| macro avg         |   0.036435  | 0.2      |  0.0615941 |     137.2 |
| weighted avg      |   0.0333833 | 0.182175 |  0.0563795 |     137.2 |

## EVALUATING TOP 50 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 0.9100
- **Mean Testing Accuracy:** 0.8645

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.941101 | 0.901478 |   0.919552 |      28.2 |
| Johnswentworth    |    0.930036 | 0.931494 |   0.928808 |      29.4 |
| Raemon            |    0.794217 | 0.8      |   0.792059 |      24   |
| Scottalexander    |    0.75382  | 0.716256 |   0.73235  |      28.2 |
| Zvi               |    0.912438 | 0.963757 |   0.936661 |      27.4 |
| macro avg         |    0.866322 | 0.862597 |   0.861886 |     137.2 |
| weighted avg      |    0.868705 | 0.864498 |   0.864102 |     137.2 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 0.9497
- **Mean Testing Accuracy:** 0.8950

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.940127 | 0.929064 |   0.932462 |      28.2 |
| Johnswentworth    |    0.940714 | 0.945517 |   0.94233  |      29.4 |
| Raemon            |    0.885983 | 0.808333 |   0.840065 |      24   |
| Scottalexander    |    0.812857 | 0.793842 |   0.797917 |      28.2 |
| Zvi               |    0.920952 | 0.98545  |   0.94973  |      27.4 |
| macro avg         |    0.900127 | 0.892441 |   0.892501 |     137.2 |
| weighted avg      |    0.900844 | 0.895049 |   0.894298 |     137.2 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 0.9748
- **Mean Testing Accuracy:** 0.9112

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.965333 | 0.936946 |   0.949028 |      28.2 |
| Johnswentworth    |    0.949597 | 0.945747 |   0.946601 |      29.4 |
| Raemon            |    0.836794 | 0.85     |   0.842007 |      24   |
| Scottalexander    |    0.853013 | 0.844335 |   0.8467   |      28.2 |
| Zvi               |    0.959605 | 0.970635 |   0.963967 |      27.4 |
| macro avg         |    0.912868 | 0.909533 |   0.909661 |     137.2 |
| weighted avg      |    0.915271 | 0.911182 |   0.911684 |     137.2 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 0.9628
- **Mean Testing Accuracy:** 0.8951

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.978799 | 0.94335  |   0.960483 |      28.2 |
| Johnswentworth    |    0.931361 | 0.938621 |   0.93354  |      29.4 |
| Raemon            |    0.831088 | 0.808333 |   0.817939 |      24   |
| Scottalexander    |    0.782597 | 0.808374 |   0.791973 |      28.2 |
| Zvi               |    0.957882 | 0.963228 |   0.95997  |      27.4 |
| macro avg         |    0.896345 | 0.892381 |   0.892781 |     137.2 |
| weighted avg      |    0.898288 | 0.895081 |   0.89511  |     137.2 |

### 5-Fold CV: Depth 50

- **Mean Training Accuracy:** 0.2143
- **Mean Testing Accuracy:** 0.2143

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |   0         | 0        |  0         |      28.2 |
| Johnswentworth    |   0.214292  | 1        |  0.352933  |      29.4 |
| Raemon            |   0         | 0        |  0         |      24   |
| Scottalexander    |   0         | 0        |  0         |      28.2 |
| Zvi               |   0         | 0        |  0         |      27.4 |
| macro avg         |   0.0428584 | 0.2      |  0.0705865 |     137.2 |
| weighted avg      |   0.0459359 | 0.214292 |  0.0756508 |     137.2 |

## EVALUATING TOP 74 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 0.9501
- **Mean Testing Accuracy:** 0.9054

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.955921 | 0.930049 |   0.941922 |      28.2 |
| Johnswentworth    |    0.91711  | 0.931494 |   0.92308  |      29.4 |
| Raemon            |    0.888022 | 0.85     |   0.86461  |      24   |
| Scottalexander    |    0.832056 | 0.81601  |   0.82199  |      28.2 |
| Zvi               |    0.947067 | 0.992857 |   0.968531 |      27.4 |
| macro avg         |    0.908035 | 0.904082 |   0.904027 |     137.2 |
| weighted avg      |    0.908548 | 0.905374 |   0.90501  |     137.2 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 0.9657
- **Mean Testing Accuracy:** 0.9301

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.952593 | 0.922906 |   0.934791 |      28.2 |
| Johnswentworth    |    0.960845 | 0.931724 |   0.944959 |      29.4 |
| Raemon            |    0.919007 | 0.916667 |   0.913032 |      24   |
| Scottalexander    |    0.879957 | 0.89335  |   0.883601 |      28.2 |
| Zvi               |    0.965025 | 0.985714 |   0.974668 |      27.4 |
| macro avg         |    0.935485 | 0.930072 |   0.93021  |     137.2 |
| weighted avg      |    0.936054 | 0.930096 |   0.930559 |     137.2 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 0.9858
- **Mean Testing Accuracy:** 0.9257

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.952078 | 0.9367   |   0.942235 |      28.2 |
| Johnswentworth    |    0.945006 | 0.938391 |   0.941405 |      29.4 |
| Raemon            |    0.894309 | 0.85     |   0.863276 |      24   |
| Scottalexander    |    0.878578 | 0.907635 |   0.888302 |      28.2 |
| Zvi               |    0.985961 | 0.985714 |   0.985447 |      27.4 |
| macro avg         |    0.931186 | 0.923688 |   0.924133 |     137.2 |
| weighted avg      |    0.931958 | 0.925748 |   0.925716 |     137.2 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 0.9610
- **Mean Testing Accuracy:** 0.8980

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.965517 | 0.929803 |   0.94525  |      28.2 |
| Johnswentworth    |    0.955649 | 0.877471 |   0.913802 |      29.4 |
| Raemon            |    0.895586 | 0.858333 |   0.876395 |      24   |
| Scottalexander    |    0.762002 | 0.851724 |   0.800837 |      28.2 |
| Zvi               |    0.952074 | 0.970899 |   0.960605 |      27.4 |
| macro avg         |    0.906166 | 0.897646 |   0.899378 |     137.2 |
| weighted avg      |    0.906646 | 0.898032 |   0.899729 |     137.2 |

### 5-Fold CV: Depth 50

- **Mean Training Accuracy:** 0.2143
- **Mean Testing Accuracy:** 0.2143

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |   0         | 0        |  0         |      28.2 |
| Johnswentworth    |   0.214292  | 1        |  0.352933  |      29.4 |
| Raemon            |   0         | 0        |  0         |      24   |
| Scottalexander    |   0         | 0        |  0         |      28.2 |
| Zvi               |   0         | 0        |  0         |      27.4 |
| macro avg         |   0.0428584 | 0.2      |  0.0705865 |     137.2 |
| weighted avg      |   0.0459359 | 0.214292 |  0.0756508 |     137.2 |

## EVALUATING ALL 107 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 0.9409
- **Mean Testing Accuracy:** 0.8922

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.948783 | 0.901232 |   0.923583 |      28.2 |
| Johnswentworth    |    0.896131 | 0.910805 |   0.902534 |      29.4 |
| Raemon            |    0.837587 | 0.866667 |   0.847827 |      24   |
| Scottalexander    |    0.839898 | 0.794581 |   0.815512 |      28.2 |
| Zvi               |    0.939532 | 0.985185 |   0.96134  |      27.4 |
| macro avg         |    0.892386 | 0.891694 |   0.890159 |     137.2 |
| weighted avg      |    0.893866 | 0.892246 |   0.891248 |     137.2 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 0.9676
- **Mean Testing Accuracy:** 0.9243

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.971675 | 0.92931  |   0.949012 |      28.2 |
| Johnswentworth    |    0.939244 | 0.938391 |   0.938368 |      29.4 |
| Raemon            |    0.866537 | 0.891667 |   0.873067 |      24   |
| Scottalexander    |    0.893354 | 0.872167 |   0.879717 |      28.2 |
| Zvi               |    0.967051 | 0.98545  |   0.97529  |      27.4 |
| macro avg         |    0.927572 | 0.923397 |   0.923091 |     137.2 |
| weighted avg      |    0.929219 | 0.924267 |   0.924471 |     137.2 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 0.9854
- **Mean Testing Accuracy:** 0.9243

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.957006 | 0.929803 |   0.94218  |      28.2 |
| Johnswentworth    |    0.947251 | 0.945287 |   0.94585  |      29.4 |
| Raemon            |    0.913578 | 0.883333 |   0.897638 |      24   |
| Scottalexander    |    0.844299 | 0.865271 |   0.854002 |      28.2 |
| Zvi               |    0.965006 | 0.992593 |   0.978432 |      27.4 |
| macro avg         |    0.925428 | 0.923257 |   0.92362  |     137.2 |
| weighted avg      |    0.92576  | 0.924278 |   0.924294 |     137.2 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 0.9916
- **Mean Testing Accuracy:** 0.8995

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.928296 | 0.922414 |   0.921897 |      28.2 |
| Johnswentworth    |    0.930781 | 0.910805 |   0.917826 |      29.4 |
| Raemon            |    0.900156 | 0.883333 |   0.888806 |      24   |
| Scottalexander    |    0.779035 | 0.800985 |   0.785768 |      28.2 |
| Zvi               |    0.992593 | 0.977778 |   0.985045 |      27.4 |
| macro avg         |    0.906172 | 0.899063 |   0.899868 |     137.2 |
| weighted avg      |    0.90589  | 0.89946  |   0.899904 |     137.2 |

### 5-Fold CV: Depth 50

- **Mean Training Accuracy:** 0.2114
- **Mean Testing Accuracy:** 0.2085

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |   0         | 0        |  0         |      28.2 |
| Johnswentworth    |   0.126701  | 0.6      |  0.20922   |      29.4 |
| Raemon            |   0         | 0        |  0         |      24   |
| Scottalexander    |   0.0817518 | 0.4      |  0.135758  |      28.2 |
| Zvi               |   0         | 0        |  0         |      27.4 |
| macro avg         |   0.0416905 | 0.2      |  0.0689955 |     137.2 |
| weighted avg      |   0.0434638 | 0.208452 |  0.071927  |     137.2 |

