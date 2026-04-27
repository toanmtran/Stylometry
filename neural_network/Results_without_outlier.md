# MLP Classifier Evaluation Results (Subsets & Full Features)

## EVALUATING TOP 15 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 0.9996
- **Mean Testing Accuracy:** 0.8761

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.903777 | 0.901478 |   0.899997 |      28.2 |
| Johnswentworth    |    0.966207 | 0.932184 |   0.948326 |      29.4 |
| Raemon            |    0.824152 | 0.883333 |   0.848919 |      24   |
| Scottalexander    |    0.783361 | 0.758128 |   0.763343 |      28.2 |
| Zvi               |    0.935608 | 0.904762 |   0.917926 |      27.4 |
| macro avg         |    0.882621 | 0.875977 |   0.875702 |     137.2 |
| weighted avg      |    0.884667 | 0.876124 |   0.876836 |     137.2 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.8557

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.926076 | 0.879803 |   0.900206 |      28.2 |
| Johnswentworth    |    0.951691 | 0.931954 |   0.941537 |      29.4 |
| Raemon            |    0.774267 | 0.85     |   0.806864 |      24   |
| Scottalexander    |    0.751197 | 0.708374 |   0.723568 |      28.2 |
| Zvi               |    0.894412 | 0.904762 |   0.895883 |      27.4 |
| macro avg         |    0.859529 | 0.854979 |   0.853611 |     137.2 |
| weighted avg      |    0.862707 | 0.855686 |   0.855634 |     137.2 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.8571

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.907718 | 0.880049 |   0.890772 |      28.2 |
| Johnswentworth    |    0.944581 | 0.918621 |   0.931266 |      29.4 |
| Raemon            |    0.813312 | 0.85     |   0.825806 |      24   |
| Scottalexander    |    0.737956 | 0.722414 |   0.724832 |      28.2 |
| Zvi               |    0.903392 | 0.912169 |   0.905365 |      27.4 |
| macro avg         |    0.861392 | 0.856651 |   0.855608 |     137.2 |
| weighted avg      |    0.863325 | 0.857146 |   0.856903 |     137.2 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.8382

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.903832 | 0.8867   |   0.89158  |      28.2 |
| Johnswentworth    |    0.914217 | 0.925287 |   0.919026 |      29.4 |
| Raemon            |    0.763312 | 0.825    |   0.787334 |      24   |
| Scottalexander    |    0.726039 | 0.65936  |   0.689459 |      28.2 |
| Zvi               |    0.900398 | 0.890212 |   0.890183 |      27.4 |
| macro avg         |    0.84156  | 0.837312 |   0.835516 |     137.2 |
| weighted avg      |    0.844151 | 0.83821  |   0.837394 |     137.2 |

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

## EVALUATING TOP 30 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9228

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.95722  | 0.943596 |   0.949298 |      28.2 |
| Johnswentworth    |    0.952593 | 0.952184 |   0.952042 |      29.4 |
| Raemon            |    0.895565 | 0.875    |   0.8807   |      24   |
| Scottalexander    |    0.837526 | 0.857389 |   0.843717 |      28.2 |
| Zvi               |    0.985714 | 0.978307 |   0.981941 |      27.4 |
| macro avg         |    0.925724 | 0.921295 |   0.92154  |     137.2 |
| weighted avg      |    0.9264   | 0.922755 |   0.922685 |     137.2 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9242

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.979064 | 0.943596 |   0.959546 |      28.2 |
| Johnswentworth    |    0.960068 | 0.945287 |   0.951541 |      29.4 |
| Raemon            |    0.874883 | 0.883333 |   0.876632 |      24   |
| Scottalexander    |    0.834094 | 0.864778 |   0.847648 |      28.2 |
| Zvi               |    0.985714 | 0.978307 |   0.981941 |      27.4 |
| macro avg         |    0.926765 | 0.92306  |   0.923462 |     137.2 |
| weighted avg      |    0.928264 | 0.924193 |   0.924803 |     137.2 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9169

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.970122 | 0.936207 |   0.951371 |      28.2 |
| Johnswentworth    |    0.959048 | 0.945747 |   0.951952 |      29.4 |
| Raemon            |    0.874437 | 0.883333 |   0.872951 |      24   |
| Scottalexander    |    0.837186 | 0.828818 |   0.830727 |      28.2 |
| Zvi               |    0.952302 | 0.98545  |   0.968137 |      27.4 |
| macro avg         |    0.918619 | 0.915911 |   0.915028 |     137.2 |
| weighted avg      |    0.920093 | 0.916894 |   0.916373 |     137.2 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.8995

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.970368 | 0.936207 |   0.951246 |      28.2 |
| Johnswentworth    |    0.949351 | 0.938621 |   0.942855 |      29.4 |
| Raemon            |    0.859238 | 0.866667 |   0.859169 |      24   |
| Scottalexander    |    0.790516 | 0.794581 |   0.792033 |      28.2 |
| Zvi               |    0.938234 | 0.956349 |   0.946433 |      27.4 |
| macro avg         |    0.901541 | 0.898485 |   0.898347 |     137.2 |
| weighted avg      |    0.903132 | 0.89946  |   0.899698 |     137.2 |

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

## EVALUATING TOP 50 FEATURES

### 5-Fold CV: Depth 1 (64,)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9548

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.97977  | 0.986207 |   0.982452 |      28.2 |
| Johnswentworth    |    0.979259 | 0.965977 |   0.972268 |      29.4 |
| Raemon            |    0.912637 | 0.916667 |   0.912142 |      24   |
| Scottalexander    |    0.912214 | 0.907635 |   0.908436 |      28.2 |
| Zvi               |    0.993103 | 0.992857 |   0.992855 |      27.4 |
| macro avg         |    0.955397 | 0.953869 |   0.95363  |     137.2 |
| weighted avg      |    0.956595 | 0.95484  |   0.954759 |     137.2 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9417

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.95908  | 0.993103 |   0.975681 |      28.2 |
| Johnswentworth    |    0.960185 | 0.952414 |   0.955791 |      29.4 |
| Raemon            |    0.923027 | 0.875    |   0.892931 |      24   |
| Scottalexander    |    0.906575 | 0.89335  |   0.897702 |      28.2 |
| Zvi               |    0.972874 | 0.98545  |   0.978551 |      27.4 |
| macro avg         |    0.944348 | 0.939863 |   0.940131 |     137.2 |
| weighted avg      |    0.944841 | 0.941722 |   0.941438 |     137.2 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9359

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.973333 | 0.978571 |   0.975163 |      28.2 |
| Johnswentworth    |    0.973103 | 0.966207 |   0.969371 |      29.4 |
| Raemon            |    0.872617 | 0.866667 |   0.866264 |      24   |
| Scottalexander    |    0.89249  | 0.871921 |   0.879484 |      28.2 |
| Zvi               |    0.972167 | 0.98545  |   0.978432 |      27.4 |
| macro avg         |    0.936742 | 0.933763 |   0.933743 |     137.2 |
| weighted avg      |    0.938728 | 0.935872 |   0.935851 |     137.2 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9169

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.979064 | 0.950493 |   0.964133 |      28.2 |
| Johnswentworth    |    0.945436 | 0.938621 |   0.941629 |      29.4 |
| Raemon            |    0.853717 | 0.858333 |   0.85316  |      24   |
| Scottalexander    |    0.840667 | 0.857635 |   0.846322 |      28.2 |
| Zvi               |    0.972167 | 0.970899 |   0.971161 |      27.4 |
| macro avg         |    0.91821  | 0.915196 |   0.915281 |     137.2 |
| weighted avg      |    0.920028 | 0.916947 |   0.917107 |     137.2 |

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

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9665

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.993103 | 0.993103 |   0.992982 |      28.2 |
| Johnswentworth    |    0.966207 | 0.972644 |   0.969371 |      29.4 |
| Raemon            |    0.959253 | 0.925    |   0.94013  |      24   |
| Scottalexander    |    0.928992 | 0.943103 |   0.934656 |      28.2 |
| Zvi               |    0.992857 | 0.992857 |   0.992727 |      27.4 |
| macro avg         |    0.968082 | 0.965342 |   0.965973 |     137.2 |
| weighted avg      |    0.968137 | 0.966497 |   0.966616 |     137.2 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9592

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.972874 | 0.985961 |   0.978941 |      28.2 |
| Johnswentworth    |    0.973978 | 0.97954  |   0.976491 |      29.4 |
| Raemon            |    0.95092  | 0.891667 |   0.916967 |      24   |
| Scottalexander    |    0.918083 | 0.94335  |   0.928509 |      28.2 |
| Zvi               |    0.992857 | 0.985714 |   0.988956 |      27.4 |
| macro avg         |    0.961742 | 0.957246 |   0.957973 |     137.2 |
| weighted avg      |    0.961943 | 0.95923  |   0.959155 |     137.2 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9417

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.95954  | 0.971675 |   0.964764 |      28.2 |
| Johnswentworth    |    0.960154 | 0.952414 |   0.955914 |      29.4 |
| Raemon            |    0.8865   | 0.891667 |   0.885947 |      24   |
| Scottalexander    |    0.919814 | 0.907882 |   0.912799 |      28.2 |
| Zvi               |    0.98545  | 0.978307 |   0.981549 |      27.4 |
| macro avg         |    0.942292 | 0.940389 |   0.940195 |     137.2 |
| weighted avg      |    0.943967 | 0.941743 |   0.941774 |     137.2 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9242

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.985961 | 0.950739 |   0.967119 |      28.2 |
| Johnswentworth    |    0.945271 | 0.938621 |   0.941782 |      29.4 |
| Raemon            |    0.900505 | 0.891667 |   0.894757 |      24   |
| Scottalexander    |    0.834325 | 0.865517 |   0.847731 |      28.2 |
| Zvi               |    0.972381 | 0.970899 |   0.971013 |      27.4 |
| macro avg         |    0.927689 | 0.923489 |   0.924481 |     137.2 |
| weighted avg      |    0.928329 | 0.924246 |   0.925185 |     137.2 |

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

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9607

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.978799 | 0.978818 |   0.978686 |      28.2 |
| Johnswentworth    |    0.97931  | 0.95931  |   0.968784 |      29.4 |
| Raemon            |    0.941422 | 0.925    |   0.93053  |      24   |
| Scottalexander    |    0.921792 | 0.935961 |   0.927372 |      28.2 |
| Zvi               |    0.985961 | 1        |   0.992855 |      27.4 |
| macro avg         |    0.961457 | 0.959818 |   0.959645 |     137.2 |
| weighted avg      |    0.962016 | 0.96069  |   0.960406 |     137.2 |

### 5-Fold CV: Depth 2 (64, 32)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9491

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.979064 | 0.979064 |   0.97882  |      28.2 |
| Johnswentworth    |    0.945874 | 0.945287 |   0.945266 |      29.4 |
| Raemon            |    0.916178 | 0.908333 |   0.90965  |      24   |
| Scottalexander    |    0.927185 | 0.914778 |   0.91806  |      28.2 |
| Zvi               |    0.986207 | 0.992593 |   0.989084 |      27.4 |
| macro avg         |    0.950902 | 0.948011 |   0.948176 |     137.2 |
| weighted avg      |    0.951578 | 0.949074 |   0.949083 |     137.2 |

### 5-Fold CV: Depth 3 (64, 64, 64)

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.9447

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.959709 | 0.985961 |   0.972282 |      28.2 |
| Johnswentworth    |    0.960845 | 0.95908  |   0.959419 |      29.4 |
| Raemon            |    0.907893 | 0.9      |   0.902915 |      24   |
| Scottalexander    |    0.90094  | 0.879557 |   0.889443 |      28.2 |
| Zvi               |    0.993103 | 0.992593 |   0.992718 |      27.4 |
| macro avg         |    0.944498 | 0.943438 |   0.943355 |     137.2 |
| weighted avg      |    0.945481 | 0.944663 |   0.944471 |     137.2 |

### 5-Fold CV: Depth 10

- **Mean Training Accuracy:** 1.0000
- **Mean Testing Accuracy:** 0.8980

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.91582  | 0.92931  |   0.920124 |      28.2 |
| Johnswentworth    |    0.9375   | 0.897701 |   0.91606  |      29.4 |
| Raemon            |    0.876369 | 0.875    |   0.872244 |      24   |
| Scottalexander    |    0.780388 | 0.794089 |   0.785309 |      28.2 |
| Zvi               |    1        | 0.992593 |   0.996226 |      27.4 |
| macro avg         |    0.902015 | 0.897739 |   0.897992 |     137.2 |
| weighted avg      |    0.90241  | 0.89799  |   0.898356 |     137.2 |

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

