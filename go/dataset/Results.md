# Zimtohrli listening test correlation results

_Updated at 2024-03-21 by zond@_

This document is intended to contain the most recent results of running the code
that calculates objective metrics on the available datasets and then correlates
them with the subjective human scores.

## coresv.net dataset

```
[
  {
    "ScoreType": "dnsmos_bak",
    "Spearman": 0.029476400990680246,
    "PerReference": {
      "SpearmanMedian": 0.6000000000000001,
      "SpearmanMean": 0.4975808365146109,
      "SpearmanStdDev": 0.4117346179316666
    }
  },
  {
    "ScoreType": "dnsmos_ovr",
    "Spearman": 0.026562573959647003,
    "PerReference": {
      "SpearmanMedian": 0.5428571428571429,
      "SpearmanMean": 0.37329512222889655,
      "SpearmanStdDev": 0.417464457916757
    }
  },
  {
    "ScoreType": "dnsmos_sig",
    "Spearman": 0.08327737621331376,
    "PerReference": {
      "SpearmanMedian": 0.3142857142857143,
      "SpearmanMean": 0.25124611780855793,
      "SpearmanStdDev": 0.46025751382062546
    }
  },
  {
    "ScoreType": "gvpmos",
    "Spearman": 0.025729617893424296,
    "PerReference": {
      "SpearmanMedian": 0.3142857142857143,
      "SpearmanMean": 0.1542857142857143,
      "SpearmanStdDev": 0.5031035517018663
    }
  },
  {
    "ScoreType": "parlaq",
    "Spearman": 0.46457109106533717,
    "PerReference": {
      "SpearmanMedian": 0.5161002296110789,
      "SpearmanMean": 0.5359841706092862,
      "SpearmanStdDev": 0.23948094336817632
    }
  },
  {
    "ScoreType": "visqol",
    "Spearman": 0.8058179322089262,
    "PerReference": {
      "SpearmanMedian": 0.8285714285714285,
      "SpearmanMean": 0.8353649828097949,
      "SpearmanStdDev": 0.13190620235030912
    }
  },
  {
    "ScoreType": "warp-q",
    "Spearman": -0.5090216657611891,
    "PerReference": {
      "SpearmanMedian": -0.5999999999999999,
      "SpearmanMean": -0.6154692636125932,
      "SpearmanStdDev": 0.18886766008744393
    }
  },
  {
    "ScoreType": "Zimtohrli",
    "Spearman": -0.642889294934578,
    "PerReference": {
      "SpearmanMedian": -0.8285714285714285,
      "SpearmanMean": -0.7582846941486171,
      "SpearmanStdDev": 0.1805314991296263
    }
  }
]
```