
# Прогнозирование оттока клиентов

Представлены исторические данные о поведении клиентов.

## Что сделала
Обучила модель находить клиентов, которые собираются уйти. Использовала разные техники борьбы с дисбалансом классов: меняла class_weight, увеличивала выборку положительных объектов, уменьшала выборку отрицательных. Также меняла порог классификации. Выбрала лучший вариант на основе метрики F1, также смотрела на ROC-кривую.

## Использованные библиотеки
```
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, roc_curve 
from sklearn.utils import shuffle
```
