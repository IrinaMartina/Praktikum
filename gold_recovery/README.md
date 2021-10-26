# Прогнозирование концентрации золота

Представлены сырые данные с параметрами производства и значениями концентраций веществ на разных стадиях производства.

## Что сделала
Разобрала и проанализировала технологический процесс, выбрала важные признаки, обучила модели предсказывать эффективность восстановления золота на разных этапах производства: после флотации и после финальной очистки. Результаты предсказаний первой модели использовала в качестве ещё одного признака для второй.

## Использованные библиотеки
```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, make_scorer
```
