# Телеком

Оператор связи «Ниединогоразрыва.ком» хочет научиться прогнозировать отток клиентов. Если выяснится, что пользователь планирует уйти, ему будут предложены промокоды и специальные условия. Команда оператора собрала персональные данные о некоторых клиентах, информацию об их тарифах и договорах. Данные представлены в четырёх таблицах.

Пока создаётся модель прогнозирования оттока клиентов, отдел маркетинга «Ниединогоразрыва.ком» приступает к проработке способов их удержания. Необходимо собрать нужную для этого информацию:

- сравнить распределения величин ежемесячных платежей всех клиентов и тех, кто ушёл;
- сравнить поведение клиентов этих двух групп, для каждой изучить:
    - долю пользователей телефонной связи;
    - долю интернет-пользователей.

## Что сделала

*prognosis_of_churn.ipynb*:

Проанализировала данные, создала новые признаки, определила корреляции, использовала критерий V Крамера и изучила значения объясняющей способности каждого признака по отношению к целевому. Использовала кластеризацию (k-means) и метод главных компонент (PCA). Обучила различные модели: логистическую регрессию, случайный лес, градиентный бустинг - для каждой подобрала оптимальные гиперпараметры. Опробовала различные способы борьбы с дисбалансом: upsampling, downsampling, SMOTE. Изучила важность признаков и обучила на вероятностях стек-модель - RidgeClassifier. Проанализировала стоимость ложно положительных и ложно отрицательных ответов модели, выбрала оптимальный порог вероятности. 

*analysis_of_churn.ipynb*:

Проанализировала влияние ежемесячной стоимости на уход клиента и отличие в поведении ушедших клиентов и оставшихся. Применила t-тест и хи-квадрат.

## Использованные библиотеки
*prognosis_of_churn.ipynb*:
```python
import pandas as pd
import numpy as np

#Для графиков
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

#Q-Q-plot
import statsmodels.api as sm

#Для поиска категориальных столбцов
from pandas.api.types import is_categorical_dtype

#Для расчёта кореляции категориальных признаков
from scipy.stats import chi2_contingency

#Для определения важности признаков
from sklearn.feature_selection import mutual_info_classif

#Для подготовки числовых и категориальных признаков
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, QuantileTransformer

#Для создания новых признаков
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#Для выделения тестовой выборки
from sklearn.model_selection import train_test_split

#Модели
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

#Для поиска гиперпараметров и оценки моделей
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

#Для борьбы с дисбалансом
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

#Метрики
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
```
*analysis_of_churn.ipynb*:
```python
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as st
```
