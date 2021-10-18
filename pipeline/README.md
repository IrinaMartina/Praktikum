# Пайплайн для DS
Набор универсальных инструментов

## Что сделала
Создала функции для анализа и предобработки данных, а также для обучения моделей, кросс-валидации и поиска гиперпараметров.

1 Анализ

1.1 Загрузка данных

  - считать csv
  - вывести head и info
  - посмотреть на пропуски и дубликаты
  - изучить распределения числовых признаков
  - посмотреть корреляции
  - изучить категориальные признаки

1.2 Предобработка данных

  - заполнить пропуски
  - обработать аномальные значения
  - изменить типы данных

1.3 Подготовка признаков

  - закодировать категориальные
  - масштабировать и нормализовать числовые
  - добавить новые признаки
  - оценить важность признаков

2 Пайплайн - пример последовательного использования всех функций

3 Обучение

3.1 Предобработка

3.2 Регрессия

3.3 Классификация


## Использованные библиотеки
```
import pandas as pd
import numpy as np

#Для графиков
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(context='talk', style='whitegrid', palette='muted')
from sklearn.manifold import TSNE

#Q-Q-plot
import statsmodels.api as sm

#Для поиска категориальных столбцов
from pandas.api.types import is_categorical_dtype

#Для подготовки числовых и категориальных признаков
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler

#Для преобразования Бокса-Кокса
from scipy.stats import boxcox

#Для создания новых признаков
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#Модели
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

#Для поиска гиперпараметров и оценки моделей
import optuna
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
```
