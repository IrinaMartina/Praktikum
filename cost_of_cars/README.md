# Определение стоимости автомобилей
На основании исторических данных необходимо обучить модель предсказывать стоимость автомобиля. В равной степени важны: качество предсказания, скорость предсказания, время обучения.

## Что сделала
Провела предобработку данных: заполнила пропуски в категориальных признаках, нашла и обработала аномалии в численных. Удалила объекты, у которых не заполнено больше половины признаков. Показала, что случайный лес и градиентный бустинг дают лучшее значение RMSE, чем линейная регрессия, и бустинг обучается быстрее леса.

## Использованные библиотеки
```
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from category_encoders import MEstimateEncoder

from lightgbm import LGBMRegressor

from time import time

import matplotlib.pyplot as plt
import seaborn as sns
```
