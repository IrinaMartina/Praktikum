# Прогнозирование количества заказов на следующий час

Дан временной ряд: количество заказов таксти за полгода с шагом в 10 минут.

## Что сделала

Проанализировала временной ряд, провела ресемплинг, построила графики, выделила важные признаки, обучила линейную регрессию, случайный лес и градиентный бустинг. Подобрала гиперпараметры. Выбрала лучшую модель. Проверила предсказания на адекватность.

## Использованные библиотеки
```
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
```
