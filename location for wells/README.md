
# Выбор локации для скважины

Для трёх регионов дана информация о скважинах месторождений: показатели качества нефти и объёмы. Нужно выбрать один регион для разработки, который принесёт наибольшую прибыль, а риск убытков будет меньше заданного процента. В этом регионе исследуют 500 точек и с помощью модели определят 200 лучших для добычи нефти.

## Что сделала
Показала, что выбрать 200 случайных скважин недостаточно, чтобы покрыть затраты на разработку. Обучила модель находить лучшие скважины для каждого региона. И для каждого региона с помощью техники Bootstrap смоделировала 1000 ситуаций выбора 200 лучших скважин из 500 случайных. Посчитала средние значения ожидаемой прибыли и 95%-ый доверительный интервал, а также вероятность убытка. Сделала сводную таблицу и написала выводы.

## Использованные библиотеки
```
import pandas as pd
from numpy.random import RandomState
from IPython.display import display
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```
