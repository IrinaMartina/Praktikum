# Определение возраста покупателей по фотографии

Данные представлены в виде набора фотографий и датафрейма с двумя колонками: имя файла и возраст.

Необходимо определить возраст покупателя, чтобы анализировать покупки разных возрастных групп и предлагать товары, которые могут быть им интересны. Кроме того, возраст покупателя важен при продаже алкоголя.

Цель - достичь значения MAE на тестовой выборке меньше 8-и лет.

## Что сделала
Проанализировала датасет, разделила изображения на обучающую и тестовую выборки, применила аугментации. Подобрала подходящую структуру нейронной сети и обучила её. Добилась хорошего значения метрики: MAE=5,56 лет.

## Использованные библиотеки
```python
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
```
