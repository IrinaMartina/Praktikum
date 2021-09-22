# Категоризация текста: токсичный/не токсичный

Данные представлены в виде наборов текстов на английском языке и отметок, является ли текст токсичным: 1 или 0.

## Что сделала
Очистила тексты на английском языке от лишних символов, провела лемматизацию. Создала векторы на основе значений важности слов и биграмм. Для использования кросс-валидации, векторизацию и модель объединила в один пайплайн. Рассмотрела модели логистической регрессии и градиентного бустинга с разными гиперпараметрами, выбрала лучшую. Для ориентира посчитала метрику F1 для dummy-модели.

## Использованные библиотеки
```
import pandas as pd
import numpy as np

import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier
```
