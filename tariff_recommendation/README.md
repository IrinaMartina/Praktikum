# Рекомендация тарифов
Представлены исторические данные о поведении клиентов.

## Что сделала
Решила задачу классификации: определила подходящий тариф по информации о поведении клиента. Обучила несколько моделей, выбрала лучшую.

## Использованные библиотеки
```
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
