# Praktikum
Мои самостоятельные работы в Яндекс.Практикуме. Отсортированы по дате создания: чем старше, тем ниже в списке.

1. *wikishop* - **Проект для «Викишоп»** - оценка комментариев на токсичность

Очистила тексты на английском языке от лишних символов, провела лемматизацию. Создала векторы на основе значений важности слов и биграмм. Для использования кросс-валидации, векторизацию и модель объединила в один пайплайн. Рассмотрела модели логистической регрессии и градиентного бустинга с разными гиперпараметрами, выбрала лучшую. Для ориентира посчитала метрику F1 для dummy-модели.

2. *taxi_orders* - **Прогнозирование заказов такси**

Проанализировала временной ряд, построила графики, выделила важные признаки, обучила линейную регрессию, случайный лес и градиентный бустинг. Подобрала гиперпараметры. Выбрала лучшую модель. Проверила предсказания на адекватность.

3. *cost_of_cars* - **Определение стоимости автомобилей**

Провела предобработку данных: заполнила пропуски в категориальных признаках, нашла и обработала аномалии в численных. Удалила объекты, у которых не заполнено больше половины признаков. В задаче было важно выбрать модель по трём критериям: точность предсказания, время на обучение, время на предсказание. Я показала, что случайный лес и градиентный бустинг дают лучшее значение RMSE, чем линейная регрессия, и бустинг обучается быстрее леса.

4. *personal_data* - **Защита персональных данных клиентов**

Показала, что умножение матрицы признаков на случайную обратимую матрицу не влияет на качество линейной регрессии.

5. *gold_recovery* - **Восстановление золота из руды**

Разобрала и проанализировала технологический процесс, выбрала важные признаки, обучила модели предсказывать эффективность восстановления золота на разных этапах производства: после флотации и после финальной очистки. Результаты предсказаний первой модели использовала в качестве ещё одного признака для второй.

6. *location for wells* - **Выбор локации для скважины**

Для трёх регионов дана информация о скважинах месторождений: показатели качества нефти и объёмы. Нужно выбрать один регион для разработки, который принесёт наибольшую прибыль, а риск убытков будет меньше заданного процента. В этом регионе исследуют 500 точек и с помощью модели определят 200 лучших для добычи нефти.

Показала, что выбрать 200 случайных скважин недостаточно, чтобы покрыть затраты на разработку. Обучила модель находить лучшие скважины для каждого региона. И для каждого региона с помощью техники Bootstrap смоделировала 1000 ситуаций выбора 200 лучших скважин из 500 случайных. Посчитала средние значения ожидаемой прибыли и 95%-ый доверительный интервал, а также вероятность убытка. Сделала сводную таблицу и написала выводы.

7. *customer_churn* - **Отток клиентов**

Обучила модель находить клиентов, которые собираются уйти. Использовала разные техники борьбы с дисбалансом классов: меняла class_weight, увеличивала выборку положительных объектов, уменьшала выборку отрицательных. Также меняла порог классификации. Выбрала лучший вариант на основе метрики F1, также смотрела на ROC-кривую.

8. *tariff_recommendation* - **Рекомендация тарифов**

Решила задачу классификации: определила подходящий тариф по информации о поведении клиента. Обучила несколько моделей, выбрала лучшую.

9. *success_of_video_games* - **Определение критериев успешности видео игр** 

Проведена предобработка данных исследовательский анализ данных, статистический анализ данных. Выделен актуальный период времени, определено время жизни игровых платформ, выявлены различия рынков Японии, Европы и Америки. 

10. *best_mobile_tariff* - **Определение перспективного тарифа для телеком компании**

Проанализировала данные, сформулировала две нулевые гипотезы и провела t-тест для каждой. На основании выборки определила пороговое значение alpha для p-value. Первую нулевую гипотезу удалось опровергнуть и подтвердить таким образом верность альтернативной гипотезы: траты пользователей разных тарифов отличаются. Вторую нулевую гипотезу опровергнуть не удалось: p-value оказалось больше alpha. Это означает, что нельзя сделать какие-то определённые выводы о различиях в тратах москвичей и пользователей из других регионов.
