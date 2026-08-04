# Исправление transductive leakage и контрольный пересчёт

Дата прогона: 29 июля 2026 года.

## Результат

В `aux_conditioning.ipynb`, `xuv_max.ipynb` и `aux_pretrain.ipynb` устранён
одинаковый источник transductive leakage: классификатор планеты теперь
обучается только на известных метках training split. Основной эксперимент
`aux_conditioning` полностью переобучен с нуля, включая пять CNN и ранжировщик.

Проверка нового run:

- 466 известных training-строк участвовали в fit классификатора планеты;
- 0 validation-строк участвовали в fit;
- среди 31 неизвестной метки исправление изменило 11 псевдометок относительно
  старого протокола: 8 в train и 3 в validation;
- метрики независимо пересчитаны из сохранённых предсказаний на 123 объектах;
- checkpoints, preprocessing, split-ID, SHA-256 входных файлов, окружение и
  raw predictions сохранены.

## Метрики clean run

Все числа ниже относятся к прежнему development holdout 123/613. Это не
независимый test.

| Режим оценки | XUV MAPE ↓ | He MAPE ↓ | logMsw MAPE ↓ | logMsw MAE, dex ↓ | linear Msw MAPE ↓ |
|---|---:|---:|---:|---:|---:|
| Diverse best-of-5, coordinate-wise oracle | **9.0165%** | **8.6242%** | **0.6594%** | **0.08254** | **19.6596%** |
| Oracle по всем 25 кандидатам | 4.2051% | 4.0262% | 0.3214% | 0.03988 | 9.0280% |
| Один совместный 3D-кандидат, oracle из всех 25 | 13.5327% | 9.6513% | 0.6499% | 0.08233 | 18.3860% |
| Один совместный 3D-кандидат, oracle внутри selected-5 | 24.0894% | 15.3972% | 0.8657% | 0.10914 | 27.1338% |
| Ranker top-1, применимо без истины | 22.1392% | 51.7733% | 1.4717% | 0.18691 | 56.4013% |
| Среднее 25 кандидатов | 31.2125% | 24.2428% | 1.5336% | 0.19464 | 51.3651% |

Классификация:

| Метрика | Значение |
|---|---:|
| H2a ROC-AUC | **0.996827** |
| H2a accuracy, threshold 0.5 | **98.374%** |
| Planet classifier accuracy на 116 известных validation-метках | **93.966%** |

Headline `best-of-5` — это метрика покрытия: для каждой координаты постфактум
выбирается лучший из пяти кандидатов. Она не равна ошибке единственного ответа.
Для single-answer корректнее смотреть на `ranker top-1`; особенно заметен
разрыв по He: 8.62% oracle против 51.77% top-1.

## Сравнение со старыми notebook outputs

| Метрика | Старый leaky output | Новый train-only run | Δ new − old |
|---|---:|---:|---:|
| XUV MAPE | 9.8% | 9.0165% | −0.7835 п.п. |
| He MAPE | 8.9% | 8.6242% | −0.2758 п.п. |
| logMsw MAPE | 0.5% | 0.6594% | +0.1594 п.п. |
| H2a ROC-AUC | 0.998 | 0.996827 | −0.001173 |

Это ориентировочное, а не причинное сравнение: старая строка взята из
округлённых сохранённых outputs notebook и не была повторно обучена тем же
standalone runner в том же окружении. Поэтому дельту нельзя приписывать только
исправлению leakage.

## Что именно исправлено

Старый протокол:

```python
clf.fit(Z[known], pnames[known])
```

использовал 116 известных validation-строк. Затем предсказания этого
классификатора становились supervision для неизвестных training-строк.

Новый протокол:

```python
fit_idx = idx_train[known[idx_train]]
clf.fit(Z[fit_idx], pnames[fit_idx])
```

использует только `known ∩ train`. Для `aux_pretrain` применяется эквивалентный
индекс `tr3`. Неизвестные training-объекты не входят в fit классификатора,
поэтому отдельный OOF-проход для них не требуется.

## Данные, архитектура и обучение

- 613 профилей, grid 101 точка от −5 до 5 с шагом 0.1.
- Stratified split по `H2a`, seed 42: 490 train / 123 validation.
- `log10` targets: XUVInt, Helium, Msw.
- Нормализация targets, масштаб профиля и PCA-16 fit только на train.
- Вход CNN: два канала, linear profile и нормированный log-profile.
- Encoder: Conv1d `2→64→128→128`, kernel 5, BatchNorm, ReLU, MaxPool.
- Planet head: embedding 1536 → 64 → 3; softmax смешивает три физических
  9-мерных вектора планет.
- Пять regression heads: 1545 → 256 → 3, dropout 0.2.
- H2a head: 1536 → 128 → 1, dropout 0.2.
- Всего 2,402,643 обучаемых параметра на CNN.
- Пять seeds: 0, 1, 2, 3, 4; 400 эпох; batch 64.
- Adam, learning rate `1e-3`; CosineAnnealingLR `T_max=400`.
- Loss: WTA-regression + `0.1 × mean-head loss` + H2a BCE +
  `0.3 × planet CE`.
- Лучшие эпохи: 106, 236, 172, 299, 189.
- Ранжировщик: MLP `19→256→256→1`, Adam `1e-3`,
  weight decay `1e-5`, 3000 steps, batch 256.
- Окружение: Python 3.12.7, PyTorch 2.12.1, NumPy 1.26.4,
  scikit-learn 1.5.1; CPU, 4 threads.
- Полный run: 1886.5 секунды, около 31 мин 27 с.

## Воспроизведение

Из каталога `pro`:

```bash
/opt/anaconda3/bin/python run_aux_conditioning.py \
  --planet-label-mode train-only \
  --epochs 400 \
  --model-seeds 0,1,2,3,4 \
  --ranker-steps 3000 \
  --output-dir runs/aux_conditioning_train_only
```

Независимая проверка сохранённых метрик:

```bash
/opt/anaconda3/bin/python verify_aux_conditioning_run.py \
  runs/aux_conditioning_train_only
```

## Остаточные ограничения

1. Holdout используется для выбора лучшей эпохи и для отчётных метрик, поэтому
   это development validation, а не независимый test.
2. Сильные best-of-5 числа являются oracle-оценкой покрытия набора кандидатов.
3. Исторические outputs `xuv_max` и `aux_pretrain` созданы до исправления.
   Их source-код исправлен, но метрики нельзя считать clean до полного retrain.
4. Для честного причинного измерения эффекта leakage нужен paired run
   `notebook-compat` против `train-only` в одном окружении.
5. Для финального вывода нужен новый untouched test либо group/planet holdout.
6. В отдельном `aux_pretrain` остаётся другой вопрос: статистики
   синтетических targets вычисляются до synthetic split. Это не относится к
   исправленной planet-pseudolabel leakage, но требует отдельной правки и
   полного retrain перед использованием его сохранённых метрик.
