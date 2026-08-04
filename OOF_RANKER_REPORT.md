# OOF-ranker: воспроизведённый leakage-aware pilot

Дата запуска: 2026-07-29.

Основной run:
`pro/runs/oof_ranker_pilot_v2`.

Скрипт:
`pro/run_oof_ranker_experiment.py`.

## 1. Краткий ответ

Предложенная логика работает в ограниченном смысле: ранкер, обученный на
реальных out-of-fold кандидатах, лучше старого ранкера выбирает **один ответ
для XUV**. На primary development subset из 118 объектов:

| Метод на одном и том же пуле из 10 кандидатов | XUV top-1 MAPE | He top-1 MAPE | logMsw top-1 MAPE | mean joint distance | simultaneous hit |
|---|---:|---:|---:|---:|---:|
| Legacy synthetic-pair ranker | 23.7096% | 40.2932% | **1.4446%** | **0.4744** | **30.51%** |
| OOF XUV objective | 19.5188% | 28.3065% | 1.5462% | 0.4900 | 26.27% |
| OOF joint objective | 21.8546% | 39.4413% | **1.3731%** | **0.4597** | **31.36%** |
| **OOF combined, выбран по OOF-CV** | **19.1166%** | **27.5244%** | 1.5081% | 0.4855 | 24.58% |

`combined` улучшает XUV относительно legacy на **4.593 процентного пункта**,
или на **19.4% относительно**. Парный percentile bootstrap по 118 объектам
даёт условный 95% CI разности `OOF − legacy`
`[-9.559; -0.247]` п.п.

Но это не полная победа:

- физическая joint-согласованность top-1 стала немного хуже;
- для shortlist из пяти кандидатов legacy остаётся не хуже;
- 7–11% XUV по-прежнему относятся к oracle/coverage по нескольким кандидатам,
  а не к автоматически выбранному одному ответу;
- 118 объектов — ранее просмотренный development subset, не новый blind test.

Практический вывод: текущий OOF-ranker — рабочее улучшение single-answer XUV,
но ещё не доказательство решения с `MAPE < 15.1%` для одного ответа и не
основание заявлять общее превосходство над статьёй.

## 2. Как objective выбирался без подглядывания в development

Все три objective были сравнены по pooled producer-fold cross-validation.
Для каждого из пяти producer folds:

1. generator-кандидаты уже были OOF;
2. ranker обучался без score-объектов этого fold;
3. эпоха выбиралась на целом producer fold;
4. после выбора эпохи модель заново обучалась на всех 490 OOF-объектах.

Pooled top-1 XUV MAPE:

| Producer fold, N | XUV objective | Joint objective | Combined objective |
|---|---:|---:|---:|
| 0, 98 | 26.02% | 25.12% | 26.75% |
| 1, 97 | 27.67% | 32.34% | 25.85% |
| 2, 93 | 27.76% | 29.28% | 25.18% |
| 3, 104 | 36.05% | 66.37% | 35.20% |
| 4, 98 | 25.35% | 28.96% | 26.95% |
| **Weighted pooled** | 28.674% | 36.862% | **28.107%** |

До просмотра development было зафиксировано правило:

1. минимальный pooled XUV MAPE;
2. если разница меньше 1 п.п. — больший simultaneous hit;
3. затем joint-coverage.

`combined` и `xuv` отличаются лишь на 0.567 п.п. В tie-break у `combined`
simultaneous hit `15.71%`, у `xuv` — `13.06%`; поэтому frozen выбор —
`combined`. Переключаться после просмотра development на более удобный
`joint` было бы post-hoc tuning.

Fold 3 заметно сложнее остальных. У его двух generators последний training
loss равен `0.3357` и `0.3653`, тогда как на других folds он примерно
`0.15–0.19`. На 200-й эпохе loss ещё снижался. Это аргумент в пользу следующей
предварительно зарегистрированной абляции с большим одинаковым fixed budget,
но не повод менять epochs после просмотра fold.

## 3. Метрики практического ответа и покрытия

### 3.1. Один автоматически выбранный кандидат

Главная deployable метрика — MAPE кандидата с максимальным score, без доступа
к истине.

| Primary 118 | Legacy | OOF combined | Разность |
|---|---:|---:|---:|
| XUV MAPE | 23.7096% | **19.1166%** | **−4.5930 п.п.** |
| He MAPE | 40.2932% | **27.5244%** | −12.7688 п.п. |
| logMsw MAPE | **1.4446%** | 1.5081% | +0.0635 п.п. |
| logMsw MAE | **0.1837 dex** | 0.1915 dex | +0.0078 dex |
| XUV hit ≤10% | 32.20% | **34.75%** | +2.55 п.п. |
| XUV hit ≤20% | 59.32% | **61.02%** | +1.70 п.п. |
| mean normalized joint distance | **0.4744** | 0.4855 | +0.0112 |
| joint distance ≤0.5 | **72.88%** | 67.80% | −5.08 п.п. |
| simultaneous XUV≤20%, He≤20%, Msw≤0.1 dex | **30.51%** | 24.58% | −5.93 п.п. |

В ranking diagnostics новый ранкер существенно лучше ориентирован на XUV:

| Диагностика | Legacy | OOF combined |
|---|---:|---:|
| mean Spearman(score, −XUV APE) | 0.341 | **0.623** |
| recall глобального XUV argmin @1 | 17.80% | **38.14%** |
| recall глобального XUV argmin @3 | 45.76% | **64.41%** |
| recall глобального XUV argmin @5 | 66.10% | **82.20%** |

Это подтверждает именно улучшение ранжирования XUV. Оно не означает, что
первый кандидат стал лучшим совместно по всем физическим координатам.

### 3.2. Ranked prefix и MMR shortlist

Для `k>1` MAPE ниже — это **coverage/oracle внутри предъявленного shortlist**.
По каждой координате берётся минимальная ошибка среди `k` кандидатов; это не
ошибка одного совместного ответа.

| XUV coverage | Legacy | OOF combined | Oracle всех 10 |
|---|---:|---:|---:|
| Top-1 | 23.7096% | **19.1166%** | 8.0111% |
| Ranked top-3 | 14.6290% | **12.9572%** | 8.0111% |
| Ranked top-5 | **10.3710%** | 10.6980% | 8.0111% |
| MMR-selected 5 | **10.1763%** | 10.4667% | 8.0111% |

Новый ранкер сокращает top-1 selection regret относительно oracle-all:

- legacy: `23.7096 − 8.0111 = 15.6985` п.п.;
- OOF combined: `19.1166 − 8.0111 = 11.1055` п.п.;
- устранено около `29.3%` legacy top-1 regret.

При `k=5` разница меняет знак и мала. Joint safety у legacy также выше:

| Top-5 safety | Legacy | OOF combined |
|---|---:|---:|
| mean minimum joint distance | **0.3309** | 0.3507 |
| joint distance ≤0.5 | **84.75%** | 83.05% |
| simultaneous hit | **54.24%** | 48.31% |
| XUV hit ≤20% | **86.44%** | 82.20% |

Следовательно, текущий OOF-ranker следует использовать как single-answer
XUV selector. Для пяти последовательных forward-проверок он не показал
преимущества над legacy shortlist.

## 4. Что именно было обучено

### 4.1. Candidate generator

Архитектура та же, что в clean conditioned run:

- вход: профиль `2 × 101`;
- Conv1d backbone `2→64→128→128`, kernel 5, BN, ReLU, MaxPool;
- пять совместных регрессионных heads на
  `(logXUV, logHe, logMsw)`;
- H2a head;
- planet auxiliary head с train-only pseudolabel classifier;
- physical conditioning через таблицу из девяти planet prototypes.

Pilot использует два seeds, поэтому один объект получает `2 × 5 = 10`
совместных кандидатов.

OOF:

- `StratifiedGroupKFold`, 5 folds;
- группы — транзитивное объединение exact target tuples и byte-identical
  интерполированных профилей;
- для каждого producer fold отдельно fit-ятся target normalization,
  profile scale, PCA и planet classifier;
- prediction rows не участвуют в gradients, preprocessing, pseudolabel fit
  или checkpoint selection.

Обучение каждого generator:

- 200 фиксированных эпох;
- batch size 64;
- Adam, LR `1e−3`;
- CosineAnnealingLR, `T_max=200`;
- WTA loss + BCE(H2a) + `0.3 × CE(planet)`;
- Gaussian noise только на train;
- checkpoint selection отсутствует.

После OOF-генерации два generators переобучены на всех 490 outer-train
объектах с теми же 200 fixed epochs и применены к 123 historical development
объектам.

### 4.2. Ranker

На каждый кандидат подаются 27 признаков:

- 16 train-fitted PCA признаков профиля;
- 3 нормированные координаты кандидата;
- 3 отклонения от pool median;
- 3 абсолютных отклонения;
- расстояние до pool median;
- среднее расстояние до трёх ближайших кандидатов.

Модель:

```text
27 → 64 → 32 → 1
ReLU, dropout 0.1
```

Оптимизация:

- AdamW, LR `1e−3`, weight decay `1e−4`;
- listwise cross-entropy к `softmax(−cost / temperature)`;
- batch: 32 объекта;
- максимум 250 эпох;
- five-seed ensemble;
- число эпох каждого seed выбирается по целому producer fold;
- затем каждый seed refit-ится на всех 490 OOF rows.

Objectives:

- `xuv`: XUV absolute percentage error;
- `joint`: евклидово расстояние в трёх нормированных координатах;
- `combined`: `0.5 × scaled XUV APE + 0.5 × scaled joint distance`.

В `combined` XUV намеренно имеет повышенный вес, потому что входит отдельно и
ещё раз как часть joint distance. Scale для combined fit-ится только на
ranker-train части, а не на held-out producer fold.

## 5. Leakage boundaries

Что устранено:

- ни одна OOF prediction row не входит в producer gradients;
- fold-specific `y_mean/y_std`, `x_scale`, PCA и planet classifier fit-ятся
  без prediction rows;
- `planet_classifier_fit_prediction_rows = 0` во всех 12 generator runs;
- exact duplicate group не пересекает producer train и producer validation;
- final generators не используют outer development ни в gradients, ни в
  preprocessing, ни в pseudolabel fit, ни в выборе checkpoint;
- internal ranker validation удерживает целый producer fold;
- после выбора epochs ranker refit-ится на всех OOF rows;
- primary оценка исключает пять development rows из пяти exact duplicate
  groups, пересекающих outer train.

Что остаётся ограничением:

1. Исходный outer split исторический и был просмотрен в прежних
   экспериментах. Primary 118 — очищенный от exact overlap development subset,
   но не новый test.
2. Ranker-level PCA и target normalization для final representation fit-ятся
   на всех 490 outer-train rows. Это не outer-dev leakage, но producer-fold
   CV не является полностью nested относительно этой нормировки.
3. Fold 3 показывает distribution shift между generators, обученными примерно
   на 390 объектах, и final generators, обученными на 490.
4. Candidate budget равен 10, тогда как некоторые прошлые runs используют 25,
   40 или 105 кандидатов. Oracle числа нельзя сравнивать без одинакового
   budget.
5. Objective, architecture и thresholds разработаны на уже изученном
   датасете. Для научного claim нужен новый заранее замороженный test.

Итого: transductive leakage outer development в candidate generator устранена.
Результат корректен как leakage-aware development pilot. Он не является
полностью nested или независимым финальным benchmark.

## 6. Сопоставление со статьёй

Логика статьи корректна, если её best-of-5 MAPE называется **oracle coverage**
или upper bound. Она некорректна, если это число интерпретируется как ошибка
ranker top-1.

В локальном аудите для `paper_comparison A` сохранено:

```text
XUV 15.1%, He 12.4%, logMsw 0.8% — oracle best-of-5.
```

Текущий pilot:

- deployable OOF-combined top-1 XUV: `19.12%`;
- ranked top-3 XUV coverage: `12.96%`;
- ranked top-5 XUV coverage: `10.70%`;
- oracle всех 10: `8.01%`.

Поэтому:

- нельзя утверждать, что один автоматически выбранный ответ уже лучше
  article oracle `15.1%`: `19.12% > 15.1%`;
- можно сказать, что текущий shortlist имеет численно меньшее development
  coverage MAPE, но split, candidate budget и протокол отличаются;
- доказательство превосходства требует одинакового split/budget и нового
  untouched test;
- статья не публикует сопоставимую top-1 MAPE ранкера, поэтому прямое
  практическое сравнение single-answer отсутствует.

## 7. Воспроизводимость

Команда полного запуска:

```bash
/opt/anaconda3/bin/python pro/run_oof_ranker_experiment.py \
  --folds 5 \
  --generator-seeds 0,1 \
  --generator-epochs 200 \
  --ranker-seeds 0,1,2,3,4 \
  --ranker-epochs 250 \
  --output-dir pro/runs/oof_ranker_pilot_v2
```

Среда:

```text
Python        3.12.7
PyTorch       2.12.1
NumPy         1.26.4
SciPy         1.13.1
scikit-learn  1.5.1
pandas        2.2.2
device        CPU
threads       4
```

Суммарное измеренное время компонентов:

- OOF generators: `1741.50 s`;
- final generators: `515.93 s`;
- rankers: `121.56 s`;
- всего: `2378.99 s`, около `39.65 min`.

Ключевые артефакты:

| Файл | SHA-256 |
|---|---|
| `config.json` | `eb9ca3b89e4603ddf384599702871cb6a52f6bb606f1d8066f0c1d6a161af798` |
| `oof_candidates.npz` | `c8036fe2f4e2c1391f31f6976cf6784e714a067dfa70f31c0e2d0299fc7d550d` |
| `dev_candidates.npz` | `d10e7b9331303286fd954f5182c3470ca5ad3f94c5f3eee8f1458a438100f1be` |
| `metrics.json` | `c9bbf77e175369c2da87cf879a3be360cabeb6f62b17ce2663b57987bea2bcf1` |

Source SHA-256:

| Source | SHA-256 |
|---|---|
| `run_oof_ranker_experiment.py` | `ae18546245c0449be11046a5ff63ed402d7904864509d3346d2553a4f427bf72` |
| `run_aux_conditioning.py` | `73418b0c492b0632ec417ab76d6eb903cf1d058dc39707e7a36c4a926b6062b0` |
| `planet_pseudolabels.py` | `1ed0861c80d8c3ab0a8cea373b3dacd7736b34974cb771b158412b24e20f19b2` |

Полный список hashes находится в `artifact_manifest.json`; split/status каждого
объекта — в `dataset_manifest.csv`; producer provenance — в
`oof_manifest.csv`.

Независимая replay-проверка прошла без расхождений: совпали `27/27` hashes
артефактов и `4/4` source hashes; финальные generator checkpoints
бит-в-бит воспроизвели development candidates, 15 ranker checkpoints —
сохранённые scores, а независимый пересчёт ключевых метрик дал максимальную
разность `0`.

Оставшийся bundle-риск: checkpoints и preprocessors пяти OOF-generator folds,
а также промежуточные состояния выбора эпох ranker не сохраняются. Поэтому
final inference и арифметика метрик replayable напрямую, а provenance OOF
training восстанавливается детерминированным переобучением, не загрузкой
полного набора промежуточных состояний.

## 8. Что делать дальше

Для минимального XUV MAPE:

1. Считать текущий `combined` frozen baseline для single-answer XUV:
   `19.12%` на historical primary subset.
2. Не оптимизировать следующий вариант по этим 118 labels.
3. Повторить generators с заранее зафиксированными 400 fixed epochs:
   fold 3 на 200 эпохах недообучен.
4. Сравнить одинаковый candidate budget: 10, 25 и 40 кандидатов.
5. Обучать physics-aware objective, но публиковать одновременно:
   top-1 XUV, He, logMsw, normalized joint distance и simultaneous hit.
6. Прогонять top-M через независимый forward simulator; simulator/ranker не
   должен видеть test truth.
7. До любого claim заморозить новый group-held-out/planet-held-out test и
   открыть его ровно один раз.

Быстрая post-run абляция показала, что простая замена MLP на
ExtraTrees/HistGradientBoosting с candidate-slot features не улучшает
producer-fold CV: лучший tree-вариант получил около `29.94%` XUV против
`28.11%` у frozen combined MLP. Добавление one-hot identity головы в listwise
MLP также не помогло (`30.77%` combined CV). Значит следующий прирост, вероятно,
потребует лучшего candidate generator или физически информативных признаков,
а не только другого универсального регрессора.

## 9. Финальный вердикт

Можно обоснованно говорить:

> На leakage-aware historical development pilot OOF-trained combined ranker
> уменьшил mean top-1 XUV MAPE с 23.71% до 19.12% на одном и том же пуле из
> 10 кандидатов.

Нельзя пока говорить:

> Доказано, что решение лучше статьи, оно даёт 7–11% MAPE для одного ответа
> или лучше
> ранжирует физически согласованный top-5 на новых данных.

Для статьи/защиты правильная формулировка — **улучшен deployable XUV top-1,
но oracle coverage и joint physical validity остаются отдельными метриками**.
