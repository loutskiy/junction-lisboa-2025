# 🚀 Система оценки технологических идей

## 📋 Описание

Данная система предназначена для оценки технологических идей на основе рыночных сигналов, клинических испытаний и других доказательств. Модель использует обогащенные фичи для принятия решений о коммерческой значимости и готовности технологий.

## 🎯 Целевые задачи

1. **Оценка коммерческой значимости** - анализ рыночных сигналов
2. **Оценка готовности технологий** - анализ клинических испытаний  
3. **Взвешивание доказательств** - учет надежности источников
4. **Классификация идей** - определение типа технологической возможности

## 📊 Структура датасета

### Основные компоненты:
- **2,400 уникальных seed документов** (технологические идеи)
- **14,428 items** (конкретные утверждения)
- **24,163 evidence items** с обогащенными фичами

### Типы evidence:
- **Market signals** (4,815) - рыночные сигналы
- **Clinical trials** (4,886) - клинические испытания
- **Patents** (4,778) - патенты
- **Papers** (4,769) - научные статьи
- **Disclosures** (4,915) - внутренние раскрытия

## 🔬 Ключевые фичи

### 1. Рыночные сигналы (Market Signals)
```python
{
    "importance_score": 2.74,      # Общая важность (0-4)
    "type_score": 4.0,             # Тип события (Product launch=4.0)
    "source_score": 3.0,           # Надежность источника (GlobalDataSim=3.0)
    "type": "Product launch",      # Тип события
    "source": "GlobalDataSim"      # Источник данных
}
```

**Шкала важности:**
- Product launch: 4.0 (максимальная важность)
- M&A: 3.5
- Partnership: 3.0
- Funding: 2.5
- Market report: 2.0

**Надежность источников:**
- GlobalDataSim: 3.0
- DealScope: 2.8
- EvaluateSim: 2.5
- CrunchSim: 2.2
- VentureWatch: 2.0

### 2. Клинические испытания (Clinical Trials)
```python
{
    "maturity_score": 1.68,        # Зрелость технологии (0-3)
    "status": "Completed",         # Статус испытания
    "phase": "Phase 3",           # Фаза испытания
    "status_score": 3.0,          # Нормализованный статус
    "phase_score": 3.0            # Нормализованная фаза
}
```

**Шкала зрелости:**
- Completed + Phase 3: 3.0 (максимальная зрелость)
- Active + Phase 2: 2.5
- Recruiting + Phase 1: 2.0
- Terminated: 0.0 (низкая зрелость)

### 3. Целевые классы
- **licensable** (3,071) - высокая коммерческая ценность
- **platform** (3,065) - технологическая платформа
- **patentable** (2,954) - патентоспособность
- **freetooperate_flag** (2,938) - свобода операций
- **none** (2,400) - отрицательные примеры

## 🛠️ Использование

### Пример оценки идеи:
```python
from evaluate_idea_example import evaluate_technology_idea

# Входные данные
idea_text = "platform architecture with modular cartridges for graphene membranes"
evidence_list = [
    {
        "type": "market",
        "meta": {
            "importance_score": 3.5,
            "type": "Product launch",
            "source": "GlobalDataSim"
        }
    },
    {
        "type": "trial", 
        "meta": {
            "maturity_score": 2.5,
            "status": "Completed",
            "phase": "Phase 3"
        }
    }
]

# Оценка
result = evaluate_technology_idea(idea_text, evidence_list)
print(f"Общий скор: {result['overall_score']:.2f}/4.0")
```

### Выходные данные:
```python
{
    "scores": {
        "commercial_importance": 3.5,    # Коммерческая важность
        "technology_maturity": 2.5,      # Зрелость технологий
        "evidence_strength": 0.8,        # Сила доказательств
        "market_activity": 1.0,          # Активность рынка
        "clinical_progress": 1.0         # Прогресс в испытаниях
    },
    "overall_score": 2.85,              # Общий скор (0-4)
    "recommendations": [                # Рекомендации
        "✅ Высокая коммерческая привлекательность",
        "✅ Технология готова к коммерциализации"
    ]
}
```

## 📈 Статистика датасета

### Распределение фичей:
- **40.1% evidence items** имеют обогащенные фичи
- **9,701 evidence items** с дополнительными метриками
- **Средний importance_score**: 2.74
- **Средний maturity_score**: 1.68

### Корреляция с лейблами:
- **licensable**: 33.96% с market evidence, 34.42% с trial evidence
- **platform**: 34.81% с market evidence, 35.99% с trial evidence
- **patentable**: 35.00% с market evidence, 35.95% с trial evidence

## 🎯 Рекомендации для модели

### Приоритетные фичи:
1. **importance_score** - рыночная важность
2. **maturity_score** - зрелость технологий
3. **type_score** - тип события
4. **source_score** - надежность источника
5. **evidence_count** - количество доказательств

### Веса для итоговой оценки:
- commercial_importance: 30%
- technology_maturity: 30%
- evidence_strength: 20%
- market_activity: 10%
- clinical_progress: 10%

## 📁 Файлы

- `data/train_final.jsonl` - основной датасет
- `main.py` - генерация датасета с фичами
- `analyze_features.py` - анализ фичей
- `evaluate_idea_example.py` - пример оценки идеи
- `README_EVALUATION.md` - данная документация

## 🚀 Следующие шаги

1. **Обучение модели** на подготовленном датасете
2. **Валидация** на тестовых данных
3. **Интеграция** в систему оценки идей
4. **Мониторинг** качества предсказаний
5. **Итеративное улучшение** на основе обратной связи
