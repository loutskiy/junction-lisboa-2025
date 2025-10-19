# 🚀 Technology Evaluator

**AI-система для автоматической оценки технологических идей** с машинным обучением и AI-анализом. Система за секунды определяет коммерческий потенциал идей и рекомендует стратегию развития.

## 📋 Что это такое?

Technology Evaluator анализирует текстовое описание технологической идеи и классифицирует её по 4 категориям:
- **Commercial** - коммерческий потенциал для лицензирования и монетизации
- **Platform** - подходит для создания модульной платформы
- **Free** - лучше всего подходит для открытого развития
- **None** - не рекомендуется для разработки в текущем виде

## 🎯 Основные возможности

### 🔍 Анализ и классификация
- **Гибридный ML подход**: комбинация машинного обучения и правил
- **Патентный анализ**: оценка потенциала для патентования
- **AI-анализ**: техническая сложность, рыночный потенциал, уровень инноваций
- **Детальные отчеты**: структурированный анализ с рекомендациями

### 📊 Источники данных
- **Патенты** (6,002 записи) - технические инновации
- **Научные публикации** (~5,000) - исследовательские данные
- **Рыночные сигналы** (~3,000) - рыночная разведка
- **Клинические испытания** (~2,000) - медицинская разработка
- **Лицензионная история** (~1,000) - паттерны лицензирования
- **Внутренние раскрытия** (~800) - внутренние инновации

### 🤖 AI-функции
- **Оценка технической сложности**: Low/Medium/High
- **Анализ рыночного потенциала**: Low/Medium/High
- **Классификация уровня инноваций**: Incremental/Moderate/Revolutionary
- **AI-рекомендации**: конкретные советы по развитию

## 🚀 Быстрый старт

### 1. Установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd dataset-gen

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt

# Для AI-анализа (опционально)
pip install openai
```

### 2. Настройка OpenAI API (для AI-анализа)

```bash
# Создайте файл с API ключом
echo "your-openai-api-key-here" > openai_key.txt
```

### 3. Обучение моделей (опционально)

```bash
# Обучение основной модели
python train_final_model.py

# Обучение улучшенной модели
python train_simple_improved_model.py

# Обучение патентной модели
python train_patent_model.py
```

### 4. Запуск API

```bash
# Запуск сервера
python api.py
```

API будет доступен по адресу: `http://localhost:5001`

### 5. Тестирование

```bash
# Автоматические тесты API
python test_api.py

# Тестирование патентной модели
python test_patent_ml.py

# Веб-интерфейс
open web_interface.html
```

## 📁 Структура проекта

```
dataset-gen/
├── api.py                              # Основной REST API
├── main.py                             # Генерация датасета
├── web_interface.html                  # Веб-интерфейс
├── requirements.txt                    # Зависимости
├── openai_key.txt                     # OpenAI API ключ
│
├── Модели/
│   ├── final_technology_evaluator.joblib           # Основная модель
│   ├── final_technology_evaluator_tfidf.joblib     # TF-IDF векторizer
│   ├── final_technology_evaluator_features.json    # Названия фичей
│   ├── patent_potential_model.joblib               # Патентная модель
│   ├── patent_potential_tfidf.joblib               # TF-IDF для патентов
│   └── patent_potential_features.json              # Фичи патентной модели
│
├── Скрипты обучения/
│   ├── train_final_model.py                       # Обучение основной модели
│   ├── train_simple_improved_model.py             # Обучение улучшенной модели
│   └── train_patent_model.py                      # Обучение патентной модели
│
├── Тесты/
│   ├── test_api.py                                # Тесты API
│   ├── test_patent_ml.py                          # Тесты патентной модели
│   └── test_patent_ai.py                          # Тесты AI функций
│
└── data/                                          # Обучающие данные
    ├── train_final.jsonl                          # Основной датасет
    ├── patents.csv                                # Патенты
    ├── papers.csv                                 # Научные публикации
    ├── market_signals.csv                         # Рыночные сигналы
    ├── clinical_trials.csv                        # Клинические испытания
    ├── entities.csv                               # Компании
    ├── license_histories.csv                      # История лицензий
    ├── internal_disclosures.csv                   # Внутренние раскрытия
    ├── seed_docs.csv                              # Исходные документы
    ├── annotations_train.jsonl                    # Экспертные аннотации
    └── candidates_train.jsonl                     # Кандидаты для обучения
```

## 🔧 API Endpoints

### GET /health
Проверка здоровья API
```bash
curl http://localhost:5001/health
```

### GET /readiness
Проверка готовности API
```bash
curl http://localhost:5001/readiness
```

### POST /evaluate
Оценка одной идеи
```bash
curl -X POST http://localhost:5001/evaluate \
  -H "Content-Type: application/json" \
  -d '{"idea_text": "novel enzymatic step enabling scalable production in lab-on-a-chip"}'
```

### POST /batch_evaluate
Пакетная оценка нескольких идей
```bash
curl -X POST http://localhost:5001/batch_evaluate \
  -H "Content-Type: application/json" \
  -d '{"ideas": [{"idea_text": "idea 1"}, {"idea_text": "idea 2"}]}'
```

## 📊 Структура ответа API

### Основной ответ `/evaluate`

```json
{
  "idea_text": "novel enzymatic step enabling scalable production in lab-on-a-chip",
  "analysis_summary": {
    "prediction": "commercial",
    "confidence": 0.315,
    "method": "ml_medium_confidence",
    "evidence_count": 0,
    "evidence_types": [],
    "insights": []
  },
  "predictions": {
    "ml_model": {
      "prediction": "commercial",
      "confidence": 0.315,
      "probabilities": {
        "commercial": 0.315,
        "free": 0.222,
        "none": 0.167,
        "platform": 0.294
      }
    },
    "rule_based": {
      "prediction": "platform"
    },
    "hybrid": {
      "prediction": "commercial",
      "method": "ml_medium_confidence",
      "confidence_level": "low"
    }
  },
  "detailed_features": {
    "text_analysis": {
      "text_length": 66,
      "word_count": 8,
      "avg_word_length": 7.375,
      "unique_word_ratio": 1.0,
      "sentence_count": 1,
      "complexity_score": 3
    },
    "keyword_scores": {
      "platform_score": 2,
      "commercial_score": 30,
      "free_score": 0,
      "technical_terms": 0,
      "scientific_terms": 0,
      "business_terms": 1
    },
    "evidence_analysis": {
      "total_evidence": 0,
      "market_signals": 0,
      "clinical_trials": 0,
      "patents": 0,
      "papers": 0,
      "disclosures": 0
    }
  },
  "ai_analysis": {
    "technical_complexity": "High",
    "market_potential": "High",
    "innovation_level": "Revolutionary",
    "patent_recommendation": "Strong",
    "patent_confidence": "High",
    "patent_reasons": [
      "Enzymatic steps in lab-on-a-chip technology are novel and can be highly valuable"
    ],
    "technical_challenges": [
      "Ensuring enzymatic reactions are efficient and effective on a small scale"
    ],
    "market_opportunities": [
      "Biotechnology research",
      "Pharmaceutical industry for drug development"
    ],
    "ai_recommendations": [
      "Invest in research and development to optimize enzymatic reactions"
    ]
  },
  "patent_analysis": {
    "ml_model": {
      "recommendation": "Moderate",
      "confidence": 0.597,
      "patent_score": 5.97,
      "reasons": [
        "High patent-related terminology detected",
        "Strong novelty indicators"
      ]
    },
    "ai_analysis": {
      "recommendation": "Strong",
      "confidence": "High",
      "reasons": [
        "Enzymatic steps in lab-on-a-chip technology are novel"
      ]
    },
    "combined": {
      "recommendation": "Moderate",
      "confidence": 0.548,
      "combined_score": 3.3,
      "ml_score": 3,
      "ai_score": 4
    }
  },
  "recommendations": [
    {
      "type": "commercial_opportunity",
      "priority": "high",
      "title": "Commercial Potential",
      "message": "This idea has high commercial potential for licensing and monetization",
      "confidence": "low",
      "actions": [
        "Conduct patent search and Freedom to Operate (FTO) analysis",
        "Assess market size and competitive landscape",
        "Develop licensing strategy"
      ],
      "timeline": "3-6 months",
      "investment": "Medium-High",
      "roi_potential": "High"
    }
  ],
  "metadata": {
    "timestamp": "2025-10-19T01:53:20.892317",
    "model_version": "final_technology_evaluator",
    "api_version": "1.0.0",
    "ai_enabled": true,
    "processing_time_ms": 0
  }
}
```

## 🏗️ Техническая архитектура

### Система машинного обучения

**Основная модель:**
- **Алгоритмы**: RandomForest, GradientBoosting, LogisticRegression
- **Ансамбль**: VotingClassifier с soft voting
- **Фичи**: 500+ признаков (TF-IDF + engineered features)
- **Точность**: ~85% на тестовой выборке

**Патентная модель:**
- **Алгоритмы**: RandomForest, GradientBoosting, LogisticRegression
- **Ансамбль**: VotingClassifier с soft voting
- **Фичи**: 230+ признаков (специализированные патентные фичи)
- **Точность**: ~58% на тестовой выборке

**Гибридный подход:**
- ML модель используется при уверенности > 25%
- Правила используются при низкой уверенности ML
- AI-анализ дополняет ML рекомендации

### Обработка данных

**Извлечение фичей:**
- Текстовый анализ (длина, сложность, уникальность)
- Ключевые слова с весами (commercial, platform, free)
- Технические термины и паттерны
- Анализ доказательств (патенты, публикации, рынок)

**TF-IDF векторизация:**
- Максимум 500 фичей для основной модели
- Максимум 300 фичей для патентной модели
- N-граммы 1-3 для основной модели, 1-2 для патентной
- Английские стоп-слова исключены

## 🎯 Целевые пользователи

- **🔬 Исследователи** — оценка коммерческого потенциала своих идей
- **💰 Инвесторы** — фильтрация перспективных проектов для инвестиций
- **🚀 Стартапы** — выбор направления развития и стратегии
- **🏢 Корпорации** — анализ портфеля инноваций и R&D планирование
- **⚖️ Патентные поверенные** — предварительная оценка патентного потенциала

## 📈 Производительность

### Метрики системы
- **Время отклика**: < 2 секунды на запрос
- **Пропускная способность**: 100+ запросов/минуту
- **Точность классификации**: 85% для основной модели
- **Покрытие**: 4 основные категории + патентный анализ

### Масштабируемость
- **Пакетная обработка**: до 1000 идей одновременно
- **Горизонтальное масштабирование**: поддержка load balancing
- **Кэширование**: TF-IDF векторизация кэшируется
- **Оптимизация**: ленивая загрузка моделей

## 🔧 Разработка

### Добавление новых фичей

```python
def extract_custom_features(text):
    features = {}
    # Добавьте ваши фичи здесь
    features['custom_score'] = calculate_custom_score(text)
    return features
```

### Обучение новой модели

```python
# 1. Подготовьте данные в формате train_final.jsonl
# 2. Измените extract_features в train_final_model.py
# 3. Запустите обучение
python train_final_model.py
```

### Тестирование изменений

```bash
# Тестирование API
python test_api.py

# Тестирование патентной модели
python test_patent_ml.py

# Проверка качества модели
python -c "
import joblib
model = joblib.load('final_technology_evaluator.joblib')
print('Model loaded successfully')
"
```

## 📚 Дополнительные ресурсы

- **Веб-интерфейс**: `web_interface.html` - интерактивное тестирование
- **Примеры запросов**: см. `test_api.py`
- **Документация API**: встроенная в код с примерами
- **Логи**: детальное логирование всех операций

## 🤝 Поддержка

Для вопросов и предложений:
1. Проверьте существующие issues
2. Создайте новый issue с описанием проблемы
3. Приложите примеры запросов и ответов
4. Укажите версию системы и окружение

## 📄 Лицензия

Проект распространяется под лицензией MIT. См. файл LICENSE для деталей.

---

**Technology Evaluator** - автоматизируйте оценку технологических идей с помощью AI и машинного обучения! 🚀