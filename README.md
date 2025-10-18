# Technology Evaluator

REST API для оценки технологических идей на основе машинного обучения и правил. Система анализирует идеи и классифицирует их по 4 категориям: коммерческие, платформенные, свободные и нерекомендуемые.

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
```

### 2. Обучение модели (опционально)

```bash
# Обучение финальной модели
python train_final_model.py
```

### 3. Запуск API

```bash
# Запуск сервера
python api.py
```

API будет доступен по адресу: `http://localhost:5001`

### 4. Тестирование

```bash
# Автоматические тесты
python test_api.py

# Веб-интерфейс
open web_interface.html
```

## 📁 Структура проекта

```
dataset-gen/
├── api.py                          # REST API сервер
├── train_final_model.py            # Обучение модели
├── test_api.py                     # Тесты API
├── web_interface.html              # Веб-интерфейс
├── requirements.txt                # Зависимости
├── README.md                       # Документация
├── main.py                         # Генерация датасета
├── data/                           # Исходные данные
│   ├── train_final.jsonl          # Обучающий датасет
│   ├── patents.csv                # Патенты
│   ├── papers.csv                 # Научные статьи
│   ├── market_signals.csv         # Рыночные сигналы
│   ├── clinical_trials.csv        # Клинические испытания
│   ├── entities.csv               # Компании
│   ├── license_histories.csv      # Лицензионная история
│   └── internal_disclosures.csv   # Внутренние раскрытия
├── final_technology_evaluator.joblib      # Обученная модель
├── final_technology_evaluator_features.json  # Названия фичей
├── final_technology_evaluator_tfidf.joblib   # TF-IDF векторизатор
└── venv/                          # Виртуальное окружение
```

## 🎯 Классы предсказаний

- **commercial** - Коммерческая идея с потенциалом лицензирования
- **platform** - Платформенная идея для модульной разработки  
- **free** - Свободная идея для open source развития
- **none** - Не рекомендуется для развития

## 📊 Точность модели

- **Общая точность**: 52.3%
- **Commercial**: 70% recall
- **None**: 100% precision
- **Platform**: 17% recall
- **Free**: 13% recall

## 🌐 API Документация

### Эндпоинты

#### GET /health
Проверка здоровья API

#### GET /readiness  
Проверка готовности API

#### POST /evaluate
Оценка одной технологической идеи

**Запрос:**
```json
{
  "idea_text": "A novel machine learning algorithm for drug discovery",
  "evidence": [
    {
      "type": "market",
      "meta": {
        "type": "Product launch",
        "importance_score": 3.5
      }
    }
  ]
}
```

**Ответ:**
```json
{
  "idea_text": "A novel machine learning algorithm for drug discovery",
  "predictions": {
    "ml_model": {
      "prediction": "commercial",
      "confidence": 0.724,
      "probabilities": {
        "commercial": 0.724,
        "free": 0.123,
        "none": 0.089,
        "platform": 0.064
      }
    },
    "rule_based": {
      "prediction": "commercial"
    },
    "hybrid": {
      "prediction": "commercial",
      "method": "ml_high_confidence"
    }
  },
  "features": {
    "text_length": 45,
    "word_count": 8,
    "platform_score": 0,
    "commercial_score": 4,
    "free_score": 0,
    "technical_terms": 2,
    "evidence_count": 1
  },
  "recommendations": [
    {
      "type": "commercial_opportunity",
      "message": "Эта идея имеет коммерческий потенциал",
      "actions": [
        "Рассмотрите возможность лицензирования",
        "Проведите патентный поиск",
        "Оцените рыночные возможности"
      ]
    }
  ]
}
```

#### POST /batch_evaluate
Пакетная оценка нескольких идей

## 🔧 Использование

### Python

```python
import requests

response = requests.post('http://localhost:5001/evaluate', json={
    'idea_text': 'A modular platform architecture',
    'evidence': []
})

result = response.json()
print(f"Prediction: {result['predictions']['hybrid']['prediction']}")
```

### cURL

```bash
curl -X POST http://localhost:5001/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "idea_text": "A novel machine learning algorithm",
    "evidence": []
  }'
```

### JavaScript

```javascript
const response = await fetch('http://localhost:5001/evaluate', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        idea_text: 'An open source framework',
        evidence: []
    })
});

const result = await response.json();
console.log(result.predictions.hybrid.prediction);
```

## 🧠 Алгоритм работы

1. **Извлечение фичей**: Анализ текста на ключевые слова, технические термины, структурные особенности
2. **TF-IDF векторизация**: Преобразование текста в числовые векторы
3. **Ансамбль моделей**: Random Forest + Gradient Boosting + Logistic Regression
4. **Гибридный подход**: Комбинация ML модели и правил на основе ключевых слов
5. **Генерация рекомендаций**: Создание практических советов на основе предсказания

## 📈 Фичи модели

- **Текстовые фичи**: длина, количество слов, сложность
- **Ключевые слова**: скоринг по категориям (commercial, platform, free)
- **Технические термины**: алгоритмы, оптимизация, инженерия
- **Рыночные данные**: важность событий, типы сигналов
- **Клинические испытания**: зрелость, фаза, статус
- **TF-IDF**: 500 наиболее важных n-грамм

## 🛠️ Разработка

### Добавление новых фичей

1. Обновите функцию `extract_enhanced_features()` в `api.py`
2. Переобучите модель: `python train_final_model.py`
3. Перезапустите API: `python api.py`

### Мониторинг

API предоставляет эндпоинты для мониторинга:
- `/health` - проверка здоровья
- `/readiness` - проверка готовности

## 📄 Лицензия

MIT License

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Внесите изменения
4. Создайте Pull Request

## 📞 Поддержка

Для вопросов и предложений создайте Issue в репозитории.
