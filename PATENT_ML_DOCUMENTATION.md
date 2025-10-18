# ML-модель для патентных рекомендаций

## Обзор

Добавлена новая ML-модель для принятия решений о патентах на основе анализа технологических идей. Модель комбинирует машинное обучение и AI-анализ для предоставления комплексных рекомендаций по патентованию.

## Функциональность

### 1. Новый эндпоинт `/patent_recommendation`

**POST** `/patent_recommendation`

Анализирует идею и предоставляет детальные рекомендации по патентам.

**Запрос:**
```json
{
    "idea_text": "A novel machine learning algorithm for drug discovery",
    "evidence": [
        {"type": "patent", "meta": {"title": "ML Patent"}},
        {"type": "paper", "meta": {"title": "Research Paper"}}
    ]
}
```

**Ответ:**
```json
{
    "patent_recommendation": {
        "ml_model": {
            "recommendation": "Strong",
            "confidence": 0.78,
            "patent_score": 7.8,
            "reasons": ["High patent-related terminology detected"],
            "features": {
                "patent_keyword_score": 9,
                "technical_innovation_score": 4,
                "complexity_score": 0,
                "patent_evidence_count": 1,
                "research_evidence_count": 1
            }
        },
        "ai_analysis": {
            "recommendation": "Strong",
            "confidence": "High",
            "reasons": ["Novel algorithm for drug discovery"]
        },
        "combined": {
            "recommendation": "Strong",
            "confidence": 0.64,
            "combined_score": 4.0,
            "ml_score": 4,
            "ai_score": 4,
            "all_reasons": ["High patent-related terminology detected", "Novel algorithm for drug discovery"],
            "decision_factors": {
                "ml_weight": 0.7,
                "ai_weight": 0.3,
                "agreement": true
            }
        }
    }
}
```

### 2. Обновленный эндпоинт `/evaluate`

Основной эндпоинт теперь включает раздел `patent_analysis` с детальной информацией о патентных рекомендациях.

### 3. Веб-интерфейс

Обновленный веб-интерфейс отображает:
- ML рекомендации по патентам
- AI рекомендации по патентам  
- Комбинированное решение
- Детали патентного скора
- Анализ фичей ML модели
- Причины рекомендаций

## ML-модель

### Извлекаемые фичи

1. **Патентные ключевые слова** (веса 1-3):
   - `novel`, `innovative`, `breakthrough` (вес 3)
   - `algorithm`, `method`, `device` (вес 2)
   - `software`, `application` (вес 1)

2. **Технические индикаторы** (веса 1-2):
   - `machine learning`, `artificial intelligence` (вес 2)
   - `blockchain`, `quantum`, `biotechnology` (вес 2)
   - `iot`, `wireless` (вес 1)

3. **Индикаторы сложности** (веса 1-3):
   - `cutting-edge`, `state-of-the-art` (вес 3)
   - `complex`, `sophisticated` (вес 2)
   - `optimized`, `efficient` (вес 1)

4. **Доказательства**:
   - Патенты (вес 2)
   - Исследования (вес 1.5)
   - Рыночные сигналы (вес 1)

### Алгоритм принятия решений

1. **Подсчет патентного скора** (0-10):
   ```
   patent_score = patent_keywords * 0.3 + technical_indicators * 0.4 + 
                  complexity * 0.2 + evidence_bonuses
   ```

2. **Классификация**:
   - ≥7: Strong (высокая рекомендация)
   - ≥5: Moderate (средняя рекомендация)
   - ≥3: Weak (слабая рекомендация)
   - <3: Not Recommended (не рекомендуется)

3. **Комбинирование с AI**:
   - ML вес: 70%
   - AI вес: 30%
   - Взвешенное среднее для финального решения

## Примеры использования

### Высокий патентный потенциал
```
Идея: "A novel machine learning algorithm for drug discovery using quantum computing"
Результат: Strong (9.6/10)
Причины: Высокая патентная терминология, технические индикаторы, доказательства
```

### Средний патентный потенциал
```
Идея: "An innovative software platform for data analysis with modular architecture"
Результат: Moderate (3.0/10)
Причины: Смешанные сигналы ML и AI
```

### Низкий патентный потенциал
```
Идея: "A simple website for online shopping"
Результат: Not Recommended (0.0/10)
Причины: Отсутствие патентных индикаторов
```

## Тестирование

Запустите тесты:
```bash
python test_patent_ml.py
```

Тесты проверяют:
- Различные уровни патентного потенциала
- Комбинирование ML и AI рекомендаций
- Интеграцию с основным эндпоинтом
- Корректность веб-интерфейса

## API Endpoints

- `GET /readiness` - проверка готовности
- `POST /evaluate` - основная оценка с патентным анализом
- `POST /patent_recommendation` - специализированные патентные рекомендации
- `POST /batch_evaluate` - пакетная оценка

## Технические детали

- **Язык**: Python 3.13
- **Фреймворк**: Flask
- **ML библиотеки**: scikit-learn, pandas, numpy
- **AI интеграция**: OpenAI GPT-3.5-turbo
- **Формат данных**: JSON
- **Кодировка**: UTF-8
