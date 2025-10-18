import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

def extract_keyword_features(text):
    """Извлечение ключевых слов и паттернов из текста"""
    
    text_lower = text.lower()
    
    features = {
        # Ключевые слова для каждого класса
        'licensable_keywords': 0,
        'platform_keywords': 0,
        'patentable_keywords': 0,
        'freetooperate_keywords': 0,
        
        # Технические термины
        'technical_terms': 0,
        'commercial_terms': 0,
        'scientific_terms': 0,
        
        # Структурные особенности
        'has_numbers': 0,
        'has_measurements': 0,
        'has_percentages': 0,
        'has_comparisons': 0,
        
        # Длина и сложность
        'text_length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
    }
    
    # Ключевые слова для licensable
    licensable_words = [
        'licensable', 'license', 'licensing', 'commercial', 'market', 'revenue',
        'profit', 'business', 'enterprise', 'industry', 'commercialization',
        'monetization', 'royalty', 'franchise', 'distribution'
    ]
    features['licensable_keywords'] = sum(1 for word in licensable_words if word in text_lower)
    
    # Ключевые слова для platform
    platform_words = [
        'platform', 'architecture', 'framework', 'infrastructure', 'system',
        'modular', 'scalable', 'extensible', 'api', 'sdk', 'integration',
        'ecosystem', 'foundation', 'base', 'core', 'engine'
    ]
    features['platform_keywords'] = sum(1 for word in platform_words if word in text_lower)
    
    # Ключевые слова для patentable
    patentable_words = [
        'patentable', 'patent', 'invention', 'novel', 'unique', 'proprietary',
        'exclusive', 'original', 'innovative', 'breakthrough', 'discovery',
        'method', 'process', 'technique', 'approach', 'solution'
    ]
    features['patentable_keywords'] = sum(1 for word in patentable_words if word in text_lower)
    
    # Ключевые слова для freetooperate
    freetooperate_words = [
        'free', 'open', 'public', 'available', 'accessible', 'unrestricted',
        'unlimited', 'unencumbered', 'clear', 'unobstructed', 'unimpeded',
        'unrestrained', 'unfettered', 'unconstrained'
    ]
    features['freetooperate_keywords'] = sum(1 for word in freetooperate_words if word in text_lower)
    
    # Технические термины
    technical_words = [
        'algorithm', 'optimization', 'efficiency', 'performance', 'processing',
        'computing', 'analysis', 'synthesis', 'engineering', 'design',
        'implementation', 'development', 'programming', 'coding', 'software'
    ]
    features['technical_terms'] = sum(1 for word in technical_words if word in text_lower)
    
    # Коммерческие термины
    commercial_words = [
        'market', 'customer', 'user', 'client', 'product', 'service',
        'sales', 'marketing', 'advertising', 'promotion', 'brand',
        'competition', 'competitive', 'advantage', 'value', 'benefit'
    ]
    features['commercial_terms'] = sum(1 for word in commercial_words if word in text_lower)
    
    # Научные термины
    scientific_words = [
        'research', 'study', 'experiment', 'trial', 'test', 'validation',
        'verification', 'hypothesis', 'theory', 'principle', 'concept',
        'discovery', 'finding', 'result', 'conclusion', 'evidence'
    ]
    features['scientific_terms'] = sum(1 for word in scientific_words if word in text_lower)
    
    # Структурные особенности
    features['has_numbers'] = int(bool(re.search(r'\d+', text)))
    features['has_measurements'] = int(bool(re.search(r'\d+\s*(mg|ml|kg|g|m|cm|mm|nm|μm|%)', text)))
    features['has_percentages'] = int('%' in text or 'percent' in text_lower)
    features['has_comparisons'] = int(any(word in text_lower for word in ['better', 'improved', 'enhanced', 'superior', 'advanced', 'increased', 'reduced']))
    
    return features

def extract_features_from_item(item):
    """Извлечение всех фичей из item"""
    
    # Базовые фичи
    features = {
        'evidence_count': len(item['evidence']),
        'is_negative': int(item['is_negative']),
        
        # Market features
        'max_importance_score': 0.0,
        'avg_importance_score': 0.0,
        'market_evidence_count': 0,
        'has_product_launch': 0,
        'has_ma': 0,
        'has_partnership': 0,
        'has_funding': 0,
        
        # Trial features
        'max_maturity_score': 0.0,
        'avg_maturity_score': 0.0,
        'trial_evidence_count': 0,
        'has_completed_trial': 0,
        'has_phase3_trial': 0,
        'has_terminated_trial': 0,
        
        # Other evidence counts
        'patent_count': 0,
        'paper_count': 0,
        'disclosure_count': 0,
    }
    
    # Добавляем ключевые слова
    keyword_features = extract_keyword_features(item['text'])
    features.update(keyword_features)
    
    # Анализируем evidence
    market_scores = []
    trial_scores = []
    
    for ev in item['evidence']:
        if ev['type'] == 'market' and 'importance_score' in ev['meta']:
            features['market_evidence_count'] += 1
            market_scores.append(ev['meta']['importance_score'])
            
            # Типы событий
            event_type = ev['meta']['type']
            if event_type == 'Product launch':
                features['has_product_launch'] = 1
            elif event_type == 'M&A':
                features['has_ma'] = 1
            elif event_type == 'Partnership':
                features['has_partnership'] = 1
            elif event_type == 'Funding':
                features['has_funding'] = 1
                
        elif ev['type'] == 'trial' and 'maturity_score' in ev['meta']:
            features['trial_evidence_count'] += 1
            trial_scores.append(ev['meta']['maturity_score'])
            
            # Статусы и фазы
            status = ev['meta']['status']
            phase = ev['meta']['phase']
            
            if status == 'Completed':
                features['has_completed_trial'] = 1
            if phase == 'Phase 3':
                features['has_phase3_trial'] = 1
            if status == 'Terminated':
                features['has_terminated_trial'] = 1
                
        elif ev['type'] == 'patent':
            features['patent_count'] += 1
        elif ev['type'] == 'paper':
            features['paper_count'] += 1
        elif ev['type'] == 'disclosure':
            features['disclosure_count'] += 1
    
    # Вычисляем агрегированные скоры
    if market_scores:
        features['max_importance_score'] = max(market_scores)
        features['avg_importance_score'] = np.mean(market_scores)
    if trial_scores:
        features['max_maturity_score'] = max(trial_scores)
        features['avg_maturity_score'] = np.mean(trial_scores)
    
    return features

def prepare_training_data():
    """Подготовка данных для обучения"""
    
    print("📊 Подготовка данных с ключевыми словами...")
    
    # Загружаем данные
    data = []
    with open('data/train_final.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Извлекаем фичи и лейблы
    X = []
    y = []
    texts = []
    
    for record in data:
        for item in record['items']:
            features = extract_features_from_item(item)
            X.append(features)
            y.append(item['label'])
            texts.append(item['text'])
    
    # Конвертируем в DataFrame
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    
    print(f"✅ Подготовлено {len(X_df)} образцов")
    print(f"✅ Количество фичей: {len(X_df.columns)}")
    print(f"✅ Распределение классов: {y_series.value_counts().to_dict()}")
    
    # Добавляем TF-IDF фичи
    print("🔤 Добавление TF-IDF фичей...")
    
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 3),
        stop_words='english',
        min_df=3,
        max_df=0.9
    )
    
    tfidf_features = tfidf.fit_transform(texts)
    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    )
    
    # Объединяем фичи
    X_combined = pd.concat([X_df, tfidf_df], axis=1)
    
    print(f"✅ Добавлено {tfidf_features.shape[1]} TF-IDF фичей")
    print(f"✅ Общее количество фичей: {X_combined.shape[1]}")
    
    return X_combined, y_series, tfidf

def train_models(X, y):
    """Обучение моделей с балансировкой классов"""
    
    print("\n🤖 Обучение моделей с балансировкой классов...")
    
    # Вычисляем веса классов
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    print(f"Веса классов: {class_weight_dict}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} образцов")
    print(f"Test set: {len(X_test)} образцов")
    
    # Модели с балансировкой классов
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=15,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=3000,
            C=0.1,
            class_weight='balanced'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nОбучение {name}...")
        
        # Обучение
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'y_test': y_test
        }
        
        print(f"Accuracy: {accuracy:.3f}")
    
    # Выбираем лучшую модель
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Лучшая модель: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.3f}")
    
    # Детальный отчет по лучшей модели
    print(f"\n📊 Детальный отчет по {best_model_name}:")
    print(classification_report(
        results[best_model_name]['y_test'], 
        results[best_model_name]['predictions']
    ))
    
    # Важность фичей
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Топ-20 важных фичей:")
        for _, row in feature_importance.head(20).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return best_model, X.columns, results

def save_model(model, feature_names, tfidf, model_name="technology_evaluator_keywords"):
    """Сохранение обученной модели"""
    
    # Сохраняем модель
    model_path = f"{model_name}.joblib"
    joblib.dump(model, model_path)
    
    # Сохраняем названия фичей
    features_path = f"{model_name}_features.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names.tolist(), f)
    
    # Сохраняем TF-IDF
    tfidf_path = f"{model_name}_tfidf.joblib"
    joblib.dump(tfidf, tfidf_path)
    
    print(f"\n💾 Модель сохранена:")
    print(f"  Модель: {model_path}")
    print(f"  Фичи: {features_path}")
    print(f"  TF-IDF: {tfidf_path}")

def main():
    """Основная функция"""
    
    print("🚀 ОБУЧЕНИЕ МОДЕЛИ С КЛЮЧЕВЫМИ СЛОВАМИ")
    print("=" * 60)
    
    # Подготовка данных
    X, y, tfidf = prepare_training_data()
    
    # Обучение моделей
    best_model, feature_names, results = train_models(X, y)
    
    # Сохранение модели
    save_model(best_model, feature_names, tfidf)
    
    print(f"\n✅ Обучение завершено!")

if __name__ == "__main__":
    main()
