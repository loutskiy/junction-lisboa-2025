import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

def rule_based_classifier(text):
    """Правила на основе ключевых слов"""
    
    text_lower = text.lower()
    
    # Счетчики для каждого класса
    platform_score = 0
    commercial_score = 0
    free_score = 0
    
    # Ключевые слова для platform с весами
    platform_keywords = {
        'platform': 3, 'architecture': 2, 'framework': 2, 'infrastructure': 2, 'system': 1,
        'modular': 2, 'scalable': 2, 'extensible': 2, 'api': 2, 'sdk': 2,
        'integration': 1, 'ecosystem': 2, 'foundation': 1, 'base': 1, 'core': 1,
        'cartridge': 2, 'modular cartridges': 4, 'platform architecture': 4,
        'engine': 1, 'interface': 1, 'component': 1, 'module': 1, 'service': 1
    }
    platform_score = sum(weight for word, weight in platform_keywords.items() if word in text_lower)
    
    # Ключевые слова для commercial с весами
    commercial_keywords = {
        'licensable': 4, 'license': 3, 'licensing': 3, 'commercial': 2, 'market': 1,
        'revenue': 2, 'profit': 2, 'business': 1, 'enterprise': 1, 'industry': 1,
        'commercialization': 4, 'monetization': 3, 'patentable': 4, 'patent': 3,
        'invention': 2, 'novel': 2, 'unique': 1, 'proprietary': 3, 'exclusive': 3,
        'original': 1, 'innovative': 2, 'breakthrough': 3, 'discovery': 2,
        'method': 1, 'process': 1, 'technique': 1, 'approach': 1, 'solution': 1,
        'valuable': 2, 'profitable': 2, 'marketable': 2, 'economic': 1
    }
    commercial_score = sum(weight for word, weight in commercial_keywords.items() if word in text_lower)
    
    # Ключевые слова для free с весами
    free_keywords = {
        'free': 3, 'open': 2, 'public': 1, 'available': 1, 'accessible': 1,
        'unrestricted': 3, 'unlimited': 3, 'unencumbered': 3, 'clear': 1,
        'unobstructed': 3, 'unimpeded': 3, 'unrestrained': 3, 'unfettered': 3,
        'freetooperate': 4, 'free to operate': 4, 'freedom to operate': 4,
        'unconstrained': 3, 'unlimited use': 3, 'open source': 3, 'public domain': 3
    }
    free_score = sum(weight for word, weight in free_keywords.items() if word in text_lower)
    
    # Определяем класс на основе максимального скора
    scores = {
        'platform': platform_score,
        'commercial': commercial_score,
        'free': free_score
    }
    
    # Если все скоры равны 0, используем эвристику
    if max(scores.values()) == 0:
        if any(word in text_lower for word in ['technology', 'innovation', 'development', 'research']):
            return 'commercial'
        elif any(word in text_lower for word in ['system', 'method', 'approach']):
            return 'platform'
        else:
            return 'commercial'
    
    return max(scores, key=scores.get)

def extract_enhanced_features(text):
    """Извлечение фичей"""
    
    text_lower = text.lower()
    
    features = {
        # Базовые характеристики
        'text_length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
        'unique_word_ratio': len(set(text.split())) / len(text.split()) if text.split() else 0,
        
        # Скоринг по ключевым словам
        'platform_score': 0,
        'commercial_score': 0,
        'free_score': 0,
        
        # Технические термины
        'technical_terms': 0,
        'scientific_terms': 0,
        'business_terms': 0,
        'engineering_terms': 0,
        
        # Структурные особенности
        'has_numbers': int(bool(re.search(r'\d+', text))),
        'has_measurements': int(bool(re.search(r'\d+\s*(mg|ml|kg|g|m|cm|mm|nm|μm|%)', text))),
        'has_percentages': int('%' in text or 'percent' in text_lower),
        'has_comparisons': int(any(word in text_lower for word in ['better', 'improved', 'enhanced', 'superior', 'advanced', 'increased', 'reduced'])),
        'has_quantifiers': int(any(word in text_lower for word in ['high', 'low', 'large', 'small', 'big', 'tiny', 'massive', 'minimal'])),
        
        # Специфичные паттерны
        'has_platform_patterns': 0,
        'has_commercial_patterns': 0,
        'has_free_patterns': 0,
        
        # Сложность текста
        'complexity_score': 0,
        'technical_density': 0,
        
        # Новые фичи для улучшения качества
        'sentence_count': len([s for s in text.split('.') if s.strip()]),
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!'),
        'has_capitals': int(any(c.isupper() for c in text)),
        'has_parentheses': int('(' in text and ')' in text),
        'has_quotes': int('"' in text or "'" in text),
        'has_dashes': int('-' in text),
        'has_colons': int(':' in text),
        'has_semicolons': int(';' in text),
    }
    
    # Подсчет ключевых слов с весами
    platform_keywords = {
        'platform': 3, 'architecture': 2, 'framework': 2, 'infrastructure': 2, 'system': 1,
        'modular': 2, 'scalable': 2, 'extensible': 2, 'api': 2, 'sdk': 2,
        'integration': 1, 'ecosystem': 2, 'foundation': 1, 'base': 1, 'core': 1,
        'cartridge': 2, 'modular cartridges': 4, 'platform architecture': 4,
        'engine': 1, 'interface': 1, 'component': 1, 'module': 1, 'service': 1
    }
    features['platform_score'] = sum(weight for word, weight in platform_keywords.items() if word in text_lower)
    
    commercial_keywords = {
        'licensable': 4, 'license': 3, 'licensing': 3, 'commercial': 2, 'market': 1,
        'revenue': 2, 'profit': 2, 'business': 1, 'enterprise': 1, 'industry': 1,
        'commercialization': 4, 'monetization': 3, 'patentable': 4, 'patent': 3,
        'invention': 2, 'novel': 2, 'unique': 1, 'proprietary': 3, 'exclusive': 3,
        'original': 1, 'innovative': 2, 'breakthrough': 3, 'discovery': 2,
        'method': 1, 'process': 1, 'technique': 1, 'approach': 1, 'solution': 1,
        'valuable': 2, 'profitable': 2, 'marketable': 2, 'economic': 1
    }
    features['commercial_score'] = sum(weight for word, weight in commercial_keywords.items() if word in text_lower)
    
    free_keywords = {
        'free': 3, 'open': 2, 'public': 1, 'available': 1, 'accessible': 1,
        'unrestricted': 3, 'unlimited': 3, 'unencumbered': 3, 'clear': 1,
        'unobstructed': 3, 'unimpeded': 3, 'unrestrained': 3, 'unfettered': 3,
        'freetooperate': 4, 'free to operate': 4, 'freedom to operate': 4,
        'unconstrained': 3, 'unlimited use': 3, 'open source': 3, 'public domain': 3
    }
    features['free_score'] = sum(weight for word, weight in free_keywords.items() if word in text_lower)
    
    # Технические термины
    technical_words = {
        'algorithm': 2, 'optimization': 2, 'efficiency': 1, 'performance': 1, 'processing': 1,
        'computing': 1, 'analysis': 1, 'synthesis': 1, 'engineering': 1, 'design': 1,
        'implementation': 1, 'development': 1, 'programming': 1, 'coding': 1, 'software': 1,
        'hardware': 1, 'device': 1, 'apparatus': 1, 'machine': 1, 'tool': 1
    }
    features['technical_terms'] = sum(weight for word, weight in technical_words.items() if word in text_lower)
    
    # Научные термины
    scientific_words = {
        'research': 1, 'study': 1, 'experiment': 1, 'trial': 1, 'test': 1, 'validation': 1,
        'verification': 1, 'hypothesis': 1, 'theory': 1, 'principle': 1, 'concept': 1,
        'discovery': 1, 'finding': 1, 'result': 1, 'conclusion': 1, 'evidence': 1,
        'data': 1, 'analysis': 1, 'statistics': 1, 'measurement': 1, 'observation': 1
    }
    features['scientific_terms'] = sum(weight for word, weight in scientific_words.items() if word in text_lower)
    
    # Бизнес термины
    business_words = {
        'market': 1, 'customer': 1, 'user': 1, 'client': 1, 'product': 1, 'service': 1,
        'sales': 1, 'marketing': 1, 'advertising': 1, 'promotion': 1, 'brand': 1,
        'competition': 1, 'competitive': 1, 'advantage': 1, 'value': 1, 'benefit': 1,
        'cost': 1, 'price': 1, 'investment': 1, 'return': 1, 'roi': 1
    }
    features['business_terms'] = sum(weight for word, weight in business_words.items() if word in text_lower)
    
    # Инженерные термины
    engineering_words = {
        'engineering': 1, 'design': 1, 'construction': 1, 'manufacturing': 1, 'production': 1,
        'assembly': 1, 'fabrication': 1, 'building': 1, 'creating': 1, 'making': 1,
        'developing': 1, 'constructing': 1, 'building': 1, 'creating': 1
    }
    features['engineering_terms'] = sum(weight for word, weight in engineering_words.items() if word in text_lower)
    
    # Специфичные паттерны
    features['has_platform_patterns'] = int(any(phrase in text_lower for phrase in [
        'platform architecture', 'modular cartridges', 'scalable system', 'extensible framework',
        'api platform', 'sdk framework', 'integration platform'
    ]))
    
    features['has_commercial_patterns'] = int(any(phrase in text_lower for phrase in [
        'commercial value', 'market opportunity', 'revenue potential', 'business model',
        'licensing opportunity', 'patent application', 'commercialization strategy'
    ]))
    
    features['has_free_patterns'] = int(any(phrase in text_lower for phrase in [
        'free to operate', 'freedom to operate', 'unrestricted use', 'open source',
        'public domain', 'unlimited access', 'clear path'
    ]))
    
    # Сложность текста
    features['complexity_score'] = len([w for w in text.split() if len(w) > 8])
    features['technical_density'] = features['technical_terms'] / max(features['word_count'], 1)
    
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
    
    # Добавляем улучшенные фичи
    enhanced_features = extract_enhanced_features(item['text'])
    features.update(enhanced_features)
    
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
    """Подготовка данных с упрощенными классами"""
    
    print("📊 Подготовка данных с упрощенными классами...")
    
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
            
            # Упрощаем классы
            original_label = item['label']
            if original_label in ['licensable', 'patentable']:
                simplified_label = 'commercial'
            elif original_label == 'platform':
                simplified_label = 'platform'
            elif original_label == 'freetooperate_flag':
                simplified_label = 'free'
            else:  # none
                simplified_label = 'none'
            
            y.append(simplified_label)
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
        min_df=2,
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

def train_final_model(X, y, tfidf):
    """Обучение финальной модели"""
    
    print("\n🤖 Обучение финальной модели...")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} образцов")
    print(f"Test set: {len(X_test)} образцов")
    
    # Финальные модели
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        random_state=42
    )
    
    lr = LogisticRegression(
        random_state=42,
        max_iter=3000,
        C=0.1,
        class_weight='balanced'
    )
    
    # Ансамбль с весами
    ensemble = VotingClassifier([
        ('rf', rf),
        ('gb', gb),
        ('lr', lr)
    ], voting='soft')
    
    # Обучение
    print("Обучение ансамбля...")
    ensemble.fit(X_train, y_train)
    
    # Предсказания
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Ensemble Accuracy: {accuracy:.3f}")
    
    # Детальный отчет
    print(f"\n📊 Детальный отчет по ансамблю:")
    print(classification_report(y_test, y_pred))
    
    return ensemble, accuracy

def save_final_model(model, feature_names, tfidf, model_name="final_technology_evaluator"):
    """Сохранение финальной модели"""
    
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
    
    print(f"\n💾 Финальная модель сохранена:")
    print(f"  Модель: {model_path}")
    print(f"  Фичи: {features_path}")
    print(f"  TF-IDF: {tfidf_path}")

def main():
    """Основная функция"""
    
    print("🚀 ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
    print("=" * 70)
    
    # Подготовка данных
    X, y, tfidf = prepare_training_data()
    
    # Обучение финальной модели
    ensemble_model, accuracy = train_final_model(X, y, tfidf)
    
    # Сохранение модели
    save_final_model(ensemble_model, X.columns, tfidf)
    
    print(f"\n✅ Обучение завершено!")
    print(f"🎯 Достигнутая точность: {accuracy:.1%}")

if __name__ == "__main__":
    main()
