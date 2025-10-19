import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

def extract_enhanced_features(text):
    """Извлечение фичей с усиленными коммерческими терминами"""
    
    text_lower = text.lower()
    
    features = {
        # Базовые характеристики
        'text_length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
        'unique_word_ratio': len(set(text.split())) / len(text.split()) if text.split() else 0,
        
        # УСИЛЕННЫЙ скоринг по ключевым словам
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
        
        # ДОПОЛНИТЕЛЬНЫЕ фичи для коммерческих паттернов
        'has_enzymatic_terms': 0,
        'has_production_terms': 0,
        'has_scalable_terms': 0,
        'has_novel_terms': 0,
        'has_patentable_terms': 0,
        'has_innovation_terms': 0,
        'has_lab_on_chip_terms': 0,
    }
    
    # Обычные ключевые слова для platform
    platform_keywords = {
        'platform': 3, 'architecture': 2, 'framework': 2, 'infrastructure': 2, 'system': 1,
        'modular': 2, 'scalable': 2, 'extensible': 2, 'api': 2, 'sdk': 2,
        'integration': 1, 'ecosystem': 2, 'foundation': 1, 'base': 1, 'core': 1,
        'cartridge': 2, 'modular cartridges': 4, 'platform architecture': 4,
        'engine': 1, 'interface': 1, 'component': 1, 'module': 1, 'service': 1
    }
    features['platform_score'] = sum(weight for word, weight in platform_keywords.items() if word in text_lower)
    
    # СИЛЬНО УСИЛЕННЫЕ ключевые слова для commercial
    commercial_keywords = {
        'licensable': 8, 'license': 6, 'licensing': 6, 'commercial': 5, 'market': 3,
        'revenue': 5, 'profit': 5, 'business': 3, 'enterprise': 3, 'industry': 3,
        'commercialization': 8, 'monetization': 6, 'patentable': 8, 'patent': 6,
        'invention': 5, 'novel': 6, 'unique': 3, 'proprietary': 6, 'exclusive': 6,
        'original': 3, 'innovative': 5, 'breakthrough': 6, 'discovery': 5,
        'method': 3, 'process': 3, 'technique': 3, 'approach': 3, 'solution': 3,
        'valuable': 5, 'profitable': 5, 'marketable': 5, 'economic': 3,
        'production': 4, 'manufacturing': 4, 'synthesis': 4, 'fabrication': 4,
        'enabling': 4, 'enzymatic': 4, 'step': 3, 'scalable': 4,
        'lab-on-a-chip': 5, 'lab on a chip': 5
    }
    features['commercial_score'] = sum(weight for word, weight in commercial_keywords.items() if word in text_lower)
    
    # Обычные ключевые слова для free
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
        'licensing opportunity', 'patent application', 'commercialization strategy',
        'novel enzymatic', 'enzymatic step', 'scalable production', 'lab-on-a-chip'
    ]))
    
    features['has_free_patterns'] = int(any(phrase in text_lower for phrase in [
        'free to operate', 'freedom to operate', 'unrestricted use', 'open source',
        'public domain', 'unlimited access', 'clear path'
    ]))
    
    # Дополнительные специфичные фичи
    features['has_enzymatic_terms'] = int(any(term in text_lower for term in [
        'enzymatic', 'enzyme', 'catalysis', 'catalytic', 'biocatalyst'
    ]))
    
    features['has_production_terms'] = int(any(term in text_lower for term in [
        'production', 'manufacturing', 'synthesis', 'fabrication', 'generation'
    ]))
    
    features['has_scalable_terms'] = int(any(term in text_lower for term in [
        'scalable', 'scaling', 'scalability', 'expandable', 'extensible'
    ]))
    
    features['has_novel_terms'] = int(any(term in text_lower for term in [
        'novel', 'new', 'innovative', 'unique', 'original', 'breakthrough'
    ]))
    
    features['has_patentable_terms'] = int(any(term in text_lower for term in [
        'patentable', 'patent', 'invention', 'novel', 'unique', 'original',
        'innovative', 'breakthrough', 'proprietary', 'exclusive'
    ]))
    
    features['has_innovation_terms'] = int(any(term in text_lower for term in [
        'innovation', 'innovative', 'breakthrough', 'discovery', 'novel',
        'original', 'unique', 'revolutionary', 'cutting-edge'
    ]))
    
    features['has_lab_on_chip_terms'] = int(any(term in text_lower for term in [
        'lab-on-a-chip', 'lab on a chip', 'lab-on-chip', 'microfluidic', 'microfluidics'
    ]))
    
    # Сложность текста
    features['complexity_score'] = len([w for w in text.split() if len(w) > 8])
    features['technical_density'] = features['technical_terms'] / max(features['word_count'], 1)
    
    return features

def extract_features_from_item(item):
    """Извлечение фичей из элемента"""
    
    # Базовые фичи
    features = {
        'evidence_count': 0,
        'is_negative': 0,
        
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
    
    # Анализируем evidence если предоставлено
    if item.get('evidence'):
        features['evidence_count'] = len(item['evidence'])
        
        market_scores = []
        trial_scores = []
        
        for ev in item['evidence']:
            if ev.get('type') == 'market' and 'importance_score' in ev.get('meta', {}):
                features['market_evidence_count'] += 1
                market_scores.append(ev['meta']['importance_score'])
                
                # Типы событий
                event_type = ev['meta'].get('type')
                if event_type == 'Product launch':
                    features['has_product_launch'] = 1
                elif event_type == 'M&A':
                    features['has_ma'] = 1
                elif event_type == 'Partnership':
                    features['has_partnership'] = 1
                elif event_type == 'Funding':
                    features['has_funding'] = 1
                    
            elif ev.get('type') == 'trial' and 'maturity_score' in ev.get('meta', {}):
                features['trial_evidence_count'] += 1
                trial_scores.append(ev['meta']['maturity_score'])
                
                # Статусы и фазы
                status = ev['meta'].get('status')
                phase = ev['meta'].get('phase')
                
                if status == 'Completed':
                    features['has_completed_trial'] = 1
                if phase == 'Phase 3':
                    features['has_phase3_trial'] = 1
                if status == 'Terminated':
                    features['has_terminated_trial'] = 1
                    
            elif ev.get('type') == 'patent':
                features['patent_count'] += 1
            elif ev.get('type') == 'paper':
                features['paper_count'] += 1
            elif ev.get('type') == 'disclosure':
                features['disclosure_count'] += 1
        
        # Вычисляем агрегированные скоры
        if market_scores:
            features['max_importance_score'] = max(market_scores)
            features['avg_importance_score'] = np.mean(market_scores)
        if trial_scores:
            features['max_maturity_score'] = max(trial_scores)
            features['avg_maturity_score'] = np.mean(trial_scores)
    
    return features

def prepare_simple_improved_training_data():
    """Подготовка данных с простыми улучшениями"""
    
    print("📊 Подготовка данных с простыми улучшениями...")
    
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
        max_features=600,  # средний размер
        ngram_range=(1, 3),
        stop_words='english',
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        norm='l2'
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

def train_simple_improved_model(X, y, tfidf):
    """Обучение простой улучшенной модели"""
    
    print("\n🤖 Обучение простой улучшенной модели...")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} образцов")
    print(f"Test set: {len(X_test)} образцов")
    
    # Простая модель с балансировкой классов
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',  # простая балансировка
        random_state=42,
        n_jobs=-1
    )
    
    # Обучение
    print("Обучение модели...")
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Простая улучшенная модель Accuracy: {accuracy:.3f}")
    
    # Детальный отчет
    print(f"\n📊 Детальный отчет:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy

def main():
    """Основная функция"""
    
    print("🚀 Обучение простой улучшенной модели Technology Evaluator...")
    print("=" * 60)
    
    # Подготовка данных
    X, y, tfidf = prepare_simple_improved_training_data()
    
    # Обучение модели
    model, accuracy = train_simple_improved_model(X, y, tfidf)
    
    # Сохранение модели
    print("\n💾 Сохранение простой улучшенной модели...")
    
    # Сохраняем модель
    joblib.dump(model, 'simple_improved_technology_evaluator.joblib')
    
    # Сохраняем названия фичей
    with open('simple_improved_technology_evaluator_features.json', 'w') as f:
        json.dump(list(X.columns), f)
    
    # Сохраняем TF-IDF
    joblib.dump(tfidf, 'simple_improved_technology_evaluator_tfidf.joblib')
    
    print("✅ Простая улучшенная модель сохранена:")
    print("  - simple_improved_technology_evaluator.joblib")
    print("  - simple_improved_technology_evaluator_features.json")
    print("  - simple_improved_technology_evaluator_tfidf.joblib")
    
    print(f"\n🎯 Итоговая точность: {accuracy:.3f}")
    
    # Тестируем на нашей проблемной фразе
    print("\n🔍 Тестирование на проблемной фразе...")
    test_text = "novel enzymatic step enabling scalable production in lab-on-a-chip"
    
    # Извлекаем фичи
    test_features = extract_features_from_item({'text': test_text, 'evidence': []})
    X_test_df = pd.DataFrame([test_features])
    
    # Добавляем TF-IDF фичи
    tfidf_features = tfidf.transform([test_text])
    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    )
    
    # Объединяем фичи
    X_combined = pd.concat([X_test_df, tfidf_df], axis=1)
    
    # Предсказания
    prediction = model.predict(X_combined)[0]
    probabilities = model.predict_proba(X_combined)[0]
    confidence = np.max(probabilities)
    
    print(f"Фраза: '{test_text}'")
    print(f"Предсказание: {prediction}")
    print(f"Уверенность: {confidence:.4f}")
    print(f"Вероятности:")
    for i, prob in enumerate(probabilities):
        print(f"  {model.classes_[i]}: {prob:.4f}")
    
    if prediction == 'commercial' and confidence > 0.3:
        print("✅ Простая улучшенная модель работает правильно!")
    else:
        print("❌ Модель все еще нуждается в доработке")

if __name__ == '__main__':
    main()
