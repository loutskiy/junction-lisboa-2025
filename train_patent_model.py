import json
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def extract_patent_features(text):
    """Извлечение фичей для патентной модели"""
    
    text_lower = text.lower()
    
    features = {
        # Базовые характеристики
        'text_length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
        'unique_word_ratio': len(set(text.split())) / len(text.split()) if text.split() else 0,
        'sentence_count': len([s for s in text.split('.') if s.strip()]),
        
        # Патентные ключевые слова с высокими весами
        'patent_keyword_score': 0,
        'novelty_score': 0,
        'innovation_score': 0,
        'technical_complexity_score': 0,
        'commercial_potential_score': 0,
        
        # Специфичные патентные термины
        'has_patent_terms': 0,
        'has_invention_terms': 0,
        'has_novelty_terms': 0,
        'has_technical_terms': 0,
        'has_commercial_terms': 0,
        
        # Сложность и детализация
        'complexity_score': 0,
        'technical_density': 0,
        'has_measurements': 0,
        'has_numbers': 0,
        'has_specifics': 0,
        
        # Структурные особенности
        'has_method_description': 0,
        'has_process_description': 0,
        'has_device_description': 0,
        'has_composition_description': 0,
    }
    
    # Патентные ключевые слова с весами
    patent_keywords = {
        'patent': 8, 'patentable': 8, 'intellectual property': 8, 'ip': 6,
        'invention': 7, 'novel': 7, 'new': 5, 'original': 5,
        'innovative': 6, 'breakthrough': 7, 'revolutionary': 7,
        'unique': 5, 'proprietary': 6, 'exclusive': 6,
        'method': 4, 'process': 4, 'technique': 4, 'approach': 3,
        'device': 4, 'apparatus': 4, 'system': 3, 'mechanism': 4,
        'composition': 5, 'formula': 5, 'compound': 5, 'material': 3,
        'algorithm': 5, 'software': 3, 'application': 3,
        'discovery': 6, 'development': 3, 'research': 2,
        'manufacturing': 4, 'production': 4, 'synthesis': 4,
        'enabling': 4, 'scalable': 4, 'efficient': 3,
        'optimized': 3, 'improved': 4, 'enhanced': 4,
        'advanced': 4, 'cutting-edge': 6, 'state-of-the-art': 6
    }
    
    features['patent_keyword_score'] = sum(weight for word, weight in patent_keywords.items() if word in text_lower)
    
    # Термины новизны
    novelty_terms = {
        'novel': 6, 'new': 4, 'original': 4, 'first': 5,
        'unprecedented': 6, 'never before': 6, 'groundbreaking': 6,
        'pioneering': 5, 'innovative': 5, 'revolutionary': 6
    }
    features['novelty_score'] = sum(weight for word, weight in novelty_terms.items() if word in text_lower)
    
    # Термины инновации
    innovation_terms = {
        'innovation': 5, 'breakthrough': 6, 'advance': 4,
        'improvement': 4, 'enhancement': 4, 'optimization': 4,
        'discovery': 5, 'invention': 6, 'creation': 4,
        'development': 3, 'research': 2, 'study': 2
    }
    features['innovation_score'] = sum(weight for word, weight in innovation_terms.items() if word in text_lower)
    
    # Техническая сложность
    technical_terms = {
        'complex': 4, 'sophisticated': 4, 'advanced': 4,
        'cutting-edge': 6, 'state-of-the-art': 6, 'high-performance': 4,
        'optimized': 3, 'efficient': 3, 'robust': 3, 'reliable': 3,
        'algorithm': 5, 'method': 4, 'process': 4, 'technique': 4,
        'engineering': 4, 'design': 3, 'implementation': 3,
        'manufacturing': 4, 'production': 4, 'fabrication': 4,
        'synthesis': 4, 'analysis': 3, 'processing': 3
    }
    features['technical_complexity_score'] = sum(weight for word, weight in technical_terms.items() if word in text_lower)
    
    # Коммерческий потенциал
    commercial_terms = {
        'commercial': 5, 'market': 4, 'business': 3, 'revenue': 5,
        'profit': 5, 'valuable': 4, 'profitable': 5, 'marketable': 5,
        'licensable': 6, 'license': 5, 'licensing': 5,
        'enterprise': 4, 'industry': 3, 'economic': 3,
        'commercialization': 6, 'monetization': 5
    }
    features['commercial_potential_score'] = sum(weight for word, weight in commercial_terms.items() if word in text_lower)
    
    # Бинарные фичи для специфичных терминов
    features['has_patent_terms'] = int(any(term in text_lower for term in ['patent', 'patentable', 'intellectual property', 'ip']))
    features['has_invention_terms'] = int(any(term in text_lower for term in ['invention', 'invent', 'inventor']))
    features['has_novelty_terms'] = int(any(term in text_lower for term in ['novel', 'new', 'original', 'first', 'unprecedented']))
    features['has_technical_terms'] = int(any(term in text_lower for term in ['technical', 'technology', 'engineering', 'algorithm', 'method']))
    features['has_commercial_terms'] = int(any(term in text_lower for term in ['commercial', 'market', 'business', 'revenue', 'profit']))
    
    # Сложность и детализация
    features['complexity_score'] = len([w for w in text.split() if len(w) > 8])
    features['technical_density'] = features['technical_complexity_score'] / max(features['word_count'], 1)
    features['has_measurements'] = int(bool(re.search(r'\d+\s*(mg|ml|kg|g|m|cm|mm|nm|μm|%|hz|mhz|ghz)', text_lower)))
    features['has_numbers'] = int(bool(re.search(r'\d+', text)))
    features['has_specifics'] = int(any(word in text_lower for word in ['specific', 'precise', 'exact', 'detailed', 'comprehensive']))
    
    # Структурные особенности
    features['has_method_description'] = int(any(phrase in text_lower for phrase in ['method for', 'method of', 'method to', 'process for', 'process of']))
    features['has_process_description'] = int(any(phrase in text_lower for phrase in ['process for', 'process of', 'process to', 'procedure for', 'procedure of']))
    features['has_device_description'] = int(any(phrase in text_lower for phrase in ['device for', 'device of', 'apparatus for', 'system for', 'machine for']))
    features['has_composition_description'] = int(any(phrase in text_lower for phrase in ['composition of', 'formula for', 'compound of', 'mixture of', 'blend of']))
    
    return features

def prepare_patent_training_data():
    """Подготовка данных для обучения патентной модели"""
    
    print("📊 Загрузка данных...")
    
    # Загружаем данные
    data = []
    with open('data/train_final.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    print(f"✅ Загружено {len(data)} записей")
    
    # Подготавливаем данные для обучения
    X_texts = []
    y_patent = []
    
    for item in data:
        # Извлекаем items из каждой записи
        for sub_item in item['items']:
            text = sub_item['text']
            original_label = sub_item['label']
            
            # Создаем патентные метки на основе оригинальных лейблов
            if original_label in ['patentable', 'licensable']:
                patent_label = 1  # Высокий патентный потенциал
            elif original_label == 'platform':
                patent_label = 0  # Низкий патентный потенциал
            elif original_label == 'freetooperate_flag':
                patent_label = 0  # Низкий патентный потенциал
            else:  # none
                patent_label = 0  # Низкий патентный потенциал
            
            X_texts.append(text)
            y_patent.append(patent_label)
    
    print(f"📈 Распределение патентных меток:")
    print(f"  Высокий потенциал (1): {sum(y_patent)}")
    print(f"  Низкий потенциал (0): {len(y_patent) - sum(y_patent)}")
    
    # Извлекаем фичи
    print("🔧 Извлечение фичей...")
    X_features = []
    for text in X_texts:
        features = extract_patent_features(text)
        X_features.append(features)
    
    # Создаем DataFrame
    X_df = pd.DataFrame(X_features)
    
    # Добавляем TF-IDF фичи
    print("📝 Создание TF-IDF фичей...")
    tfidf = TfidfVectorizer(
        max_features=300,  # Меньше фичей для патентной модели
        ngram_range=(1, 2),  # Только биграммы
        stop_words='english',
        min_df=3,
        max_df=0.8
    )
    
    tfidf_features = tfidf.fit_transform(X_texts)
    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=[f'patent_tfidf_{i}' for i in range(tfidf_features.shape[1])]
    )
    
    # Объединяем фичи
    X_combined = pd.concat([X_df, tfidf_df], axis=1)
    
    print(f"✅ Итоговые фичи: {X_combined.shape[1]}")
    print(f"  Базовые фичи: {X_df.shape[1]}")
    print(f"  TF-IDF фичи: {tfidf_df.shape[1]}")
    
    return X_combined, np.array(y_patent), tfidf, X_df.columns.tolist()

def train_patent_model():
    """Обучение патентной модели"""
    
    print("🚀 Обучение патентной модели...")
    
    # Подготавливаем данные
    X, y, tfidf, feature_names = prepare_patent_training_data()
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Размер обучающей выборки: {X_train.shape}")
    print(f"📊 Размер тестовой выборки: {X_test.shape}")
    
    # Создаем модели
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
    
    lr = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
    
    # Создаем ансамбль
    patent_model = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft'
    )
    
    # Обучаем модель
    print("🎯 Обучение модели...")
    patent_model.fit(X_train, y_train)
    
    # Предсказания на тестовой выборке
    y_pred = patent_model.predict(X_test)
    y_pred_proba = patent_model.predict_proba(X_test)[:, 1]
    
    # Оценка качества
    print("\n📊 Результаты на тестовой выборке:")
    print(f"Точность: {accuracy_score(y_test, y_pred):.4f}")
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred, target_names=['Low Patent Potential', 'High Patent Potential']))
    
    # Кросс-валидация
    print("\n🔄 Кросс-валидация (5-fold):")
    cv_scores = cross_val_score(patent_model, X, y, cv=5, scoring='accuracy')
    print(f"Средняя точность: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Анализ важности фичей (только для базовых фичей)
    print("\n🔍 Топ-10 важных фичей:")
    base_feature_importance = patent_model.named_estimators_['rf'].feature_importances_[:len(feature_names)]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': base_feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Сохраняем модель и фичи
    print("\n💾 Сохранение модели...")
    joblib.dump(patent_model, 'patent_potential_model.joblib')
    joblib.dump(tfidf, 'patent_potential_tfidf.joblib')
    
    with open('patent_potential_features.json', 'w') as f:
        json.dump(feature_names, f)
    
    print("✅ Патентная модель сохранена!")
    print("   - patent_potential_model.joblib")
    print("   - patent_potential_tfidf.joblib")
    print("   - patent_potential_features.json")
    
    return patent_model, tfidf, feature_names

if __name__ == "__main__":
    train_patent_model()
