from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Глобальные переменные для модели
model = None
feature_names = None
tfidf = None

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
        # Анализируем контекст
        if any(word in text_lower for word in ['technology', 'innovation', 'development', 'research']):
            return 'commercial'  # По умолчанию коммерческий
        elif any(word in text_lower for word in ['system', 'method', 'approach']):
            return 'platform'  # Платформа
        else:
            return 'commercial'  # По умолчанию коммерческий
    
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

def extract_features_from_idea(idea_text, evidence_data=None):
    """Извлечение фичей из идеи"""
    
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
    enhanced_features = extract_enhanced_features(idea_text)
    features.update(enhanced_features)
    
    # Анализируем evidence если предоставлено
    if evidence_data:
        features['evidence_count'] = len(evidence_data)
        
        market_scores = []
        trial_scores = []
        
        for ev in evidence_data:
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

def load_model():
    """Загрузка обученной модели"""
    global model, feature_names, tfidf
    
    if model is None:
        print("📦 Загрузка модели...")
        
        # Загружаем модель
        model = joblib.load('final_technology_evaluator.joblib')
        
        # Загружаем названия фичей
        with open('final_technology_evaluator_features.json', 'r') as f:
            feature_names = json.load(f)
        
        # Загружаем TF-IDF
        tfidf = joblib.load('final_technology_evaluator_tfidf.joblib')
        
        print("✅ Модель загружена успешно")

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка здоровья API"""
    return jsonify({
        'status': 'healthy',
        'message': 'Technology Evaluator API is running',
        'model_loaded': model is not None
    })

@app.route('/readiness', methods=['GET'])
def readiness_check():
    """Проверка готовности API"""
    if model is None:
        return jsonify({
            'status': 'not_ready',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'ready',
        'message': 'API is ready to process requests'
    }), 200

@app.route('/evaluate', methods=['POST'])
def evaluate_idea():
    """Оценка технологической идеи"""
    
    # Загружаем модель если не загружена
    load_model()
    
    try:
        data = request.get_json()
        
        if not data or 'idea_text' not in data:
            return jsonify({
                'error': 'Missing required field: idea_text'
            }), 400
        
        idea_text = data['idea_text']
        evidence_data = data.get('evidence', [])
        
        # Извлекаем фичи
        features = extract_features_from_idea(idea_text, evidence_data)
        
        # Конвертируем в DataFrame
        X_df = pd.DataFrame([features])
        
        # Добавляем TF-IDF фичи
        tfidf_features = tfidf.transform([idea_text])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Объединяем фичи
        X_combined = pd.concat([X_df, tfidf_df], axis=1)
        
        # Предсказания
        ml_prediction = model.predict(X_combined)[0]
        ml_probabilities = model.predict_proba(X_combined)[0]
        ml_confidence = np.max(ml_probabilities)
        
        # Правила
        rule_prediction = rule_based_classifier(idea_text)
        
        # Гибридный подход
        if ml_confidence > 0.7:  # Высокая уверенность ML модели
            hybrid_prediction = ml_prediction
            method = 'ml_high_confidence'
        elif ml_confidence > 0.4:  # Средняя уверенность - взвешенное решение
            if rule_prediction == ml_prediction:
                hybrid_prediction = ml_prediction
                method = 'ml_rule_agreement'
            else:
                hybrid_prediction = ml_prediction
                method = 'ml_medium_confidence'
        else:  # Низкая уверенность - используем правила
            hybrid_prediction = rule_prediction
            method = 'rule_based'
        
        # Анализ доказательств
        evidence_analysis = analyze_evidence(evidence_data)
        
        # Формируем подробный ответ
        response = {
            'idea_text': idea_text,
            'analysis_summary': {
                'prediction': hybrid_prediction,
                'confidence': float(ml_confidence),
                'method': method,
                'evidence_count': evidence_analysis['count'],
                'evidence_types': evidence_analysis['types'],
                'insights': evidence_analysis['insights']
            },
            'predictions': {
                'ml_model': {
                    'prediction': ml_prediction,
                    'confidence': float(ml_confidence),
                    'probabilities': {
                        'commercial': float(ml_probabilities[0]),
                        'free': float(ml_probabilities[1]),
                        'none': float(ml_probabilities[2]),
                        'platform': float(ml_probabilities[3])
                    }
                },
                'rule_based': {
                    'prediction': rule_prediction
                },
                'hybrid': {
                    'prediction': hybrid_prediction,
                    'method': method,
                    'confidence_level': get_confidence_level(features)
                }
            },
            'detailed_features': {
                'text_analysis': {
                    'text_length': features['text_length'],
                    'word_count': features['word_count'],
                    'avg_word_length': features['avg_word_length'],
                    'unique_word_ratio': features['unique_word_ratio'],
                    'sentence_count': features['sentence_count'],
                    'complexity_score': features['complexity_score']
                },
                'keyword_scores': {
                    'platform_score': features['platform_score'],
                    'commercial_score': features['commercial_score'],
                    'free_score': features['free_score'],
                    'technical_terms': features['technical_terms'],
                    'scientific_terms': features['scientific_terms'],
                    'business_terms': features['business_terms']
                },
                'evidence_analysis': {
                    'total_evidence': features['evidence_count'],
                    'market_signals': features['market_evidence_count'],
                    'clinical_trials': features['trial_evidence_count'],
                    'patents': features['patent_count'],
                    'papers': features['paper_count'],
                    'disclosures': features['disclosure_count']
                },
                'market_indicators': {
                    'max_importance_score': features['max_importance_score'],
                    'avg_importance_score': features['avg_importance_score'],
                    'product_launches': features['has_product_launch'],
                    'mergers_acquisitions': features['has_ma'],
                    'partnerships': features['has_partnership'],
                    'funding_activity': features['has_funding']
                },
                'clinical_indicators': {
                    'max_maturity_score': features['max_maturity_score'],
                    'avg_maturity_score': features['avg_maturity_score'],
                    'completed_trials': features['has_completed_trial'],
                    'phase3_trials': features['has_phase3_trial'],
                    'terminated_trials': features['has_terminated_trial']
                },
                'pattern_analysis': {
                    'has_platform_patterns': features['has_platform_patterns'],
                    'has_commercial_patterns': features['has_commercial_patterns'],
                    'has_free_patterns': features['has_free_patterns'],
                    'has_numbers': features['has_numbers'],
                    'has_measurements': features['has_measurements'],
                    'has_comparisons': features['has_comparisons']
                }
            },
            'recommendations': generate_recommendations(hybrid_prediction, features, evidence_data),
            'metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_version': 'final_technology_evaluator',
                'api_version': '1.0.0',
                'processing_time_ms': 0  # Можно добавить измерение времени
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

def generate_recommendations(prediction, features, evidence_data):
    """Генерация подробных рекомендаций на основе предсказания"""
    
    recommendations = []
    
    # Анализ доказательств
    evidence_analysis = analyze_evidence(evidence_data)
    
    if prediction == 'commercial':
        # Основная рекомендация
        recommendations.append({
            'type': 'commercial_opportunity',
            'priority': 'high',
            'title': 'Коммерческий потенциал',
            'message': 'Эта идея имеет высокий коммерческий потенциал для лицензирования и монетизации',
            'confidence': get_confidence_level(features),
            'actions': [
                'Проведите патентный поиск и анализ свободы действий (FTO)',
                'Оцените рыночный размер и конкурентную среду',
                'Разработайте стратегию лицензирования',
                'Рассмотрите партнерство с крупными компаниями',
                'Подготовьте бизнес-план коммерциализации'
            ],
            'timeline': '3-6 месяцев',
            'investment': 'Средний-высокий',
            'roi_potential': 'Высокий'
        })
        
        # Рыночные возможности
        if features['market_evidence_count'] > 0:
            recommendations.append({
                'type': 'market_analysis',
                'priority': 'high',
                'title': 'Рыночные возможности',
                'message': f'Обнаружено {features["market_evidence_count"]} рыночных сигналов',
                'details': {
                    'product_launches': features['has_product_launch'],
                    'mergers_acquisitions': features['has_ma'],
                    'partnerships': features['has_partnership'],
                    'funding_rounds': features['has_funding'],
                    'market_importance_score': features['max_importance_score']
                },
                'actions': [
                    'Изучите успешные кейсы запуска продуктов',
                    'Проанализируйте стратегии M&A в отрасли',
                    'Оцените возможности партнерства',
                    'Исследуйте источники финансирования'
                ]
            })
        
        # Клинические испытания
        if features['trial_evidence_count'] > 0:
            recommendations.append({
                'type': 'clinical_development',
                'priority': 'medium',
                'title': 'Клиническое развитие',
                'message': f'Найдено {features["trial_evidence_count"]} связанных клинических испытаний',
                'details': {
                    'maturity_score': features['max_maturity_score'],
                    'completed_trials': features['has_completed_trial'],
                    'phase3_trials': features['has_phase3_trial'],
                    'terminated_trials': features['has_terminated_trial']
                },
                'actions': [
                    'Изучите результаты завершенных испытаний',
                    'Оцените регуляторные требования',
                    'Планируйте собственные клинические исследования',
                    'Рассмотрите сотрудничество с медицинскими центрами'
                ]
            })
        
        # Патентная стратегия
        if features['patent_count'] > 0:
            recommendations.append({
                'type': 'patent_strategy',
                'priority': 'high',
                'title': 'Патентная стратегия',
                'message': f'Обнаружено {features["patent_count"]} связанных патентов',
                'actions': [
                    'Проведите патентный ландшафт-анализ',
                    'Оцените свободу действий (Freedom to Operate)',
                    'Разработайте стратегию патентования',
                    'Рассмотрите лицензирование существующих патентов'
                ]
            })
    
    elif prediction == 'platform':
        recommendations.append({
            'type': 'platform_development',
            'priority': 'medium',
            'title': 'Платформенная разработка',
            'message': 'Эта идея идеально подходит для создания модульной платформы',
            'confidence': get_confidence_level(features),
            'actions': [
                'Спроектируйте модульную архитектуру с четкими интерфейсами',
                'Разработайте API и SDK для разработчиков',
                'Создайте экосистему плагинов и расширений',
                'Планируйте горизонтальное масштабирование',
                'Обеспечьте обратную совместимость версий'
            ],
            'timeline': '6-12 месяцев',
            'investment': 'Средний',
            'roi_potential': 'Средний-высокий'
        })
        
        # Техническая архитектура
        recommendations.append({
            'type': 'technical_architecture',
            'priority': 'high',
            'title': 'Техническая архитектура',
            'message': 'Рекомендации по техническому дизайну платформы',
            'actions': [
                'Используйте микросервисную архитектуру',
                'Реализуйте API-first подход',
                'Обеспечьте безопасность и аутентификацию',
                'Добавьте мониторинг и аналитику',
                'Создайте документацию для разработчиков'
            ]
        })
    
    elif prediction == 'free':
        recommendations.append({
            'type': 'open_source',
            'priority': 'low',
            'title': 'Открытое развитие',
            'message': 'Эта идея лучше всего подходит для открытого развития и публичного доступа',
            'confidence': get_confidence_level(features),
            'actions': [
                'Выберите подходящую open source лицензию (MIT, Apache 2.0, GPL)',
                'Создайте активное сообщество разработчиков',
                'Обеспечьте качественную документацию и примеры',
                'Настройте CI/CD для автоматического тестирования',
                'Планируйте регулярные релизы и обратную связь'
            ],
            'timeline': '1-3 месяца',
            'investment': 'Низкий',
            'roi_potential': 'Низкий-средний'
        })
        
        # Сообщество и экосистема
        recommendations.append({
            'type': 'community_building',
            'priority': 'medium',
            'title': 'Построение сообщества',
            'message': 'Стратегия создания активного сообщества пользователей',
            'actions': [
                'Создайте GitHub репозиторий с четким README',
                'Настройте issue tracker и pull request процесс',
                'Организуйте регулярные митапы и конференции',
                'Создайте форум или Discord сервер',
                'Привлекайте контрибьюторов через hackathons'
            ]
        })
    
    else:  # none
        recommendations.append({
            'type': 'not_recommended',
            'priority': 'low',
            'title': 'Не рекомендуется',
            'message': 'Эта идея не рекомендуется для развития в текущем виде',
            'confidence': get_confidence_level(features),
            'reasons': get_rejection_reasons(features),
            'actions': [
                'Пересмотрите основную концепцию и ценностное предложение',
                'Изучите альтернативные технологические подходы',
                'Проведите дополнительное исследование рынка',
                'Рассмотрите пивот в другую область применения',
                'Получите экспертную оценку от отраслевых специалистов'
            ],
            'timeline': '1-3 месяца',
            'investment': 'Низкий',
            'roi_potential': 'Низкий'
        })
    
    # Общие рекомендации
    recommendations.append({
        'type': 'general_advice',
        'priority': 'medium',
        'title': 'Общие рекомендации',
        'message': 'Дополнительные советы для успешной реализации',
        'actions': [
            'Проведите детальный анализ конкурентов',
            'Изучите регуляторные требования в вашей юрисдикции',
            'Оцените необходимые ресурсы и команду',
            'Разработайте план защиты интеллектуальной собственности',
            'Создайте MVP для валидации концепции'
        ]
    })
    
    return recommendations

def analyze_evidence(evidence_data):
    """Анализ доказательств и извлечение инсайтов"""
    
    if not evidence_data:
        return {'count': 0, 'types': [], 'insights': []}
    
    evidence_types = {}
    insights = []
    
    for evidence in evidence_data:
        evidence_type = evidence.get('type', 'unknown')
        evidence_types[evidence_type] = evidence_types.get(evidence_type, 0) + 1
        
        # Анализ метаданных
        meta = evidence.get('meta', {})
        
        if evidence_type == 'market':
            if meta.get('type') == 'Product launch':
                insights.append('Обнаружен запуск продукта в связанной области')
            elif meta.get('type') == 'M&A':
                insights.append('Активность слияний и поглощений в отрасли')
            elif meta.get('type') == 'Funding':
                insights.append('Инвестиционная активность в секторе')
        
        elif evidence_type == 'trial':
            if meta.get('status') == 'Completed':
                insights.append('Завершенные клинические испытания')
            elif meta.get('phase') == 'Phase 3':
                insights.append('Клинические испытания фазы 3')
        
        elif evidence_type == 'patent':
            insights.append('Существующие патенты в области')
        
        elif evidence_type == 'paper':
            insights.append('Научные публикации по теме')
    
    return {
        'count': len(evidence_data),
        'types': evidence_types,
        'insights': insights
    }

def get_confidence_level(features):
    """Определение уровня уверенности на основе фичей"""
    
    confidence_score = 0
    
    # Базовые индикаторы уверенности
    if features['commercial_score'] > 5:
        confidence_score += 2
    if features['platform_score'] > 5:
        confidence_score += 2
    if features['free_score'] > 5:
        confidence_score += 2
    
    # Доказательства
    if features['evidence_count'] > 0:
        confidence_score += 1
    if features['market_evidence_count'] > 0:
        confidence_score += 1
    if features['trial_evidence_count'] > 0:
        confidence_score += 1
    
    # Технические индикаторы
    if features['technical_terms'] > 3:
        confidence_score += 1
    if features['has_commercial_patterns']:
        confidence_score += 1
    if features['has_platform_patterns']:
        confidence_score += 1
    if features['has_free_patterns']:
        confidence_score += 1
    
    if confidence_score >= 6:
        return 'high'
    elif confidence_score >= 3:
        return 'medium'
    else:
        return 'low'

def get_rejection_reasons(features):
    """Определение причин отклонения идеи"""
    
    reasons = []
    
    if features['commercial_score'] < 2 and features['platform_score'] < 2 and features['free_score'] < 2:
        reasons.append('Недостаточно ключевых слов для классификации')
    
    if features['technical_terms'] < 2:
        reasons.append('Низкая техническая сложность')
    
    if features['evidence_count'] == 0:
        reasons.append('Отсутствие подтверждающих доказательств')
    
    if features['text_length'] < 50:
        reasons.append('Недостаточно детальное описание')
    
    if features['word_count'] < 10:
        reasons.append('Слишком краткое описание')
    
    return reasons

@app.route('/batch_evaluate', methods=['POST'])
def batch_evaluate():
    """Пакетная оценка нескольких идей"""
    
    # Загружаем модель если не загружена
    load_model()
    
    try:
        data = request.get_json()
        
        if not data or 'ideas' not in data:
            return jsonify({
                'error': 'Missing required field: ideas'
            }), 400
        
        ideas = data['ideas']
        results = []
        
        for i, idea_data in enumerate(ideas):
            idea_text = idea_data.get('idea_text', '')
            evidence_data = idea_data.get('evidence', [])
            
            # Извлекаем фичи
            features = extract_features_from_idea(idea_text, evidence_data)
            
            # Конвертируем в DataFrame
            X_df = pd.DataFrame([features])
            
            # Добавляем TF-IDF фичи
            tfidf_features = tfidf.transform([idea_text])
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            
            # Объединяем фичи
            X_combined = pd.concat([X_df, tfidf_df], axis=1)
            
            # Предсказания
            ml_prediction = model.predict(X_combined)[0]
            ml_probabilities = model.predict_proba(X_combined)[0]
            ml_confidence = np.max(ml_probabilities)
            
            # Правила
            rule_prediction = rule_based_classifier(idea_text)
            
            # Гибридный подход
            if ml_confidence > 0.7:
                hybrid_prediction = ml_prediction
                method = 'ml_high_confidence'
            elif ml_confidence > 0.4:
                if rule_prediction == ml_prediction:
                    hybrid_prediction = ml_prediction
                    method = 'ml_rule_agreement'
                else:
                    hybrid_prediction = ml_prediction
                    method = 'ml_medium_confidence'
            else:
                hybrid_prediction = rule_prediction
                method = 'rule_based'
            
            results.append({
                'id': i,
                'idea_text': idea_text,
                'prediction': hybrid_prediction,
                'confidence': float(ml_confidence),
                'method': method
            })
        
        return jsonify({
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Запуск Technology Evaluator API...")
    print("=" * 50)
    
    # Загружаем модель при запуске
    load_model()
    
    print("🌐 API доступен по адресу: http://localhost:5001")
    print("📚 Документация API:")
    print("  GET  /health - проверка здоровья")
    print("  GET  /readiness - проверка готовности")
    print("  POST /evaluate - оценка одной идеи")
    print("  POST /batch_evaluate - пакетная оценка")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
