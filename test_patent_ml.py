#!/usr/bin/env python3
"""
Тестирование ML-модели для патентных рекомендаций
"""

import requests
import json

API_BASE_URL = 'http://localhost:5001'

def test_patent_recommendation():
    """Тестирование эндпоинта патентных рекомендаций"""
    print("🧪 Тестирование патентных рекомендаций...")
    
    test_cases = [
        {
            "name": "Высокий патентный потенциал",
            "idea_text": "A novel machine learning algorithm for drug discovery that can identify new therapeutic targets using quantum computing",
            "evidence": [
                {"type": "patent", "meta": {"title": "ML Drug Discovery Patent"}},
                {"type": "paper", "meta": {"title": "AI in Drug Discovery"}},
                {"type": "market", "meta": {"type": "Product launch", "importance_score": 4.0}}
            ]
        },
        {
            "name": "Средний патентный потенциал", 
            "idea_text": "An innovative software platform for data analysis with modular architecture",
            "evidence": [
                {"type": "paper", "meta": {"title": "Data Analysis Framework"}}
            ]
        },
        {
            "name": "Низкий патентный потенциал",
            "idea_text": "A simple website for online shopping",
            "evidence": []
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Тест {i}: {test_case['name']}")
        print(f"Идея: {test_case['idea_text'][:60]}...")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/patent_recommendation",
                json={
                    "idea_text": test_case["idea_text"],
                    "evidence": test_case["evidence"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                patent_rec = result['patent_recommendation']
                
                print(f"✅ ML Recommendation: {patent_rec['ml_model']['recommendation']}")
                print(f"✅ AI Recommendation: {patent_rec['ai_analysis']['recommendation']}")
                print(f"✅ Combined Decision: {patent_rec['combined']['recommendation']}")
                print(f"✅ Confidence: {patent_rec['combined']['confidence']:.2f}")
                print(f"✅ Patent Score: {patent_rec['ml_model']['patent_score']:.1f}/10")
                
                if patent_rec['combined']['all_reasons']:
                    print(f"✅ Reasons: {', '.join(patent_rec['combined']['all_reasons'][:2])}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")

def test_main_evaluate_with_patents():
    """Тестирование основного эндпоинта с патентными рекомендациями"""
    print("\n🧪 Тестирование основного эндпоинта с патентами...")
    
    test_idea = "A revolutionary blockchain-based identity verification system using zero-knowledge proofs"
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/evaluate",
            json={
                "idea_text": test_idea,
                "evidence": [
                    {"type": "patent", "meta": {"title": "Blockchain Identity Patent"}},
                    {"type": "market", "meta": {"type": "Product launch", "importance_score": 3.5}}
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Основная рекомендация: {result['analysis_summary']['prediction']}")
            
            if 'patent_analysis' in result:
                patent_analysis = result['patent_analysis']
                print(f"✅ Патентная рекомендация: {patent_analysis['combined']['recommendation']}")
                print(f"✅ Патентный скор: {patent_analysis['ml_model']['patent_score']:.1f}/10")
                print(f"✅ Уверенность: {patent_analysis['combined']['confidence']:.2f}")
                
                # Показываем фичи ML модели
                features = patent_analysis['ml_model']['features']
                print(f"✅ Патентные ключевые слова: {features['patent_keyword_score']}")
                print(f"✅ Технические индикаторы: {features['technical_innovation_score']}")
                print(f"✅ Доказательства: патенты={features['patent_evidence_count']}, исследования={features['research_evidence_count']}")
            else:
                print("❌ Патентный анализ не найден в ответе")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_api_health():
    """Проверка здоровья API"""
    print("🏥 Проверка здоровья API...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/readiness", timeout=10)
        if response.status_code == 200:
            print("✅ API готов к работе")
            return True
        else:
            print(f"❌ API не готов: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API недоступен: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск тестов патентной ML-модели")
    print("=" * 50)
    
    if test_api_health():
        test_patent_recommendation()
        test_main_evaluate_with_patents()
    
    print("\n✅ Тестирование завершено!")
