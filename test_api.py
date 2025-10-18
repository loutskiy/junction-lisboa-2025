import requests
import json

# Базовый URL API
BASE_URL = "http://localhost:5001"

def test_health():
    """Тест проверки здоровья API"""
    print("🔍 Тестирование /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_readiness():
    """Тест проверки готовности API"""
    print("🔍 Тестирование /readiness...")
    response = requests.get(f"{BASE_URL}/readiness")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_evaluate_single():
    """Тест оценки одной идеи"""
    print("🔍 Тестирование /evaluate...")
    
    test_idea = {
        "idea_text": "A novel machine learning algorithm for drug discovery that can be licensed to pharmaceutical companies",
        "evidence": [
            {
                "type": "market",
                "meta": {
                    "type": "Product launch",
                    "importance_score": 3.5
                }
            },
            {
                "type": "patent",
                "meta": {
                    "patent_id": "US123456789"
                }
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/evaluate", json=test_idea)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['predictions']['hybrid']['prediction']}")
        print(f"Confidence: {result['predictions']['ml_model']['confidence']:.3f}")
        print(f"Method: {result['predictions']['hybrid']['method']}")
        print("Recommendations:")
        for rec in result['recommendations']:
            print(f"  - {rec['message']}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_evaluate_platform():
    """Тест оценки платформенной идеи"""
    print("🔍 Тестирование платформенной идеи...")
    
    test_idea = {
        "idea_text": "A modular platform architecture with scalable cartridges for data processing",
        "evidence": []
    }
    
    response = requests.post(f"{BASE_URL}/evaluate", json=test_idea)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['predictions']['hybrid']['prediction']}")
        print(f"Confidence: {result['predictions']['ml_model']['confidence']:.3f}")
        print(f"Method: {result['predictions']['hybrid']['method']}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_evaluate_free():
    """Тест оценки свободной идеи"""
    print("🔍 Тестирование свободной идеи...")
    
    test_idea = {
        "idea_text": "An open source framework for free to operate data analysis tools",
        "evidence": []
    }
    
    response = requests.post(f"{BASE_URL}/evaluate", json=test_idea)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['predictions']['hybrid']['prediction']}")
        print(f"Confidence: {result['predictions']['ml_model']['confidence']:.3f}")
        print(f"Method: {result['predictions']['hybrid']['method']}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_batch_evaluate():
    """Тест пакетной оценки"""
    print("🔍 Тестирование /batch_evaluate...")
    
    test_ideas = {
        "ideas": [
            {
                "idea_text": "A commercial software solution for enterprise data management",
                "evidence": []
            },
            {
                "idea_text": "A platform architecture with modular components",
                "evidence": []
            },
            {
                "idea_text": "An open source library for free to operate applications",
                "evidence": []
            },
            {
                "idea_text": "Some random text that doesn't make sense",
                "evidence": []
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/batch_evaluate", json=test_ideas)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total processed: {result['total_processed']}")
        for item in result['results']:
            print(f"  ID {item['id']}: {item['prediction']} (confidence: {item['confidence']:.3f})")
    else:
        print(f"Error: {response.json()}")
    print()

def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ TECHNOLOGY EVALUATOR API")
    print("=" * 60)
    
    try:
        # Тестируем все эндпоинты
        test_health()
        test_readiness()
        test_evaluate_single()
        test_evaluate_platform()
        test_evaluate_free()
        test_batch_evaluate()
        
        print("✅ Все тесты завершены успешно!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка подключения к API. Убедитесь, что сервер запущен на http://localhost:5000")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
