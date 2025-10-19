# 🚀 Technology Evaluator

**AI system for automatic evaluation of technology ideas** with machine learning and AI analysis. The system determines the commercial potential of ideas in seconds and recommends development strategies.

## 📋 What is this?

Technology Evaluator analyzes text descriptions of technology ideas and classifies them into 4 categories:
- **Commercial** - commercial potential for licensing and monetization
- **Platform** - suitable for creating a modular platform
- **Free** - best suited for open development
- **None** - not recommended for development in current form

## 🎯 Key Features

### 🔍 Analysis and Classification
- **Hybrid ML approach**: combination of machine learning and rules
- **Patent analysis**: evaluation of patenting potential
- **AI analysis**: technical complexity, market potential, innovation level
- **Detailed reports**: structured analysis with recommendations

### 📊 Data Sources
- **Patents** (6,002 records) - technical innovations
- **Scientific publications** (~5,000) - research data
- **Market signals** (~3,000) - market intelligence
- **Clinical trials** (~2,000) - medical development
- **License history** (~1,000) - licensing patterns
- **Internal disclosures** (~800) - internal innovations

### 🤖 AI Functions
- **Technical complexity assessment**: Low/Medium/High
- **Market potential analysis**: Low/Medium/High
- **Innovation level classification**: Incremental/Moderate/Revolutionary
- **AI recommendations**: specific development advice

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd dataset-gen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For AI analysis (optional)
pip install openai
```

### 2. OpenAI API Setup (for AI analysis)

```bash
# Create file with API key
echo "your-openai-api-key-here" > openai_key.txt
```

### 3. Model Training (optional)

```bash
# Train main model
python train_final_model.py

# Train improved model
python train_simple_improved_model.py

# Train patent model
python train_patent_model.py
```

### 4. Run API

```bash
# Start server
python api.py
```

API will be available at: `http://localhost:5001`

### 5. Testing

```bash
# Automatic API tests
python test_api.py

# Patent model testing
python test_patent_ml.py

# Web interface
open web_interface.html
```

## 📁 Project Structure

```
dataset-gen/
├── api.py                              # Main REST API
├── main.py                             # Dataset generation
├── web_interface.html                  # Web interface
├── requirements.txt                    # Dependencies
├── openai_key.txt                     # OpenAI API key
│
├── Models/
│   ├── final_technology_evaluator.joblib           # Main model
│   ├── final_technology_evaluator_tfidf.joblib     # TF-IDF vectorizer
│   ├── final_technology_evaluator_features.json    # Feature names
│   ├── patent_potential_model.joblib               # Patent model
│   ├── patent_potential_tfidf.joblib               # TF-IDF for patents
│   └── patent_potential_features.json              # Patent model features
│
├── Training Scripts/
│   ├── train_final_model.py                       # Train main model
│   ├── train_simple_improved_model.py             # Train improved model
│   └── train_patent_model.py                      # Train patent model
│
├── Tests/
│   ├── test_api.py                                # API tests
│   ├── test_patent_ml.py                          # Patent model tests
│   └── test_patent_ai.py                          # AI function tests
│
└── data/                                          # Training data
    ├── train_final.jsonl                          # Main dataset
    ├── patents.csv                                # Patents
    ├── papers.csv                                 # Scientific publications
    ├── market_signals.csv                         # Market signals
    ├── clinical_trials.csv                        # Clinical trials
    ├── entities.csv                               # Companies
    ├── license_histories.csv                      # License history
    ├── internal_disclosures.csv                   # Internal disclosures
    ├── seed_docs.csv                              # Source documents
    ├── annotations_train.jsonl                    # Expert annotations
    └── candidates_train.jsonl                     # Training candidates
```

## 🔧 API Endpoints

### GET /health
API health check
```bash
curl http://localhost:5001/health
```

### GET /readiness
API readiness check
```bash
curl http://localhost:5001/readiness
```

### POST /evaluate
Evaluate single idea
```bash
curl -X POST http://localhost:5001/evaluate \
  -H "Content-Type: application/json" \
  -d '{"idea_text": "novel enzymatic step enabling scalable production in lab-on-a-chip"}'
```

### POST /batch_evaluate
Batch evaluation of multiple ideas
```bash
curl -X POST http://localhost:5001/batch_evaluate \
  -H "Content-Type: application/json" \
  -d '{"ideas": [{"idea_text": "idea 1"}, {"idea_text": "idea 2"}]}'
```

## 📊 API Response Structure

### Main `/evaluate` Response

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

## 🏗️ Technical Architecture

### Machine Learning System

**Main Model:**
- **Algorithms**: RandomForest, GradientBoosting, LogisticRegression
- **Ensemble**: VotingClassifier with soft voting
- **Features**: 500+ features (TF-IDF + engineered features)
- **Accuracy**: ~85% on test set

**Patent Model:**
- **Algorithms**: RandomForest, GradientBoosting, LogisticRegression
- **Ensemble**: VotingClassifier with soft voting
- **Features**: 230+ features (specialized patent features)
- **Accuracy**: ~58% on test set

**Hybrid Approach:**
- ML model used when confidence > 25%
- Rules used when ML confidence is low
- AI analysis complements ML recommendations

### Data Processing

**Feature Extraction:**
- Text analysis (length, complexity, uniqueness)
- Keyword scoring (commercial, platform, free)
- Technical terms and patterns
- Evidence analysis (patents, publications, market)

**TF-IDF Vectorization:**
- Maximum 500 features for main model
- Maximum 300 features for patent model
- N-grams 1-3 for main model, 1-2 for patent
- English stop words excluded

## 🎯 Target Users

- **🔬 Researchers** — evaluate commercial potential of their ideas
- **💰 Investors** — filter promising projects for investment
- **🚀 Startups** — choose development direction and strategy
- **🏢 Corporations** — analyze innovation portfolio and R&D planning
- **⚖️ Patent Attorneys** — preliminary patent potential assessment

## 📈 Performance

### System Metrics
- **Response time**: < 2 seconds per request
- **Throughput**: 100+ requests/minute
- **Classification accuracy**: 85% for main model
- **Coverage**: 4 main categories + patent analysis

### Scalability
- **Batch processing**: up to 1000 ideas simultaneously
- **Horizontal scaling**: load balancing support
- **Caching**: TF-IDF vectorization cached
- **Optimization**: lazy model loading

## 🔧 Development

### Adding New Features

```python
def extract_custom_features(text):
    features = {}
    # Add your features here
    features['custom_score'] = calculate_custom_score(text)
    return features
```

### Training New Model

```python
# 1. Prepare data in train_final.jsonl format
# 2. Modify extract_features in train_final_model.py
# 3. Run training
python train_final_model.py
```

### Testing Changes

```bash
# API testing
python test_api.py

# Patent model testing
python test_patent_ml.py

# Model quality check
python -c "
import joblib
model = joblib.load('final_technology_evaluator.joblib')
print('Model loaded successfully')
"
```

## 📚 Additional Resources

- **Web Interface**: `web_interface.html` - interactive testing
- **Request Examples**: see `test_api.py`
- **API Documentation**: built into code with examples
- **Logs**: detailed logging of all operations

## 🤝 Support

For questions and suggestions:
1. Check existing issues
2. Create new issue with problem description
3. Attach request and response examples
4. Specify system version and environment

## 📄 License

Project is distributed under MIT license. See LICENSE file for details.

---

**Technology Evaluator** - automate technology idea evaluation with AI and machine learning! 🚀