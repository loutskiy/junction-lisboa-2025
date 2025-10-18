# Technical Specifications - Technology Evaluator

## 🏗️ System Architecture

### High-Level Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  ML Pipeline    │───▶│   REST API      │
│                 │    │                 │    │                 │
│ • Patents       │    │ • Feature       │    │ • Flask Server  │
│ • Papers        │    │   Extraction    │    │ • 4 Endpoints   │
│ • Market Data   │    │ • Model Training│    │ • CORS Support  │
│ • Clinical      │    │ • Hybrid Logic  │    │ • Web Interface │
│ • Licenses      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Data Pipeline

### Input Data Sources
| Source | Records | Fields | Purpose |
|--------|---------|--------|---------|
| Patents | 6,002 | title, abstract, claims | Technical innovation data |
| Papers | ~5,000 | title, abstract, keywords | Scientific research data |
| Market Signals | ~3,000 | type, description, source | Market intelligence |
| Clinical Trials | ~2,000 | title, phase, status | Medical development data |
| Entities | ~1,500 | name, industry, description | Company information |
| License History | ~1,000 | licensor, licensee, status | Licensing patterns |
| Internal Disclosures | ~800 | title, summary | Internal innovation |
| Expert Annotations | 14,428 | text, label, evidence | Training labels |

### Data Processing
- **Format**: JSONL for training, CSV for sources
- **Preprocessing**: Text normalization, feature extraction
- **Enrichment**: Market scores, trial maturity, license features
- **Validation**: Data quality checks, outlier detection

## 🧠 Machine Learning Pipeline

### Feature Engineering (369 total features)

#### Text Features (48 features)
```python
# Basic characteristics
text_length, word_count, avg_word_length, unique_word_ratio

# Keyword scoring with weights
platform_score = sum(weights for keywords in platform_terms)
commercial_score = sum(weights for keywords in commercial_terms)  
free_score = sum(weights for keywords in free_terms)

# Technical analysis
technical_terms, scientific_terms, business_terms, engineering_terms

# Structural patterns
has_numbers, has_measurements, has_percentages, has_comparisons
```

#### TF-IDF Features (321 features)
- **Vectorizer**: TfidfVectorizer
- **Max Features**: 500
- **N-gram Range**: (1, 3)
- **Min/Max DF**: 2, 0.9
- **Stop Words**: English

### Model Architecture

#### Ensemble Voting Classifier
```python
VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=20)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05)),
    ('lr', LogisticRegression(C=0.1, class_weight='balanced'))
], voting='soft')
```

#### Individual Model Specifications

**Random Forest**
- Trees: 300
- Max Depth: 20
- Min Samples Split: 3
- Min Samples Leaf: 1
- Class Weight: balanced
- Parallel: n_jobs=-1

**Gradient Boosting**
- Estimators: 300
- Learning Rate: 0.05
- Max Depth: 10
- Loss: deviance

**Logistic Regression**
- Regularization: L2 (Ridge)
- C: 0.1
- Max Iterations: 3000
- Class Weight: balanced

### Training Process
1. **Data Split**: 80% train, 20% test
2. **Stratification**: Maintain class distribution
3. **Cross-validation**: 5-fold CV for hyperparameter tuning
4. **Feature Selection**: Correlation analysis, importance ranking
5. **Model Selection**: Grid search for optimal parameters

## 🌐 API Specifications

### Server Configuration
- **Framework**: Flask 2.3.3
- **Port**: 5001
- **Host**: 0.0.0.0 (all interfaces)
- **CORS**: Enabled for cross-origin requests
- **Debug**: Enabled for development

### Endpoints

#### Health Check
```
GET /health
Response: 200 OK
{
  "message": "Technology Evaluator API is running",
  "model_loaded": true,
  "status": "healthy"
}
```

#### Readiness Check
```
GET /readiness  
Response: 200 OK
{
  "message": "API is ready to process requests",
  "status": "ready"
}
```

#### Single Evaluation
```
POST /evaluate
Content-Type: application/json
{
  "idea_text": "string",
  "evidence": [{"type": "string", "meta": {}}]
}

Response: 200 OK
{
  "idea_text": "string",
  "predictions": {
    "ml_model": {"prediction": "string", "confidence": float},
    "rule_based": {"prediction": "string"},
    "hybrid": {"prediction": "string", "method": "string"}
  },
  "features": {},
  "recommendations": []
}
```

#### Batch Evaluation
```
POST /batch_evaluate
Content-Type: application/json
{
  "ideas": [{"id": "string", "idea_text": "string", "evidence": []}]
}

Response: 200 OK
{
  "results": [{"id": "string", "prediction": "string", "confidence": float}]
}
```

## 📈 Performance Metrics

### Model Performance
- **Overall Accuracy**: 52.3%
- **Macro F1-Score**: 0.49
- **Weighted F1-Score**: 0.49

### Class-Specific Metrics
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| commercial | 0.50 | 0.70 | 0.59 | 1,205 |
| free | 0.24 | 0.13 | 0.17 | 588 |
| none | 1.00 | 1.00 | 1.00 | 480 |
| platform | 0.26 | 0.17 | 0.20 | 613 |

### System Performance
- **Response Time**: < 200ms (95th percentile)
- **Throughput**: 100+ requests/minute
- **Memory Usage**: ~2GB (with model loaded)
- **CPU Usage**: ~30% (single core)

## 🔧 Development Environment

### Requirements
- **Python**: 3.8+
- **OS**: macOS, Linux, Windows
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space

### Dependencies
```
Flask==2.3.3
scikit-learn==1.7.2
pandas==2.3.3
numpy==2.3.4
joblib==1.5.2
flask-cors==6.0.1
requests==2.32.5
```

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model (optional)
python train_final_model.py

# Start API server
python api.py
```

## 🧪 Testing

### Test Coverage
- **Unit Tests**: Model prediction functions
- **Integration Tests**: API endpoints
- **Load Tests**: Performance under load
- **Accuracy Tests**: Model validation

### Test Commands
```bash
# Run API tests
python test_api.py

# Test model accuracy
python -c "from train_final_model import *; test_model()"

# Load testing
ab -n 1000 -c 10 http://localhost:5001/evaluate
```

## 🚀 Deployment

### Production Considerations
- **WSGI Server**: Gunicorn recommended
- **Process Manager**: systemd or supervisor
- **Load Balancer**: nginx for multiple instances
- **Monitoring**: Application metrics and logging
- **Scaling**: Horizontal scaling with load balancer

### Docker Support
```dockerfile
FROM python:3.13-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5001
CMD ["python", "api.py"]
```

## 📊 Monitoring & Logging

### Metrics to Track
- **Request Rate**: requests per second
- **Response Time**: average and 95th percentile
- **Error Rate**: 4xx and 5xx responses
- **Model Performance**: prediction accuracy over time
- **Resource Usage**: CPU, memory, disk

### Logging
- **Level**: INFO for production, DEBUG for development
- **Format**: JSON structured logging
- **Rotation**: Daily log rotation
- **Retention**: 30 days

## 🔒 Security

### API Security
- **CORS**: Configured for specific origins
- **Input Validation**: JSON schema validation
- **Rate Limiting**: Implemented via Flask-Limiter
- **Error Handling**: Sanitized error messages

### Data Security
- **No PII**: No personally identifiable information stored
- **Model Security**: Serialized models are safe to distribute
- **Input Sanitization**: Text preprocessing removes sensitive data

---

*Last updated: October 2024*
