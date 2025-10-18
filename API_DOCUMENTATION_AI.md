# 🔌 API Documentation with AI Analysis

## Overview

The Technology Evaluator API now includes advanced AI analysis capabilities powered by OpenAI's GPT-3.5-turbo model, providing comprehensive technology evaluation beyond traditional machine learning predictions.

## Base URL
```
http://localhost:5001
```

## Authentication
No authentication required for local development.

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and the model is loaded.

#### Response
```json
{
  "message": "Technology Evaluator API is running",
  "model_loaded": true,
  "status": "healthy"
}
```

### 2. Readiness Check
**GET** `/readiness`

Check if the API is ready to process requests.

#### Response
```json
{
  "message": "API is ready to process requests",
  "model_loaded": true,
  "ai_enabled": true,
  "status": "ready"
}
```

### 3. Evaluate Single Idea
**POST** `/evaluate`

Evaluate a single technology idea with AI analysis.

#### Request Body
```json
{
  "idea_text": "AI-powered drug discovery platform using machine learning",
  "evidence": [
    {
      "type": "patent",
      "text": "Related patent information",
      "meta": {
        "patent_id": "US123456789",
        "title": "Machine Learning for Drug Discovery"
      }
    },
    {
      "type": "market",
      "text": "Market signal about AI in healthcare",
      "meta": {
        "signal_id": "MS001",
        "type": "Product launch",
        "importance_score": 4.0
      }
    }
  ]
}
```

#### Response
```json
{
  "idea_text": "AI-powered drug discovery platform using machine learning",
  "analysis_summary": {
    "prediction": "commercial",
    "confidence": 0.752,
    "method": "ml_high_confidence",
    "evidence_count": 2,
    "evidence_types": {
      "patent": 1,
      "market": 1
    },
    "insights": [
      "Existing patents in the field",
      "Product launch detected in related area"
    ]
  },
  "predictions": {
    "ml_model": {
      "prediction": "commercial",
      "confidence": 0.752,
      "probabilities": {
        "commercial": 0.752,
        "platform": 0.156,
        "free": 0.067,
        "none": 0.025
      }
    },
    "rule_based": {
      "prediction": "commercial"
    },
    "hybrid": {
      "prediction": "commercial",
      "confidence_level": "high",
      "method": "ml_high_confidence"
    }
  },
  "detailed_features": {
    "text_analysis": {
      "text_length": 95,
      "word_count": 12,
      "sentence_count": 1,
      "avg_word_length": 7.0,
      "unique_word_ratio": 1.0,
      "complexity_score": 4
    },
    "keyword_scores": {
      "commercial_score": 3,
      "platform_score": 2,
      "free_score": 0,
      "technical_terms": 2,
      "scientific_terms": 1,
      "business_terms": 1
    },
    "evidence_analysis": {
      "total_evidence": 2,
      "patents": 1,
      "papers": 0,
      "market_signals": 1,
      "clinical_trials": 0,
      "disclosures": 0
    },
    "market_indicators": {
      "product_launches": 1,
      "mergers_acquisitions": 0,
      "partnerships": 0,
      "funding_activity": 0,
      "max_importance_score": 4.0,
      "avg_importance_score": 4.0
    },
    "clinical_indicators": {
      "completed_trials": 0,
      "phase3_trials": 0,
      "terminated_trials": 0,
      "max_maturity_score": 0.0,
      "avg_maturity_score": 0.0
    },
    "pattern_analysis": {
      "has_numbers": 0,
      "has_measurements": 0,
      "has_comparisons": 0,
      "has_commercial_patterns": 1,
      "has_platform_patterns": 0,
      "has_free_patterns": 0
    }
  },
  "recommendations": [
    {
      "type": "commercial_opportunity",
      "priority": "high",
      "title": "Commercial Potential",
      "message": "This idea has high commercial potential for licensing and monetization",
      "confidence": "high",
      "actions": [
        "Conduct patent search and Freedom to Operate (FTO) analysis",
        "Assess market size and competitive landscape",
        "Develop licensing strategy",
        "Consider partnerships with major companies",
        "Prepare commercialization business plan"
      ],
      "timeline": "3-6 months",
      "investment": "Medium-High",
      "roi_potential": "High"
    }
  ],
  "ai_analysis": {
    "technical_complexity": "High",
    "market_potential": "High",
    "innovation_level": "Revolutionary",
    "technical_challenges": [
      "Data quality and availability for training AI models",
      "Integration with existing drug discovery workflows",
      "Regulatory compliance and validation requirements"
    ],
    "market_opportunities": [
      "Large pharmaceutical companies seeking AI solutions",
      "Biotech startups looking for competitive advantages",
      "Academic institutions for research collaboration"
    ],
    "ai_recommendations": [
      "Invest in high-quality data collection and curation for training AI models",
      "Collaborate with domain experts in drug discovery to ensure AI-generated insights are actionable",
      "Continuously validate and improve AI algorithms through feedback loops with experimental results"
    ],
    "ai_insights": [
      "Technical complexity: High",
      "Market potential: High",
      "Innovation level: Revolutionary"
    ]
  },
  "metadata": {
    "timestamp": "2025-10-19T00:00:00.000Z",
    "model_version": "final_technology_evaluator",
    "api_version": "1.0.0",
    "ai_enabled": true,
    "processing_time_ms": 1250
  }
}
```

### 4. Batch Evaluation
**POST** `/batch_evaluate`

Evaluate multiple technology ideas in a single request.

#### Request Body
```json
{
  "ideas": [
    {
      "id": "idea_1",
      "idea_text": "AI-powered drug discovery platform",
      "evidence": []
    },
    {
      "id": "idea_2", 
      "idea_text": "Open source data analysis framework",
      "evidence": []
    }
  ]
}
```

#### Response
```json
{
  "results": [
    {
      "id": "idea_1",
      "analysis_summary": {
        "prediction": "commercial",
        "confidence": 0.752
      },
      "ai_analysis": {
        "technical_complexity": "High",
        "market_potential": "High",
        "innovation_level": "Revolutionary"
      }
    },
    {
      "id": "idea_2",
      "analysis_summary": {
        "prediction": "free",
        "confidence": 0.634
      },
      "ai_analysis": {
        "technical_complexity": "Medium",
        "market_potential": "High",
        "innovation_level": "Incremental"
      }
    }
  ],
  "summary": {
    "total_processed": 2,
    "successful": 2,
    "failed": 0,
    "processing_time_ms": 2100
  }
}
```

## AI Analysis Fields

### Technical Complexity
- **Low**: Simple implementation, minimal technical requirements
- **Medium**: Moderate complexity, requires some expertise  
- **High**: Complex implementation, requires significant technical expertise

### Market Potential
- **Low**: Limited market opportunity
- **Medium**: Moderate market potential
- **High**: Significant market opportunity

### Innovation Level
- **Incremental**: Small improvements to existing solutions
- **Moderate**: Notable advances in existing technology
- **Revolutionary**: Breakthrough innovations that could change the industry

## Error Responses

### 400 Bad Request
```json
{
  "error": "Missing required field: idea_text"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error: AI analysis temporarily unavailable"
}
```

### AI Analysis Errors
When AI analysis fails, the response includes:
```json
{
  "ai_analysis": {
    "technical_complexity": "Unknown",
    "market_potential": "Unknown", 
    "innovation_level": "Unknown",
    "ai_recommendations": [],
    "ai_insights": ["AI analysis temporarily unavailable"]
  },
  "metadata": {
    "ai_enabled": false
  }
}
```

## Rate Limiting

### OpenAI API Limits
- **Requests per minute**: 60 (default)
- **Tokens per minute**: 90,000 (default)
- **Daily usage limit**: Based on billing plan

### Recommendations
- Implement client-side rate limiting
- Cache responses for similar requests
- Use batch evaluation for multiple ideas

## Response Times

### Typical Performance
- **ML Prediction**: 50-100ms
- **AI Analysis**: 1-3 seconds
- **Total Response**: 1.5-4 seconds

### Optimization Tips
- Use batch evaluation for multiple ideas
- Cache frequently requested analyses
- Implement request queuing for high volume

## Testing

### cURL Examples

#### Basic Evaluation
```bash
curl -X POST http://localhost:5001/evaluate \
  -H "Content-Type: application/json" \
  -d '{"idea_text": "AI-powered drug discovery platform"}'
```

#### With Evidence
```bash
curl -X POST http://localhost:5001/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "idea_text": "Machine learning algorithm for drug discovery",
    "evidence": [
      {
        "type": "patent",
        "text": "Related patent information",
        "meta": {"patent_id": "US123456"}
      }
    ]
  }'
```

#### Batch Evaluation
```bash
curl -X POST http://localhost:5001/batch_evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "ideas": [
      {"id": "1", "idea_text": "AI drug discovery"},
      {"id": "2", "idea_text": "Open source framework"}
    ]
  }'
```

### Python Examples

```python
import requests

# Basic evaluation
response = requests.post('http://localhost:5001/evaluate', json={
    'idea_text': 'AI-powered drug discovery platform'
})

result = response.json()
print(f"Prediction: {result['analysis_summary']['prediction']}")
print(f"AI Complexity: {result['ai_analysis']['technical_complexity']}")

# With evidence
response = requests.post('http://localhost:5001/evaluate', json={
    'idea_text': 'Machine learning algorithm for drug discovery',
    'evidence': [
        {
            'type': 'patent',
            'text': 'Related patent information',
            'meta': {'patent_id': 'US123456'}
        }
    ]
})
```

## Monitoring and Logging

### Health Monitoring
- **Endpoint**: `/health` - Basic health check
- **Readiness**: `/readiness` - Full readiness check including AI
- **Metrics**: Response times, success rates, error rates

### Logging
- **API Requests**: All incoming requests logged
- **AI Analysis**: Success/failure of AI analysis
- **Performance**: Response times and resource usage
- **Errors**: Detailed error logging with stack traces

### Debugging
Enable debug mode by setting environment variable:
```bash
export FLASK_DEBUG=1
```

## Security Considerations

### API Key Protection
- Store OpenAI API key in secure file (`openai_key.txt`)
- Add `openai_key.txt` to `.gitignore`
- Use environment variables in production

### Input Validation
- Validate JSON structure
- Sanitize text inputs
- Limit request size and complexity

### Rate Limiting
- Implement client-side rate limiting
- Monitor API usage patterns
- Set appropriate timeouts
