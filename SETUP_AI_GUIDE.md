# 🚀 Setup Guide with AI Integration

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Required for OpenAI API access

### Required Accounts
- **OpenAI Account**: For AI analysis capabilities
- **Git**: For version control (optional)

## Installation Steps

### 1. Clone or Download Project
```bash
git clone <repository-url>
cd dataset-gen
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install openai  # Additional AI dependency
```

### 4. Set Up OpenAI API Key

#### Option A: File-based (Recommended for development)
Create `openai_key.txt` in the project root:
```bash
echo "sk-proj-your-api-key-here" > openai_key.txt
```

#### Option B: Environment Variable (Recommended for production)
```bash
export OPENAI_API_KEY="sk-proj-your-api-key-here"
```

### 5. Verify Installation
```bash
python -c "import openai; print('OpenAI installed successfully')"
```

## Configuration

### API Key Setup

#### Getting OpenAI API Key
1. Visit [platform.openai.com](https://platform.openai.com)
2. Sign up or log in to your account
3. Go to "API Keys" section
4. Click "Create new secret key"
5. Copy the key (starts with `sk-proj-`)

#### Security Best Practices
- **Never commit API keys to version control**
- **Use environment variables in production**
- **Set up billing alerts in OpenAI dashboard**
- **Regularly rotate API keys**

### File Structure
```
dataset-gen/
├── api.py                          # Main API server
├── web_interface.html              # Web interface
├── openai_key.txt                  # API key (DO NOT COMMIT)
├── .gitignore                      # Git ignore file
├── requirements.txt                # Python dependencies
├── data/                          # Training data
│   ├── annotations_train.jsonl
│   ├── candidates_train.jsonl
│   └── ...
├── final_technology_evaluator.joblib      # ML model
├── final_technology_evaluator_tfidf.joblib # TF-IDF vectorizer
└── docs/                          # Documentation
    ├── AI_INTEGRATION.md
    ├── WEB_INTERFACE_AI.md
    └── API_DOCUMENTATION_AI.md
```

## Running the Application

### 1. Start the API Server
```bash
source venv/bin/activate
python api.py
```

Expected output:
```
🚀 Запуск Technology Evaluator API...
==================================================
📦 Загрузка модели...
✅ Модель загружена успешно
🌐 API доступен по адресу: http://localhost:5001
📚 Документация API:
  GET  /health - проверка здоровья
  GET  /readiness - проверка готовности
  POST /evaluate - оценка одной идеи
  POST /batch_evaluate - пакетная оценка
 * Serving Flask app 'api'
 * Debug mode: on
 * Running on http://127.0.0.1:5001
```

### 2. Test the API
```bash
curl http://localhost:5001/health
```

Expected response:
```json
{
  "message": "Technology Evaluator API is running",
  "model_loaded": true,
  "status": "healthy"
}
```

### 3. Open Web Interface
Open `web_interface.html` in your web browser.

## Testing AI Integration

### Basic Test
```bash
curl -X POST http://localhost:5001/evaluate \
  -H "Content-Type: application/json" \
  -d '{"idea_text": "AI-powered drug discovery platform"}'
```

### Expected AI Response
Look for `ai_analysis` section in the response:
```json
{
  "ai_analysis": {
    "technical_complexity": "High",
    "market_potential": "High",
    "innovation_level": "Revolutionary",
    "ai_recommendations": [
      "Invest in high-quality data collection...",
      "Collaborate with domain experts...",
      "Continuously validate AI algorithms..."
    ]
  }
}
```

## Troubleshooting

### Common Issues

#### 1. "AI_AVAILABLE: False"
**Problem**: AI integration not working
**Solutions**:
- Check if `openai` package is installed: `pip list | grep openai`
- Verify API key file exists: `ls -la openai_key.txt`
- Check API key format (should start with `sk-proj-`)

#### 2. "You exceeded your current quota"
**Problem**: OpenAI API quota exceeded
**Solutions**:
- Check billing status at [platform.openai.com](https://platform.openai.com)
- Add payment method if needed
- Check usage limits in OpenAI dashboard

#### 3. "AI analysis temporarily unavailable"
**Problem**: AI analysis failing
**Solutions**:
- Check internet connectivity
- Verify OpenAI API status
- Check API key validity
- Review error logs in console

#### 4. "Address already in use"
**Problem**: Port 5001 is occupied
**Solutions**:
```bash
# Find process using port 5001
lsof -ti:5001

# Kill the process
lsof -ti:5001 | xargs kill -9

# Or use a different port
export PORT=5002
python api.py
```

### Debug Mode

#### Enable Detailed Logging
Add to `api.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check AI Integration Status
```bash
python -c "
import api
print('AI Available:', api.AI_AVAILABLE)
print('Client created:', api.client is not None)
"
```

## Production Deployment

### Environment Variables
```bash
export OPENAI_API_KEY="your-production-key"
export FLASK_ENV="production"
export PORT=5001
```

### Security Considerations
- Use environment variables for API keys
- Implement proper authentication
- Set up rate limiting
- Use HTTPS in production
- Monitor API usage and costs

### Performance Optimization
- Use production WSGI server (Gunicorn)
- Implement response caching
- Set up load balancing
- Monitor resource usage

## Cost Management

### OpenAI API Pricing
- **GPT-3.5-turbo**: ~$0.001-0.002 per request
- **Typical usage**: $1-5 per month for development
- **Production**: Set spending limits

### Cost Optimization
- Cache similar requests
- Use batch evaluation
- Implement request queuing
- Monitor usage patterns

### Setting Spending Limits
1. Go to [platform.openai.com](https://platform.openai.com)
2. Navigate to "Billing" → "Usage limits"
3. Set monthly spending limit
4. Enable email alerts

## Monitoring and Maintenance

### Health Checks
```bash
# Basic health check
curl http://localhost:5001/health

# Full readiness check
curl http://localhost:5001/readiness
```

### Log Monitoring
- Monitor API response times
- Track AI analysis success rates
- Watch for error patterns
- Monitor OpenAI API usage

### Regular Maintenance
- Update dependencies monthly
- Rotate API keys quarterly
- Review and optimize prompts
- Monitor cost trends

## Support and Resources

### Documentation
- [AI Integration Guide](AI_INTEGRATION.md)
- [Web Interface Documentation](WEB_INTERFACE_AI.md)
- [API Documentation](API_DOCUMENTATION_AI.md)

### External Resources
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Machine Learning Model Documentation](README.md)

### Getting Help
1. Check this documentation first
2. Review error logs and console output
3. Test with simple examples
4. Verify all dependencies are installed
5. Check OpenAI API status page
