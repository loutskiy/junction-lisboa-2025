#!/usr/bin/env python3
"""
Test script for patent recommendation AI analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import api

def test_patent_analysis():
    print("🧪 Testing Patent Recommendation AI Analysis")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "AI-powered drug discovery platform using machine learning",
        "Simple mobile app for tracking water intake",
        "Blockchain-based supply chain tracking system",
        "Open source framework for data analysis"
    ]
    
    for i, idea in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {idea}")
        print("-" * 40)
        
        try:
            result = api.ai_analyze_idea(idea, [])
            
            print(f"✅ Technical Complexity: {result.get('technical_complexity', 'Unknown')}")
            print(f"✅ Market Potential: {result.get('market_potential', 'Unknown')}")
            print(f"✅ Innovation Level: {result.get('innovation_level', 'Unknown')}")
            print(f"✅ Patent Recommendation: {result.get('patent_recommendation', 'Unknown')}")
            print(f"✅ Patent Confidence: {result.get('patent_confidence', 'Unknown')}")
            
            if result.get('patent_reasons'):
                print("📋 Patent Reasons:")
                for reason in result['patent_reasons'][:3]:
                    print(f"  • {reason}")
            
            if result.get('ip_strategy'):
                print("💡 IP Strategy:")
                for strategy in result['ip_strategy'][:3]:
                    print(f"  • {strategy}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Patent recommendation testing completed!")

if __name__ == "__main__":
    test_patent_analysis()
