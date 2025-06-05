#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_clarity_improvements.py - Fixed version with proper imports
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# CRITICAL FIX: Add scripts directory to Python path
sys.path.insert(0, 'scripts')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('validate_clarity')

def validate_improvements():
    """Validasi peningkatan clarity dengan sample texts - FIXED VERSION"""
    
    print("🧪 VALIDATING CLARITY IMPROVEMENTS")
    print("="*50)
    
    # Check if enhanced_clarity_framework exists
    framework_path = os.path.join('scripts', 'enhanced_clarity_framework.py')
    if not os.path.exists(framework_path):
        print("❌ ERROR: enhanced_clarity_framework.py not found in scripts/")
        print("📝 SOLUTION: Please ensure the framework file exists in scripts/ directory")
        
        # Create minimal framework if it doesn't exist
        print("🔧 Creating minimal framework for testing...")
        create_minimal_framework()
    
    try:
        # Import with better error handling
        print("📦 Importing enhanced framework...")
        from enhanced_clarity_framework import (
            EnhancedBankSentralLexicon, 
            ContextSensitivePreprocessor, 
            EnhancedClarityCalculator
        )
        print("✅ Framework imported successfully")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("🔧 Using fallback minimal framework...")
        
        # Use inline minimal framework as fallback
        return validate_with_fallback_framework()
    
    # Sample texts untuk testing
    test_samples = [
        {
            'id': 'sample_1',
            'text': """Bank Indonesia mempertahankan BI 7-Day Reverse Repo Rate pada level 6,00%. 
                      Keputusan ini sejalan dengan proyeksi inflasi yang diperkirakan akan tetap 
                      terkendali dalam kisaran sasaran 3,0±1% pada tahun 2025.""",
            'expected_clarity': 'medium',
            'complexity_level': 'standard'
        },
        {
            'id': 'sample_2', 
            'text': """Implementasi kebijakan makroprudensial melalui penerapan rasio loan-to-value 
                      dan debt-service-ratio bertujuan untuk memitigasi risiko sistemik dalam 
                      sistem perbankan dengan tetap menjaga intermediasi yang optimal.""",
            'expected_clarity': 'low',
            'complexity_level': 'high'
        },
        {
            'id': 'sample_3',
            'text': """Bank Indonesia akan terus menjaga stabilitas nilai tukar rupiah. 
                      Kebijakan ini penting untuk mendukung perekonomian nasional.""",
            'expected_clarity': 'high', 
            'complexity_level': 'low'
        },
        {
            'id': 'sample_4',
            'text': """Inflasi terkendali sesuai target yang ditetapkan oleh Bank Indonesia.""",
            'expected_clarity': 'high',
            'complexity_level': 'low'
        },
        {
            'id': 'sample_5',
            'text': """Transmisi kebijakan moneter melalui saluran suku bunga, kredit, dan nilai tukar 
                      berfungsi secara efektif dalam mendukung pencapaian sasaran inflasi dan 
                      pertumbuhan ekonomi yang berkelanjutan.""",
            'expected_clarity': 'medium',
            'complexity_level': 'standard'
        }
    ]
    
    # Initialize enhanced components
    print("⚡ Initializing enhanced components...")
    lexicon = EnhancedBankSentralLexicon()
    preprocessor = ContextSensitivePreprocessor(lexicon)
    calculator = EnhancedClarityCalculator(lexicon)
    print(f"✅ Components initialized with {len(lexicon.get_all_terms())} economic terms")
    
    validation_results = []
    
    print("\n🔍 Processing validation samples...")
    
    for sample in test_samples:
        print(f"\n📝 Processing {sample['id']}...")
        
        # Process text
        cleaned = preprocessor.smart_text_cleaning(sample['text'])
        filtered = preprocessor.context_aware_stopword_removal(cleaned)
        
        # Calculate clarity
        metrics = calculator.calculate_enhanced_clarity(filtered)
        
        # Evaluate against expectations
        clarity_score = metrics['composite_clarity_score'] * 100
        
        if sample['expected_clarity'] == 'high':
            target_min = 70
        elif sample['expected_clarity'] == 'medium':
            target_min = 50
        else:
            target_min = 30
        
        meets_expectation = clarity_score >= target_min
        
        result = {
            'sample_id': sample['id'],
            'original_text': sample['text'][:100] + '...',
            'processed_text': filtered[:100] + '...',
            'expected_clarity': sample['expected_clarity'],
            'actual_score': clarity_score,
            'target_minimum': target_min,
            'meets_expectation': meets_expectation,
            'sentence_length': metrics['avg_sentence_length'],
            'technical_density': metrics['technical_density'],
            'readability': metrics['contextual_readability'],
            'word_count': metrics['word_count']
        }
        
        validation_results.append(result)
        
        status = "✅ PASS" if meets_expectation else "❌ FAIL"
        print(f"  {status} Score: {clarity_score:.1f} (Target: ≥{target_min})")
    
    # Create validation report and visualization
    print("\n📊 Creating validation analysis...")
    create_validation_analysis(validation_results)
    
    # Generate summary
    total_samples = len(validation_results)
    passed_samples = sum(1 for r in validation_results if r['meets_expectation'])
    success_rate = (passed_samples / total_samples) * 100
    
    print(f"\n📈 VALIDATION SUMMARY:")
    print(f"  Total Samples: {total_samples}")
    print(f"  Passed: {passed_samples}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🏆 EXCELLENT: Validation highly successful!")
    elif success_rate >= 60:
        print("✅ GOOD: Validation successful with room for improvement")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Consider parameter tuning")
    
    return validation_results

def validate_with_fallback_framework():
    """Fallback validation using simple metrics"""
    
    print("🔧 Using fallback validation framework...")
    
    test_samples = [
        {
            'id': 'fallback_1',
            'text': "Bank Indonesia mempertahankan suku bunga acuan pada level 6 persen.",
            'expected_clarity': 'high'
        },
        {
            'id': 'fallback_2', 
            'text': "Implementasi kebijakan makroprudensial melalui penerapan rasio loan-to-value dan debt-service-ratio untuk mitigasi risiko sistemik perbankan.",
            'expected_clarity': 'low'
        },
        {
            'id': 'fallback_3',
            'text': "Inflasi terkendali sesuai target.",
            'expected_clarity': 'high'
        }
    ]
    
    results = []
    
    for sample in test_samples:
        text = sample['text']
        words = text.split()
        sentences = len([s for s in text.split('.') if s.strip()])
        
        # Simple metrics
        avg_sentence_length = len(words) / max(sentences, 1)
        word_complexity = np.mean([len(word) for word in words])
        
        # Simple clarity score
        clarity_score = max(0, 100 - (avg_sentence_length * 2 + word_complexity * 5))
        
        target = 70 if sample['expected_clarity'] == 'high' else 30
        meets_expectation = clarity_score >= target
        
        results.append({
            'sample_id': sample['id'],
            'original_text': text,
            'actual_score': clarity_score,
            'target_minimum': target,
            'meets_expectation': meets_expectation,
            'sentence_length': avg_sentence_length,
            'word_complexity': word_complexity
        })
        
        status = "✅ PASS" if meets_expectation else "❌ FAIL"
        print(f"{sample['id']}: {status} Score: {clarity_score:.1f}")
    
    passed = sum(1 for r in results if r['meets_expectation'])
    success_rate = (passed / len(results)) * 100
    
    print(f"\nFallback Validation Success Rate: {success_rate:.1f}%")
    
    return results

def create_minimal_framework():
    """Create minimal framework file if it doesn't exist"""
    
    os.makedirs('scripts', exist_ok=True)
    
    minimal_content = '''# Minimal Enhanced Clarity Framework
import re
import numpy as np

class EnhancedBankSentralLexicon:
    def __init__(self):
        self.economic_terms = {
            'bank', 'sentral', 'moneter', 'kebijakan', 'inflasi', 'suku', 'bunga',
            'repo', 'kredit', 'nilai', 'tukar', 'rupiah', 'ekonomi', 'pertumbuhan'
        }
        self.context_sensitive_stopwords = {
            'preserve': {'dapat', 'akan', 'perlu', 'harus'},
            'safe_remove': {'itu', 'ini', 'tersebut', 'adalah'}
        }
    
    def get_all_terms(self):
        return self.economic_terms

class ContextSensitivePreprocessor:
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.economic_terms = lexicon.get_all_terms()
    
    def smart_text_cleaning(self, text):
        if not text: return ""
        text = text.lower()
        text = re.sub(r'[^\\w\\s\\.]', ' ', text)
        return re.sub(r'\\s+', ' ', text).strip()
    
    def context_aware_stopword_removal(self, text):
        if not text: return ""
        words = text.split()
        preserve = self.lexicon.context_sensitive_stopwords['preserve']
        safe_remove = self.lexicon.context_sensitive_stopwords['safe_remove']
        
        filtered = []
        for word in words:
            if word in self.economic_terms or word in preserve:
                filtered.append(word)
            elif word not in safe_remove:
                filtered.append(word)
        return ' '.join(filtered)

class EnhancedClarityCalculator:
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.economic_terms = lexicon.get_all_terms()
    
    def calculate_enhanced_clarity(self, text):
        if not text: 
            return {'composite_clarity_score': 0, 'avg_sentence_length': 0, 
                   'technical_density': 0, 'contextual_readability': 0, 'word_count': 0}
        
        words = text.split()
        sentences = len([s for s in text.split('.') if s.strip()])
        
        if len(words) < 3:
            return {'composite_clarity_score': 0, 'avg_sentence_length': 0,
                   'technical_density': 0, 'contextual_readability': 0, 'word_count': len(words)}
        
        avg_sentence_length = len(words) / max(sentences, 1)
        tech_words = [w for w in words if w in self.economic_terms]
        technical_density = len(tech_words) / len(words)
        
        readability = max(0, 100 - (avg_sentence_length * 2 + len(max(words, key=len)) * 3))
        
        sentence_score = 1.0 if 10 <= avg_sentence_length <= 20 else 0.6
        tech_score = min(1.0, 0.5 + technical_density * 3)
        composite = (sentence_score + tech_score + readability/100) / 3
        
        return {
            'composite_clarity_score': composite,
            'avg_sentence_length': avg_sentence_length,
            'technical_density': technical_density,
            'contextual_readability': readability,
            'word_count': len(words)
        }
'''
    
    with open('scripts/enhanced_clarity_framework.py', 'w', encoding='utf-8') as f:
        f.write(minimal_content)
    
    print("✅ Created minimal framework file")

def create_validation_analysis(validation_results):
    """Create validation analysis and visualization"""
    
    if not validation_results:
        print("❌ No validation results to analyze")
        return
    
    # Ensure output directory exists
    os.makedirs('output/enhanced_clarity', exist_ok=True)
    
    # Save results to Excel
    validation_df = pd.DataFrame(validation_results)
    output_file = 'output/enhanced_clarity/validation_results.xlsx'
    validation_df.to_excel(output_file, index=False)
    print(f"✅ Validation results saved to: {output_file}")
    
    # Create visualization
    try:
        plt.figure(figsize=(15, 10))
        
        # Score comparison
        plt.subplot(2, 3, 1)
        sample_ids = [r['sample_id'] for r in validation_results]
        actual_scores = [r['actual_score'] for r in validation_results]
        target_mins = [r['target_minimum'] for r in validation_results]
        
        x = range(len(sample_ids))
        plt.bar([i-0.2 for i in x], actual_scores, 0.4, label='Actual Score', alpha=0.8, color='blue')
        plt.bar([i+0.2 for i in x], target_mins, 0.4, label='Target Minimum', alpha=0.8, color='red')
        plt.xlabel('Sample')
        plt.ylabel('Clarity Score')
        plt.title('Validation: Actual vs Target Scores')
        plt.xticks(x, sample_ids, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Success rate pie chart
        plt.subplot(2, 3, 2)
        success_count = sum(r['meets_expectation'] for r in validation_results)
        total_count = len(validation_results)
        success_rate = success_count / total_count * 100
        
        plt.pie([success_count, total_count - success_count], 
                labels=['Meets Expectation', 'Below Expectation'],
                colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        plt.title(f'Success Rate: {success_rate:.1f}%')
        
        # Sentence length analysis
        plt.subplot(2, 3, 3)
        sentence_lengths = [r['sentence_length'] for r in validation_results]
        colors = ['green' if r['meets_expectation'] else 'red' for r in validation_results]
        
        plt.scatter(range(len(sample_ids)), sentence_lengths, c=colors, alpha=0.7, s=100)
        plt.xlabel('Sample')
        plt.ylabel('Average Sentence Length')
        plt.title('Sentence Length by Sample')
        plt.xticks(range(len(sample_ids)), sample_ids, rotation=45)
        plt.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Optimal (~20)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Technical density analysis
        plt.subplot(2, 3, 4)
        technical_densities = [r.get('technical_density', 0) for r in validation_results]
        
        plt.scatter(range(len(sample_ids)), technical_densities, c=colors, alpha=0.7, s=100)
        plt.xlabel('Sample')
        plt.ylabel('Technical Density')
        plt.title('Technical Density by Sample')
        plt.xticks(range(len(sample_ids)), sample_ids, rotation=45)
        plt.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Optimal (~0.15)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Score distribution
        plt.subplot(2, 3, 5)
        plt.hist(actual_scores, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Clarity Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.axvline(np.mean(actual_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(actual_scores):.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Summary text
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        summary_text = f"""VALIDATION SUMMARY
        
Total Samples: {total_count}
Passed: {success_count}
Success Rate: {success_rate:.1f}%

Average Score: {np.mean(actual_scores):.1f}
Score Range: {min(actual_scores):.1f} - {max(actual_scores):.1f}

Status: {"✅ GOOD" if success_rate >= 60 else "⚠️ NEEDS WORK"}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = 'output/enhanced_clarity/validation_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Validation plot saved to: {plot_file}")
        
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
    
    # Generate text report
    report_content = f"""
CLARITY VALIDATION REPORT
=========================

Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Samples Tested: {len(validation_results)}

OVERALL PERFORMANCE:
- Samples Meeting Expectations: {sum(r['meets_expectation'] for r in validation_results)}
- Success Rate: {(sum(r['meets_expectation'] for r in validation_results) / len(validation_results)) * 100:.1f}%

DETAILED RESULTS:
"""
    
    for result in validation_results:
        status = "✅ PASS" if result['meets_expectation'] else "❌ FAIL"
        report_content += f"""
{result['sample_id']} - {status}:
  Expected: {result['expected_clarity']} clarity
  Score: {result['actual_score']:.1f} (target: ≥{result['target_minimum']})
  Sentence Length: {result['sentence_length']:.1f} words
  Technical Density: {result.get('technical_density', 0):.1%}
  Word Count: {result.get('word_count', 0)}
"""
    
    report_file = 'output/enhanced_clarity/validation_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Validation report saved to: {report_file}")

def main():
    """Main execution function"""
    try:
        print("🧪 STARTING CLARITY VALIDATION")
        print("="*50)
        
        results = validate_improvements()
        
        print("\n🎉 VALIDATION COMPLETED!")
        print("Check output/enhanced_clarity/ for detailed results")
        
        return results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()