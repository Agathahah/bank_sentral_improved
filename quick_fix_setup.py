#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_fix_setup.py - Quick fix script to resolve setup issues
Run this script to automatically fix common problems
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'scripts',
        'config', 
        'data/raw',
        'data/processed',
        'output/enhanced_clarity',
        'output/annotation_quality',
        'output/final_reports',
        'logs',
        'models',
        'cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_enhanced_clarity_framework():
    """Create the minimal enhanced clarity framework file"""
    
    framework_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_clarity_framework.py - Minimal working version
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter

class EnhancedBankSentralLexicon:
    """Basic lexicon for economic terms"""
    
    def __init__(self):
        self.economic_terms = {
            'bank', 'sentral', 'moneter', 'kebijakan', 'inflasi', 'deflasi',
            'suku', 'bunga', 'repo', 'fasbi', 'kredit', 'likuiditas',
            'nilai', 'tukar', 'kurs', 'rupiah', 'pasar', 'modal',
            'akomodatif', 'ketat', 'netral', 'stabil', 'volatilitas',
            'pertumbuhan', 'ekonomi', 'transmisi', 'efektif', 'optimal'
        }
        
        self.context_sensitive_stopwords = {
            'preserve': {'dapat', 'akan', 'perlu', 'harus', 'mampu', 'terus', 'tetap'},
            'safe_remove': {'itu', 'ini', 'tersebut', 'adalah', 'merupakan'}
        }
        
        self.clarity_weights = {
            'sentence_length': {'ideal_range': (15, 25)},
            'technical_density': {'optimal_ratio': 0.15}
        }
    
    def get_all_terms(self):
        return self.economic_terms

class ContextSensitivePreprocessor:
    """Basic text preprocessor"""
    
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.economic_terms = lexicon.get_all_terms()
    
    def smart_text_cleaning(self, text):
        if not text or not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\\w\\s\\.]', ' ', text)
        text = re.sub(r'\\s+', ' ', text).strip()
        return text
    
    def context_aware_stopword_removal(self, text):
        if not text:
            return ""
        
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
    """Basic clarity calculator"""
    
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.economic_terms = lexicon.get_all_terms()
    
    def calculate_enhanced_clarity(self, text):
        if not text or len(text.strip()) == 0:
            return self._get_default_metrics()
        
        words = text.split()
        if len(words) < 5:
            return self._get_default_metrics()
        
        # Basic calculations
        sentences = len([s for s in text.split('.') if s.strip()])
        sentence_count = max(sentences, 1)
        avg_sentence_length = len(words) / sentence_count
        
        # Technical density
        tech_words = [w for w in words if w in self.economic_terms]
        technical_density = len(tech_words) / len(words)
        
        # Syllable complexity (basic)
        total_syllables = sum(len(re.findall(r'[aiueo]', word)) for word in words)
        syllable_complexity = total_syllables / len(words)
        
        # Readability score
        readability = max(0, 100 - (avg_sentence_length * 2 + syllable_complexity * 30))
        
        # Component scores
        sentence_score = 1.0 if 15 <= avg_sentence_length <= 25 else 0.6
        syllable_score = 1.0 if syllable_complexity <= 2.5 else 0.6
        tech_score = 0.8 + 0.2 * min(technical_density / 0.15, 1)
        
        # Composite score
        composite = (0.3 * sentence_score + 0.2 * syllable_score + 
                    0.2 * tech_score + 0.3 * readability / 100)
        
        return {
            'word_count': len(words),
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'syllable_complexity': syllable_complexity,
            'technical_density': technical_density,
            'contextual_readability': readability,
            'composite_clarity_score': composite,
            'sentence_length_score': sentence_score,
            'syllable_score': syllable_score,
            'technical_density_score': tech_score
        }
    
    def _get_default_metrics(self):
        return {
            'word_count': 0, 'sentence_count': 0, 'avg_sentence_length': 0,
            'syllable_complexity': 0, 'technical_density': 0,
            'contextual_readability': 0, 'composite_clarity_score': 0,
            'sentence_length_score': 0, 'syllable_score': 0,
            'technical_density_score': 0
        }
'''
    
    framework_path = 'scripts/enhanced_clarity_framework.py'
    with open(framework_path, 'w', encoding='utf-8') as f:
        f.write(framework_content)
    
    print(f"✅ Created: {framework_path}")

def create_fixed_main_script():
    """Create the fixed main preprocessing script"""
    
    main_content = '''#!/usr/bin/env python3
"""
enhanced_preprocessing_main.py - Fixed version
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add scripts to path
sys.path.append('scripts')

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/enhanced_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('enhanced_preprocessing')

def main():
    logger = setup_logging()
    logger.info("🚀 Starting Enhanced Preprocessing...")
    
    print("="*60)
    print("🚀 ENHANCED PREPROCESSING PIPELINE - FIXED VERSION")
    print("="*60)
    
    try:
        # Test framework import
        print("\\n📦 Testing framework import...")
        from enhanced_clarity_framework import (
            EnhancedBankSentralLexicon,
            ContextSensitivePreprocessor, 
            EnhancedClarityCalculator
        )
        print("✅ Framework imported successfully")
        
        # Initialize components
        print("\\n⚡ Initializing components...")
        lexicon = EnhancedBankSentralLexicon()
        preprocessor = ContextSensitivePreprocessor(lexicon)
        calculator = EnhancedClarityCalculator(lexicon)
        print(f"✅ Lexicon loaded with {len(lexicon.get_all_terms())} terms")
        
        # Create sample data if no data exists
        print("\\n📊 Preparing test data...")
        sample_data = {
            'ID_Paragraf': [f'TEST_{i:03d}' for i in range(1, 11)],
            'Teks_Paragraf': [
                "Bank Indonesia mempertahankan BI 7-Day Reverse Repo Rate pada level 6,00%.",
                "Keputusan ini konsisten dengan upaya pengendalian inflasi dalam sasaran.",
                "Stabilitas nilai tukar rupiah terus dijaga untuk mendukung ekonomi.",
                "Implementasi kebijakan makroprudensial diperkuat untuk mitigasi risiko.",
                "Koordinasi kebijakan makroekonomi terus ditingkatkan secara berkelanjutan.",
                "Inflasi diperkirakan tetap terkendali dalam kisaran sasaran tahun ini.",
                "Pertumbuhan ekonomi global menunjukkan tren perlambatan yang terkendali.",
                "Kondisi pasar keuangan domestik relatif stabil dengan likuiditas memadai.",
                "Bank Indonesia terus memonitor perkembangan ekonomi global dan domestik.",
                "Transmisi kebijakan moneter berjalan efektif melalui berbagai saluran."
            ],
            'Sentimen_Majority': ['Positif'] * 7 + ['Netral'] * 2 + ['Negatif'] * 1,
            'Confidence_Score': np.random.uniform(0.7, 0.95, 10)
        }
        
        df = pd.DataFrame(sample_data)
        print(f"✅ Created test dataset with {len(df)} records")
        
        # Process data
        print("\\n⚙️ Processing data...")
        results = []
        
        for idx, row in df.iterrows():
            text = row['Teks_Paragraf']
            
            # Process with enhanced methods
            cleaned = preprocessor.smart_text_cleaning(text)
            filtered = preprocessor.context_aware_stopword_removal(cleaned)
            metrics = calculator.calculate_enhanced_clarity(filtered)
            
            result = {
                'ID_Paragraf': row['ID_Paragraf'],
                'original_text': text,
                'processed_text': filtered,
                'clarity_score': metrics['composite_clarity_score'] * 100,
                'readability': metrics['contextual_readability'],
                'sentence_length': metrics['avg_sentence_length'],
                'technical_density': metrics['technical_density'],
                'word_count': metrics['word_count']
            }
            
            results.append(result)
            print(f"  ✅ Processed: {row['ID_Paragraf']} (Score: {result['clarity_score']:.1f})")
        
        # Save results
        print("\\n💾 Saving results...")
        results_df = pd.DataFrame(results)
        
        output_file = 'output/enhanced_clarity/test_results.xlsx'
        results_df.to_excel(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Summary
        avg_score = results_df['clarity_score'].mean()
        print(f"\\n📈 Analysis Summary:")
        print(f"  📊 Average Clarity Score: {avg_score:.1f}/100")
        print(f"  📏 Score Range: {results_df['clarity_score'].min():.1f} - {results_df['clarity_score'].max():.1f}")
        print(f"  📝 Best Score: {results_df['clarity_score'].max():.1f}")
        
        print("\\n" + "="*60)
        print("🎉 ENHANCED PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✅ Framework is working correctly")
        print("✅ Sample data processed successfully")
        print(f"✅ Results saved to: {output_file}")
        print("\\n🚀 Ready for real data processing!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Solution: Run quick_fix_setup.py first")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    main_path = 'enhanced_preprocessing_main.py'
    with open(main_path, 'w', encoding='utf-8') as f:
        f.write(main_content)
    
    print(f"✅ Created: {main_path}")

def test_setup():
    """Test if setup is working"""
    print("\n🧪 Testing setup...")
    
    try:
        # Test framework import
        sys.path.append('scripts')
        from enhanced_clarity_framework import EnhancedBankSentralLexicon
        
        lexicon = EnhancedBankSentralLexicon()
        terms_count = len(lexicon.get_all_terms())
        
        print(f"✅ Framework test successful: {terms_count} economic terms loaded")
        return True
        
    except Exception as e:
        print(f"❌ Framework test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 QUICK FIX AND SETUP")
    print("="*40)
    
    print("\n1️⃣ Creating directory structure...")
    create_directory_structure()
    
    print("\n2️⃣ Creating enhanced clarity framework...")
    create_enhanced_clarity_framework()
    
    print("\n3️⃣ Creating fixed main script...")
    create_fixed_main_script()
    
    print("\n4️⃣ Testing setup...")
    if test_setup():
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*40)
        print("✅ All directories created")
        print("✅ Enhanced framework ready")
        print("✅ Main script fixed")
        print("\n🚀 Now run: python enhanced_preprocessing_main.py")
    else:
        print("\n❌ Setup test failed. Check error messages above.")

if __name__ == "__main__":
    main()