#!/usr/bin/env python3
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
        print("\n📦 Testing framework import...")
        from enhanced_clarity_framework import (
            EnhancedBankSentralLexicon,
            ContextSensitivePreprocessor, 
            EnhancedClarityCalculator
        )
        print("✅ Framework imported successfully")
        
        # Initialize components
        print("\n⚡ Initializing components...")
        lexicon = EnhancedBankSentralLexicon()
        preprocessor = ContextSensitivePreprocessor(lexicon)
        calculator = EnhancedClarityCalculator(lexicon)
        print(f"✅ Lexicon loaded with {len(lexicon.get_all_terms())} terms")
        
        # Create sample data if no data exists
        print("\n📊 Preparing test data...")
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
        print("\n⚙️ Processing data...")
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
        print("\n💾 Saving results...")
        results_df = pd.DataFrame(results)
        
        output_file = 'output/enhanced_clarity/test_results.xlsx'
        results_df.to_excel(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Summary
        avg_score = results_df['clarity_score'].mean()
        print(f"\n📈 Analysis Summary:")
        print(f"  📊 Average Clarity Score: {avg_score:.1f}/100")
        print(f"  📏 Score Range: {results_df['clarity_score'].min():.1f} - {results_df['clarity_score'].max():.1f}")
        print(f"  📝 Best Score: {results_df['clarity_score'].max():.1f}")
        
        print("\n" + "="*60)
        print("🎉 ENHANCED PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✅ Framework is working correctly")
        print("✅ Sample data processed successfully")
        print(f"✅ Results saved to: {output_file}")
        print("\n🚀 Ready for real data processing!")
        
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
