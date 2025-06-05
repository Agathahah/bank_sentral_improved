#!/usr/bin/env python3
"""
enhanced_lexicon_analysis.py - Build comprehensive lexicon dictionary for central bank communication
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import json

class CentralBankLexiconBuilder:
    """Build domain-specific lexicon for central bank communication sentiment analysis"""
    
    def __init__(self):
        self.lexicon_dict = {}
        self.contextual_rules = {}
        
    def build_central_bank_lexicon(self):
        """Build comprehensive sentiment lexicon for central bank communication"""
        
        # POSITIVE SENTIMENT WORDS (Bank Sentral Context)
        positive_words = {
            # Economic Growth & Stability
            'tumbuh': 0.7, 'meningkat': 0.6, 'stabil': 0.8, 'kokoh': 0.8,
            'menguat': 0.7, 'membaik': 0.8, 'optimis': 0.9, 'positif': 0.8,
            'kondusif': 0.8, 'sehat': 0.8, 'solid': 0.8, 'robust': 0.9,
            
            # Policy Effectiveness
            'efektif': 0.8, 'berhasil': 0.9, 'tepat': 0.7, 'optimal': 0.9,
            'terkendali': 0.8, 'sesuai': 0.7, 'mendukung': 0.7, 'sejalan': 0.6,
            
            # Market Confidence
            'kepercayaan': 0.8, 'kredibel': 0.9, 'terpercaya': 0.8, 'yakin': 0.7,
            'konsisten': 0.8, 'transparan': 0.7, 'akuntabel': 0.8,
            
            # Economic Performance
            'ekspansi': 0.6, 'recovery': 0.8, 'pemulihan': 0.8, 'progres': 0.7,
            'kemajuan': 0.7, 'pencapaian': 0.8, 'prestasi': 0.8,
            
            # Monetary Policy
            'akomodatif': 0.6, 'longgar': 0.5, 'stimulus': 0.6, 'dukungan': 0.7,
            'fasilitasi': 0.6, 'akselerasi': 0.7
        }
        
        # NEGATIVE SENTIMENT WORDS (Bank Sentral Context)
        negative_words = {
            # Economic Decline
            'menurun': -0.6, 'melemah': -0.7, 'turun': -0.6, 'anjlok': -0.9,
            'kontraksi': -0.8, 'resesi': -0.9, 'krisis': -0.9, 'kolaps': -1.0,
            
            # Risk & Uncertainty
            'risiko': -0.7, 'bahaya': -0.8, 'ancaman': -0.8, 'ketidakpastian': -0.7,
            'volatilitas': -0.6, 'fluktuasi': -0.5, 'instabilitas': -0.8,
            
            # Policy Challenges
            'gagal': -0.9, 'tidak efektif': -0.8, 'bermasalah': -0.7, 'kendala': -0.6,
            'hambatan': -0.6, 'tantangan': -0.5, 'kesulitan': -0.6,
            
            # Market Stress
            'tekanan': -0.7, 'stress': -0.8, 'ketegangan': -0.7, 'kekhawatiran': -0.7,
            'pesimis': -0.8, 'negatif': -0.7, 'buruk': -0.8,
            
            # Monetary Tightening (context-dependent)
            'ketat': -0.4, 'restriktif': -0.5, 'hawkish': -0.4, 'agresif': -0.6
        }
        
        # NEUTRAL/CONTEXT-DEPENDENT WORDS
        neutral_words = {
            'kebijakan': 0.0, 'strategi': 0.0, 'langkah': 0.0, 'tindakan': 0.0,
            'koordinasi': 0.0, 'sinkronisasi': 0.0, 'evaluasi': 0.0, 'monitoring': 0.0,
            'observasi': 0.0, 'analisis': 0.0, 'proyeksi': 0.0, 'prediksi': 0.0,
            'normalisasi': 0.0, 'penyesuaian': 0.0, 'kalibrasi': 0.0
        }
        
        # Combine all lexicons
        self.lexicon_dict.update(positive_words)
        self.lexicon_dict.update(negative_words)
        self.lexicon_dict.update(neutral_words)
        
        return self.lexicon_dict
    
    def build_contextual_rules(self):
        """Build contextual sentiment rules"""
        
        self.contextual_rules = {
            # Negation handling
            'negation_words': ['tidak', 'bukan', 'belum', 'tanpa', 'kurang'],
            'negation_window': 3,  # words before sentiment word
            
            # Intensifiers
            'intensifiers': {
                'sangat': 1.5, 'amat': 1.4, 'begitu': 1.3, 'sekali': 1.2,
                'cukup': 0.8, 'agak': 0.7, 'sedikit': 0.6, 'hampir': 0.9
            },
            
            # Economic context modifiers
            'economic_context': {
                'proyeksi': 0.8,  # reduce certainty for projections
                'prakiraan': 0.8,
                'perkiraan': 0.8,
                'target': 0.9,    # increase weight for targets
                'sasaran': 0.9,
                'realisasi': 1.1,  # increase weight for actual results
                'aktual': 1.1
            },
            
            # Temporal modifiers
            'temporal_context': {
                'jangka_pendek': 0.9,
                'jangka_menengah': 1.0,
                'jangka_panjang': 1.1,  # long-term more important
                'sementara': 0.7,       # temporary conditions less weight
                'berkelanjutan': 1.2    # sustained conditions more weight
            }
        }
        
        return self.contextual_rules
    
    def calculate_lexicon_sentiment(self, text, apply_context=True):
        """Calculate sentiment score using lexicon with contextual rules"""
        
        if not text or not isinstance(text, str):
            return 0.0, 0.0  # sentiment_score, confidence
        
        words = text.lower().split()
        sentiment_scores = []
        word_sentiments = []
        
        for i, word in enumerate(words):
            if word in self.lexicon_dict:
                base_score = self.lexicon_dict[word]
                confidence = 1.0
                
                if apply_context and self.contextual_rules:
                    # Apply contextual modifications
                    base_score, confidence = self._apply_contextual_rules(
                        word, base_score, words, i
                    )
                
                sentiment_scores.append(base_score)
                word_sentiments.append({
                    'word': word,
                    'base_sentiment': self.lexicon_dict[word],
                    'contextual_sentiment': base_score,
                    'confidence': confidence
                })
        
        if not sentiment_scores:
            return 0.0, 0.0
        
        # Calculate overall sentiment
        avg_sentiment = np.mean(sentiment_scores)
        avg_confidence = np.mean([ws['confidence'] for ws in word_sentiments])
        
        # Normalize to [-1, 1] range
        normalized_sentiment = np.tanh(avg_sentiment)
        
        return normalized_sentiment, avg_confidence
    
    def _apply_contextual_rules(self, word, base_score, words, word_index):
        """Apply contextual rules to sentiment score"""
        
        modified_score = base_score
        confidence = 1.0
        
        # Check for negation
        negation_start = max(0, word_index - self.contextual_rules['negation_window'])
        preceding_words = words[negation_start:word_index]
        
        for neg_word in self.contextual_rules['negation_words']:
            if neg_word in preceding_words:
                modified_score *= -0.8  # flip and reduce intensity
                confidence *= 0.9
                break
        
        # Check for intensifiers
        if word_index > 0:
            prev_word = words[word_index - 1]
            if prev_word in self.contextual_rules['intensifiers']:
                multiplier = self.contextual_rules['intensifiers'][prev_word]
                modified_score *= multiplier
                confidence *= 1.1 if multiplier > 1 else 0.9
        
        # Apply economic context
        for context_word, modifier in self.contextual_rules['economic_context'].items():
            if context_word in words:
                modified_score *= modifier
                confidence *= modifier
        
        return modified_score, min(confidence, 1.0)
    
    def save_lexicon(self, filepath):
        """Save lexicon dictionary to file"""
        
        lexicon_data = {
            'lexicon_dict': self.lexicon_dict,
            'contextual_rules': self.contextual_rules,
            'metadata': {
                'total_words': len(self.lexicon_dict),
                'positive_words': len([w for w, s in self.lexicon_dict.items() if s > 0]),
                'negative_words': len([w for w, s in self.lexicon_dict.items() if s < 0]),
                'neutral_words': len([w for w, s in self.lexicon_dict.items() if s == 0])
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(lexicon_data, f, indent=2, ensure_ascii=False)
        
        print(f"Lexicon saved to: {filepath}")
        return lexicon_data

# Usage example
if __name__ == "__main__":
    # Build lexicon
    lexicon_builder = CentralBankLexiconBuilder()
    lexicon_dict = lexicon_builder.build_central_bank_lexicon()
    contextual_rules = lexicon_builder.build_contextual_rules()
    
    # Test sentiment calculation
    test_text = "Pertumbuhan ekonomi menunjukkan tren yang positif dan sangat menggembirakan"
    sentiment, confidence = lexicon_builder.calculate_lexicon_sentiment(test_text)
    
    print(f"Text: {test_text}")
    print(f"Sentiment Score: {sentiment:.3f}")
    print(f"Confidence: {confidence:.3f}")
    
    # Save lexicon
    lexicon_builder.save_lexicon('central_bank_lexicon.json')