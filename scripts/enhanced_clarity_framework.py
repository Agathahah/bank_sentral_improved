#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_clarity_framework.py - Complete Enhanced Framework
"""

import pandas as pd
import numpy as np
import re
import json
import os
from collections import defaultdict, Counter
from datetime import datetime
import logging

class EnhancedBankSentralLexicon:
    """Enhanced lexicon for Bank Sentral terminology"""
    
    def __init__(self):
        self.logger = logging.getLogger('EnhancedLexicon')
        
        # Comprehensive economic terms by category
        self.economic_terms = {
            'monetary_policy': {
                'bank', 'sentral', 'indonesia', 'bi', 'moneter', 'kebijakan', 
                'repo', 'rate', 'reverse', 'suku', 'bunga', 'acuan', 'fasbi',
                'akomodatif', 'ketat', 'netral', 'ekspansif', 'kontraktif'
            },
            'inflation': {
                'inflasi', 'deflasi', 'disinflasi', 'harga', 'konsumen',
                'administered', 'volatile', 'food', 'core', 'inti', 'sasaran',
                'target', 'terkendali', 'meningkat', 'menurun', 'stabil'
            },
            'exchange_rate': {
                'nilai', 'tukar', 'kurs', 'rupiah', 'dolar', 'usd', 'apresiasi',
                'depresiasi', 'volatilitas', 'stabilitas', 'intervensi',
                'cadangan', 'devisa', 'valas', 'forex'
            },
            'financial_markets': {
                'pasar', 'modal', 'uang', 'keuangan', 'obligasi', 'saham',
                'sbn', 'sukuk', 'likuiditas', 'investor', 'asing', 'domestik',
                'yield', 'spread', 'risk', 'premium'
            },
            'banking': {
                'perbankan', 'kredit', 'dana', 'pihak', 'ketiga', 'dpk',
                'loan', 'deposit', 'intermediasi', 'penyaluran', 'pembiayaan',
                'npf', 'npl', 'car', 'ldr', 'fdr'
            },
            'macroprudential': {
                'makroprudensial', 'sistemik', 'risiko', 'ltv', 'dsr', 'ccyb',
                'lcr', 'nsfr', 'mitigasi', 'stabilitas', 'sistem', 'buffer'
            },
            'economic_indicators': {
                'ekonomi', 'pertumbuhan', 'gdp', 'pdb', 'konsumsi', 'investasi',
                'ekspor', 'impor', 'neraca', 'pembayaran', 'transaksi', 'berjalan'
            },
            'communication': {
                'komunikasi', 'transparan', 'forward', 'guidance', 'ekspektasi',
                'proyeksi', 'outlook', 'assessment', 'evaluasi', 'monitoring'
            }
        }
        
        # Flatten all terms
        self.all_economic_terms = set()
        for category_terms in self.economic_terms.values():
            self.all_economic_terms.update(category_terms)
        
        # Context-sensitive stopwords
        self.context_sensitive_stopwords = {
            'preserve': {
                'dapat', 'akan', 'perlu', 'harus', 'mampu', 'terus', 'tetap',
                'lebih', 'kurang', 'sangat', 'cukup', 'masih', 'sudah',
                'belum', 'tidak', 'bukan', 'juga', 'serta', 'maupun'
            },
            'safe_remove': {
                'itu', 'ini', 'tersebut', 'adalah', 'merupakan', 'berupa',
                'yaitu', 'yakni', 'antara', 'lain', 'dimana', 'dimana'
            }
        }
        
        # Acronym expansion
        self.acronym_expansions = {
            'bi': 'bank indonesia',
            'bis': 'bank for international settlements',
            'imf': 'international monetary fund',
            'fed': 'federal reserve',
            'ecb': 'european central bank',
            'boj': 'bank of japan',
            'pbc': 'peoples bank of china',
            'rbi': 'reserve bank of india'
        }
        
        # Clarity optimization weights
        self.clarity_weights = {
            'sentence_length': {
                'ideal_range': (15, 25),
                'penalty_factor': 0.1
            },
            'technical_density': {
                'optimal_ratio': 0.15,
                'max_ratio': 0.30
            },
            'syllable_complexity': {
                'target_avg': 2.5,
                'max_avg': 4.0
            }
        }
        
        self.logger.info(f"Lexicon initialized with {len(self.all_economic_terms)} terms")
    
    def get_all_terms(self):
        return self.all_economic_terms
    
    def get_terms_by_category(self, category):
        return self.economic_terms.get(category, set())
    
    def expand_acronyms(self, text):
        """Expand common acronyms"""
        words = text.lower().split()
        expanded = []
        
        for word in words:
            if word in self.acronym_expansions:
                expanded.append(self.acronym_expansions[word])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)

class ContextSensitivePreprocessor:
    """Enhanced preprocessor with context awareness"""
    
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.economic_terms = lexicon.get_all_terms()
        self.logger = logging.getLogger('ContextPreprocessor')
    
    def smart_text_cleaning(self, text):
        """Intelligent text cleaning preserving important terms"""
        if not text or not isinstance(text, str):
            return ""
        
        # Expand acronyms first
        text = self.lexicon.expand_acronyms(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but preserve periods for sentence detection
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def context_aware_stopword_removal(self, text):
        """Remove stopwords while preserving important economic context"""
        if not text:
            return ""
        
        words = text.split()
        preserve = self.lexicon.context_sensitive_stopwords['preserve']
        safe_remove = self.lexicon.context_sensitive_stopwords['safe_remove']
        
        filtered_words = []
        
        for word in words:
            # Always preserve economic terms
            if word in self.economic_terms:
                filtered_words.append(word)
            # Preserve context-important words
            elif word in preserve:
                filtered_words.append(word)
            # Remove safe-to-remove words
            elif word in safe_remove:
                continue
            # Keep everything else
            else:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def extract_sentences(self, text):
        """Extract sentences for analysis"""
        if not text:
            return []
        
        # Simple sentence splitting
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences

class EnhancedClarityCalculator:
    """Enhanced clarity calculator with comprehensive metrics"""
    
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.economic_terms = lexicon.get_all_terms()
        self.logger = logging.getLogger('ClarityCalculator')
    
    def calculate_enhanced_clarity(self, text):
        """Calculate comprehensive clarity metrics"""
        if not text or len(text.strip()) == 0:
            return self._get_default_metrics()
        
        # Basic text statistics
        words = text.split()
        if len(words) < 3:
            return self._get_default_metrics()
        
        sentences = self._extract_sentences(text)
        sentence_count = max(len(sentences), 1)
        
        # Core metrics
        metrics = {
            'word_count': len(words),
            'sentence_count': sentence_count,
            'avg_sentence_length': len(words) / sentence_count,
            'syllable_complexity': self._calculate_syllable_complexity(words),
            'technical_density': self._calculate_technical_density(words),
            'contextual_readability': self._calculate_contextual_readability(text, words, sentences)
        }
        
        # Component scores
        metrics.update({
            'sentence_length_score': self._score_sentence_length(metrics['avg_sentence_length']),
            'syllable_score': self._score_syllable_complexity(metrics['syllable_complexity']),
            'technical_density_score': self._score_technical_density(metrics['technical_density'])
        })
        
        # Composite clarity score
        metrics['composite_clarity_score'] = self._calculate_composite_score(metrics)
        
        return metrics
    
    def _extract_sentences(self, text):
        """Extract sentences from text"""
        return [s.strip() for s in text.split('.') if s.strip()]
    
    def _calculate_syllable_complexity(self, words):
        """Calculate average syllables per word"""
        total_syllables = 0
        
        for word in words:
            # Simple syllable counting using vowel patterns
            syllables = len(re.findall(r'[aiueo]', word.lower()))
            total_syllables += max(syllables, 1)  # Minimum 1 syllable per word
        
        return total_syllables / len(words)
    
    def _calculate_technical_density(self, words):
        """Calculate ratio of technical/economic terms"""
        technical_words = [w for w in words if w in self.economic_terms]
        return len(technical_words) / len(words)
    
    def _calculate_contextual_readability(self, text, words, sentences):
        """Calculate context-aware readability score"""
        # Flesch-Kincaid inspired but adjusted for economic content
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_syllables = self._calculate_syllable_complexity(words)
        
        # Base readability calculation
        base_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        # Adjust for economic context
        technical_density = self._calculate_technical_density(words)
        
        # Penalty for too many technical terms
        if technical_density > 0.25:
            base_score -= (technical_density - 0.25) * 100
        
        # Bonus for appropriate technical density
        elif 0.10 <= technical_density <= 0.20:
            base_score += 10
        
        return max(0, min(100, base_score))
    
    def _score_sentence_length(self, avg_length):
        """Score sentence length (0-1 scale)"""
        ideal_min, ideal_max = self.lexicon.clarity_weights['sentence_length']['ideal_range']
        
        if ideal_min <= avg_length <= ideal_max:
            return 1.0
        elif avg_length < ideal_min:
            # Penalty for too short
            return 0.7 + 0.3 * (avg_length / ideal_min)
        else:
            # Penalty for too long
            penalty = (avg_length - ideal_max) * 0.05
            return max(0.3, 1.0 - penalty)
    
    def _score_syllable_complexity(self, avg_syllables):
        """Score syllable complexity (0-1 scale)"""
        target = self.lexicon.clarity_weights['syllable_complexity']['target_avg']
        max_avg = self.lexicon.clarity_weights['syllable_complexity']['max_avg']
        
        if avg_syllables <= target:
            return 1.0
        elif avg_syllables <= max_avg:
            return 1.0 - 0.5 * ((avg_syllables - target) / (max_avg - target))
        else:
            return 0.3
    
    def _score_technical_density(self, density):
        """Score technical density (0-1 scale)"""
        optimal = self.lexicon.clarity_weights['technical_density']['optimal_ratio']
        max_ratio = self.lexicon.clarity_weights['technical_density']['max_ratio']
        
        if density <= optimal:
            return 0.8 + 0.2 * (density / optimal)
        elif density <= max_ratio:
            return 1.0 - 0.3 * ((density - optimal) / (max_ratio - optimal))
        else:
            return 0.4
    
    def _calculate_composite_score(self, metrics):
        """Calculate weighted composite clarity score"""
        # Default weights
        weights = {
            'sentence_length': 0.25,
            'syllable': 0.20,
            'technical_density': 0.15,
            'readability': 0.40
        }
        
        composite = (
            weights['sentence_length'] * metrics['sentence_length_score'] +
            weights['syllable'] * metrics['syllable_score'] +
            weights['technical_density'] * metrics['technical_density_score'] +
            weights['readability'] * (metrics['contextual_readability'] / 100)
        )
        
        return composite
    
    def _get_default_metrics(self):
        """Return default metrics for empty/invalid text"""
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0,
            'syllable_complexity': 0,
            'technical_density': 0,
            'contextual_readability': 0,
            'composite_clarity_score': 0,
            'sentence_length_score': 0,
            'syllable_score': 0,
            'technical_density_score': 0
        }

# Utility functions
def initialize_enhanced_framework():
    """Initialize the complete enhanced framework"""
    lexicon = EnhancedBankSentralLexicon()
    preprocessor = ContextSensitivePreprocessor(lexicon)
    calculator = EnhancedClarityCalculator(lexicon)
    
    return lexicon, preprocessor, calculator

def process_text_batch(texts, lexicon=None, preprocessor=None, calculator=None):
    """Process multiple texts efficiently"""
    if not lexicon:
        lexicon, preprocessor, calculator = initialize_enhanced_framework()
    
    results = []
    
    for i, text in enumerate(texts):
        try:
            # Preprocess
            cleaned = preprocessor.smart_text_cleaning(str(text))
            filtered = preprocessor.context_aware_stopword_removal(cleaned)
            
            # Calculate clarity
            metrics = calculator.calculate_enhanced_clarity(filtered)
            
            # Add metadata
            result = {
                'text_id': i,
                'original_length': len(str(text).split()),
                'processed_text': filtered,
                **metrics
            }
            
            results.append(result)
            
        except Exception as e:
            logging.error(f"Error processing text {i}: {e}")
            continue
    
    return results

# Main execution for testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the framework
    print("Testing Enhanced Clarity Framework...")
    
    test_texts = [
        "Bank Indonesia mempertahankan BI 7-Day Reverse Repo Rate pada level 6,00%.",
        "Implementasi kebijakan makroprudensial melalui penerapan rasio loan-to-value dan debt-service-ratio dalam konteks mitigasi risiko sistemik perbankan dengan mempertimbangkan aspek intermediasi yang berkelanjutan dan optimal.",
        "Inflasi terkendali sesuai target yang ditetapkan."
    ]
    
    # Initialize framework
    lexicon, preprocessor, calculator = initialize_enhanced_framework()
    
    print(f"Framework initialized with {len(lexicon.get_all_terms())} economic terms")
    
    # Process test texts
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Original: {text}")
        
        cleaned = preprocessor.smart_text_cleaning(text)
        filtered = preprocessor.context_aware_stopword_removal(cleaned)
        metrics = calculator.calculate_enhanced_clarity(filtered)
        
        print(f"Processed: {filtered}")
        print(f"Clarity Score: {metrics['composite_clarity_score']*100:.1f}/100")
        print(f"Readability: {metrics['contextual_readability']:.1f}")
        print(f"Technical Density: {metrics['technical_density']:.1%}")
    
    print("\n✅ Framework test completed successfully!")