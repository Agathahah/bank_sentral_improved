#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robust_balanced_preprocessing.py - More robust preprocessing for merged annotation data
"""

import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
import re
import argparse
import logging
from tqdm import tqdm
import os
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('robust_preprocessing')

# Economic terminology (same as before)
ECONOMIC_TERMINOLOGY = {
    'monetary_policy': ['inflasi', 'deflasi', 'disinflasi', 'moneter', 'kebijakan', 'transmisi', 
                       'stance', 'akomodatif', 'ketat', 'netral'],
    'banking': ['bank', 'sentral', 'komersial', 'kredit', 'pinjaman', 'simpanan', 'tabungan',
               'deposito', 'likuiditas', 'cadangan', 'intermediasi'],
    'interest_rates': ['suku', 'bunga', 'rate', 'bi7drrr', 'repo', 'reverse', 'fasbi',
                      'lelang', 'tender', 'operasi', 'pasar', 'terbuka'],
    'exchange_rates': ['nilai', 'tukar', 'kurs', 'rupiah', 'dollar', 'valuta', 'asing',
                      'depresiasi', 'apresiasi', 'volatilitas', 'stabilitas'],
    'economic_indicators': ['pertumbuhan', 'pdb', 'gdp', 'produksi', 'konsumsi', 'investasi',
                  'ekspor', 'impor', 'neraca', 'pembayaran', 'perdagangan'],
    'financial_markets': ['pasar', 'modal', 'saham', 'obligasi', 'surat', 'berharga',
                         'yield', 'spread', 'premi', 'risiko', 'volatilitas'],
    'macroprudential': ['makroprudensial', 'sistemik', 'risiko', 'stabilitas', 'finansial',
                       'lwm', 'ctm', 'dsr', 'ltv', 'penyangga', 'konservasi'],
    'temporal': ['kuartal', 'triwulan', 'semester', 'tahunan', 'bulanan', 'harian',
                'periode', 'jangka', 'pendek', 'menengah', 'panjang'],
    'quantitative': ['persen', 'persentase', 'basis', 'poin', 'bps', 'indeks', 'rasio',
                    'tingkat', 'laju', 'volume', 'nominal', 'riil']
}

def validate_and_clean_text_data(df, text_col):
    """Validate and clean text data before preprocessing"""
    logger.info("Validating and cleaning text data...")
    
    # Check if column exists
    if text_col not in df.columns:
        logger.error(f"Text column '{text_col}' not found in dataframe")
        return None
    
    # Create a copy
    df_clean = df.copy()
    
    # Basic statistics before cleaning
    logger.info(f"Before cleaning:")
    logger.info(f"- Total rows: {len(df_clean)}")
    logger.info(f"- Non-null texts: {df_clean[text_col].notna().sum()}")
    logger.info(f"- Null texts: {df_clean[text_col].isna().sum()}")
    
    # Fill NaN with empty string
    df_clean[text_col] = df_clean[text_col].fillna('')
    
    # Convert to string if not already
    df_clean[text_col] = df_clean[text_col].astype(str)
    
    # Remove texts that are too short (less than 10 words)
    word_counts = df_clean[text_col].str.split().str.len()
    df_clean['word_count_original'] = word_counts
    
    # Mark invalid texts
    df_clean['text_valid'] = (word_counts >= 10) & (df_clean[text_col].str.strip() != '')
    
    logger.info(f"After validation:")
    logger.info(f"- Valid texts: {df_clean['text_valid'].sum()}")
    logger.info(f"- Invalid texts: {(~df_clean['text_valid']).sum()}")
    
    # Log sample of invalid texts
    invalid_texts = df_clean[~df_clean['text_valid']]
    if len(invalid_texts) > 0:
        logger.info(f"Sample invalid texts:")
        for idx in invalid_texts.head(3).index:
            text = df_clean.loc[idx, text_col]
            logger.info(f"  Row {idx}: '{text[:50]}...' (words: {df_clean.loc[idx, 'word_count_original']})")
    
    return df_clean

def safe_calculate_clarity_metrics(text):
    """Calculate clarity metrics with robust error handling"""
    metrics = {
        'flesch_reading_ease': np.nan,
        'flesch_kincaid_grade': np.nan,
        'smog_index': np.nan,
        'avg_sentence_length': np.nan,
        'avg_syllables_per_word': np.nan,
        'word_count': 0,
        'sentence_count': 0,
        'complex_word_ratio': 0
    }
    
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return metrics
    
    # Clean text for analysis
    text = text.strip()
    words = text.split()
    word_count = len(words)
    
    if word_count < 10:
        return metrics
    
    # Count sentences (must have at least one)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    
    metrics['word_count'] = word_count
    metrics['sentence_count'] = sentence_count
    metrics['avg_sentence_length'] = word_count / sentence_count
    
    try:
        # Flesch Reading Ease
        # FRE = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        syllable_count = textstat.syllable_count(text)
        avg_syllables = syllable_count / word_count if word_count > 0 else 0
        
        fre = 206.835 - (1.015 * metrics['avg_sentence_length']) - (84.6 * avg_syllables)
        
        # Clamp FRE to reasonable range (0-100)
        fre = max(0, min(100, fre))
        metrics['flesch_reading_ease'] = fre
        
        # Flesch-Kincaid Grade Level
        fkg = 0.39 * metrics['avg_sentence_length'] + 11.8 * avg_syllables - 15.59
        fkg = max(0, min(20, fkg))  # Clamp to reasonable grade levels
        metrics['flesch_kincaid_grade'] = fkg
        
        # SMOG Index
        complex_words = [word for word in words if textstat.syllable_count(word) >= 3]
        complex_word_count = len(complex_words)
        metrics['complex_word_ratio'] = complex_word_count / word_count
        
        if sentence_count >= 3:
            smog = 1.0430 * np.sqrt(complex_word_count * (30 / sentence_count)) + 3.1291
            metrics['smog_index'] = max(0, smog)
        
        metrics['avg_syllables_per_word'] = avg_syllables
        
    except Exception as e:
        logger.debug(f"Error calculating metrics: {e}")
    
    return metrics

def load_comprehensive_economic_terms():
    """Load comprehensive economic terminology"""
    all_terms = set()
    for category, terms in ECONOMIC_TERMINOLOGY.items():
        all_terms.update(terms)
    
    logger.info(f"Loaded {len(all_terms)} economic terms across {len(ECONOMIC_TERMINOLOGY)} categories")
    return all_terms

def advanced_text_cleaning(text):
    """Advanced text cleaning with economic context preservation"""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs but preserve the fact that a URL was there
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Preserve numbers with context
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 persen', text)
    text = re.sub(r'(\d+(?:\.\d+)?)\s*bps', r'\1 basis poin', text)
    
    # Remove excessive punctuation but keep sentence boundaries
    text = re.sub(r'[^\w\s\.]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_smart_stopwords(economic_terms):
    """Create intelligent stopwords list that preserves economic terminology"""
    stopword_factory = StopWordRemoverFactory()
    sastrawi_stopwords = set(stopword_factory.get_stop_words())
    
    try:
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords as nltk_stopwords
        indonesian_stopwords = set(nltk_stopwords.words('indonesian'))
        all_stopwords = sastrawi_stopwords.union(indonesian_stopwords)
    except:
        logger.warning("NLTK stopwords not available, using only Sastrawi")
        all_stopwords = sastrawi_stopwords
    
    # Remove economic terms from stopwords
    smart_stopwords = all_stopwords - economic_terms
    
    # Add common non-informative words
    additional_stopwords = {
        'hal', 'cara', 'saat', 'waktu', 'tempat', 'orang', 'bagian', 'bagaimana',
        'dimana', 'kapan', 'siapa', 'apa', 'mengapa', 'bilamana', 'seberapa'
    }
    
    smart_stopwords.update(additional_stopwords - economic_terms)
    
    logger.info(f"Created smart stopwords: {len(smart_stopwords)} words")
    
    return list(smart_stopwords)

def intelligent_stemming(text, stemmer, economic_terms):
    """Apply intelligent stemming that preserves important economic terms"""
    if not text:
        return ""
    
    words = text.split()
    stemmed_words = []
    
    for word in words:
        if word in economic_terms:
            stemmed_words.append(word)
        else:
            stemmed_word = stemmer.stem(word)
            stemmed_words.append(stemmed_word)
    
    return ' '.join(stemmed_words)

def safe_calculate_preservation_score(original_text, processed_text):
    """Calculate preservation score with error handling"""
    if not original_text or not processed_text:
        return 0.0
    
    # Simple word overlap method as fallback
    original_words = set(original_text.lower().split())
    processed_words = set(processed_text.lower().split())
    
    if not original_words:
        return 0.0
    
    overlap = len(original_words.intersection(processed_words))
    return overlap / len(original_words)

def robust_balanced_preprocessing(df, text_col):
    """Apply balanced preprocessing with robust error handling"""
    logger.info("Starting robust balanced preprocessing...")
    
    # Validate and clean data first
    df = validate_and_clean_text_data(df, text_col)
    if df is None:
        return None, {}
    
    # Load economic terms
    economic_terms = load_comprehensive_economic_terms()
    
    # Initialize tools
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    smart_stopwords = create_smart_stopwords(economic_terms)
    
    # Initialize result dataframe
    result_df = df.copy()
    
    # Create preprocessing variants
    preprocessing_variants = {
        'clean_only': 'Only basic cleaning',
        'clean_no_stopwords': 'Cleaning + stopword removal',
        'clean_stemmed': 'Cleaning + intelligent stemming',
        'clean_stemmed_no_stopwords': 'Full preprocessing',
        'economic_preserved': 'Economic terms preserved + stemming'
    }
    
    # Initialize columns
    for variant in preprocessing_variants:
        result_df[f'processed_{variant}'] = ""
    
    # Process only valid texts
    valid_mask = result_df['text_valid']
    
    # Process each valid text
    for idx in tqdm(result_df[valid_mask].index, desc="Preprocessing texts"):
        original_text = result_df.loc[idx, text_col]
        
        if not original_text or not isinstance(original_text, str):
            continue
        
        # Variant 1: Clean only
        cleaned = advanced_text_cleaning(original_text)
        result_df.at[idx, 'processed_clean_only'] = cleaned
        
        # Variant 2: Clean + remove stopwords
        def remove_smart_stopwords(text):
            words = text.split()
            filtered = [word for word in words if word not in smart_stopwords]
            return ' '.join(filtered)
        
        clean_no_stopwords = remove_smart_stopwords(cleaned)
        result_df.at[idx, 'processed_clean_no_stopwords'] = clean_no_stopwords
        
        # Variant 3: Clean + intelligent stemming
        clean_stemmed = intelligent_stemming(cleaned, stemmer, economic_terms)
        result_df.at[idx, 'processed_clean_stemmed'] = clean_stemmed
        
        # Variant 4: Full preprocessing
        clean_stemmed_no_stopwords = remove_smart_stopwords(clean_stemmed)
        result_df.at[idx, 'processed_clean_stemmed_no_stopwords'] = clean_stemmed_no_stopwords
        
        # Variant 5: Economic preserved
        economic_preserved = intelligent_stemming(clean_no_stopwords, stemmer, economic_terms)
        result_df.at[idx, 'processed_economic_preserved'] = economic_preserved
    
    # Calculate preservation scores for valid texts only
    logger.info("Calculating preservation scores...")
    
    preservation_scores = {}
    for variant in preprocessing_variants:
        scores = []
        for idx in result_df[valid_mask].index:
            original = result_df.loc[idx, text_col]
            processed = result_df.loc[idx, f'processed_{variant}']
            
            score = safe_calculate_preservation_score(original, processed)
            scores.append(score)
        
        if scores:
            preservation_scores[variant] = np.mean(scores)
            result_df.loc[valid_mask, f'preservation_score_{variant}'] = scores
        else:
            preservation_scores[variant] = 0.0
    
    # Log preservation scores
    logger.info("Preservation scores by variant:")
    for variant, score in preservation_scores.items():
        logger.info(f"  {variant}: {score:.3f}")
    
    return result_df, preservation_scores

def calculate_clarity_metrics_batch_robust(df, text_columns):
    """Calculate clarity metrics with robust error handling"""
    logger.info("Calculating clarity metrics for all variants...")
    
    clarity_results = []
    
    # Only process valid texts
    valid_mask = df['text_valid'] if 'text_valid' in df.columns else pd.Series([True] * len(df))
    
    for col in text_columns:
        if col not in df.columns:
            continue
            
        logger.info(f"Processing clarity metrics for: {col}")
        
        for idx in tqdm(df[valid_mask].index, desc=f"Clarity: {col}"):
            text = df.loc[idx, col] if pd.notna(df.loc[idx, col]) else ""
            metrics = safe_calculate_clarity_metrics(text)
            
            result_row = {'row_id': idx, 'text_variant': col}
            result_row.update(metrics)
            clarity_results.append(result_row)
    
    return pd.DataFrame(clarity_results)

def create_robust_preprocessing_analysis(df, preservation_scores, clarity_df, output_dir):
    """Create preprocessing analysis with robust calculations"""
    logger.info("Creating preprocessing analysis...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter clarity_df for valid metrics only
    clarity_df_valid = clarity_df.dropna(subset=['flesch_reading_ease'])
    
    plt.figure(figsize=(20, 15))
    
    # 1. Preservation scores comparison
    plt.subplot(3, 3, 1)
    if preservation_scores:
        variants = list(preservation_scores.keys())
        scores = list(preservation_scores.values())
        plt.bar(variants, scores)
        plt.title('Information Preservation by Preprocessing Variant')
        plt.ylabel('Preservation Score')
        plt.xticks(rotation=45, ha='right')
    
    # 2. Clarity comparison (with valid data only)
    plt.subplot(3, 3, 2)
    if not clarity_df_valid.empty:
        clarity_means = clarity_df_valid.groupby('text_variant')['flesch_reading_ease'].mean()
        if not clarity_means.empty:
            plt.bar(clarity_means.index, clarity_means.values)
            plt.title('Average Flesch Reading Ease by Variant')
            plt.ylabel('Reading Ease Score')
            plt.xticks(rotation=45, ha='right')
            # Add reference lines
            plt.axhline(y=60, color='g', linestyle='--', alpha=0.5, label='Easy')
            plt.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Difficult')
            plt.legend()
    
    # 3. Word count reduction
    plt.subplot(3, 3, 3)
    if not clarity_df_valid.empty:
        word_counts = clarity_df_valid.groupby('text_variant')['word_count'].mean()
        if not word_counts.empty:
            plt.bar(word_counts.index, word_counts.values)
            plt.title('Average Word Count by Variant')
            plt.ylabel('Word Count')
            plt.xticks(rotation=45, ha='right')
    
    # Continue with other plots...
    # (Rest of the visualization code remains similar but with validity checks)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'preprocessing_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate comprehensive scores with valid data
    comprehensive_scores = {}
    
    if clarity_df_valid.empty:
        # Use simple scoring if clarity metrics failed
        for variant in preservation_scores.keys():
            comprehensive_scores[variant] = preservation_scores.get(variant, 0.0)
    else:
        clarity_means = clarity_df_valid.groupby('text_variant')['flesch_reading_ease'].mean()
        
        for variant in preservation_scores.keys():
            preservation = preservation_scores.get(variant, 0.0)
            
            # Normalize clarity score (0-100 to 0-1)
            clarity_raw = clarity_means.get(f'processed_{variant}', 50)
            clarity_norm = max(0, min(1, clarity_raw / 100))
            
            # Simple weighted average
            score = (0.5 * preservation + 0.5 * clarity_norm)
            comprehensive_scores[variant] = score
    
    # Find best variant
    if comprehensive_scores:
        best_variant = max(comprehensive_scores.items(), key=lambda x: x[1])[0]
    else:
        best_variant = 'processed_clean_only'  # Default fallback
    
    # Save detailed statistics
    stats_summary = pd.DataFrame({
        'Variant': list(preservation_scores.keys()),
        'Preservation_Score': list(preservation_scores.values()),
        'Avg_Reading_Ease': [clarity_means.get(f'processed_{v}', np.nan) if 'clarity_means' in locals() else np.nan 
                            for v in preservation_scores.keys()],
        'Comprehensive_Score': [comprehensive_scores.get(v, np.nan) for v in preservation_scores.keys()]
    })
    
    stats_summary.to_excel(os.path.join(output_dir, 'preprocessing_comparison.xlsx'), index=False)
    
    # Create recommendation
    valid_text_count = df['text_valid'].sum() if 'text_valid' in df.columns else len(df)
    total_text_count = len(df)
    
    recommendation = f"""
REKOMENDASI PREPROCESSING
========================

Berdasarkan analisis komprehensif terhadap {len(preservation_scores)} varian preprocessing:

DATA QUALITY:
- Total paragraf: {total_text_count}
- Valid texts (>10 words): {valid_text_count}
- Invalid/short texts: {total_text_count - valid_text_count}

VARIANT TERBAIK: {best_variant}
- Preservation Score: {preservation_scores.get(best_variant, 'N/A')}
- Comprehensive Score: {comprehensive_scores.get(best_variant, 'N/A')}

RANKING SEMUA VARIANT:
"""
    
    for i, (variant, score) in enumerate(sorted(comprehensive_scores.items(), key=lambda x: x[1], reverse=True), 1):
        recommendation += f"{i}. {variant}: {score:.3f}\n"
    
    recommendation += """
CATATAN:
- Preservation Score: Seberapa banyak informasi asli yang dipertahankan
- Reading Ease: Kemudahan membaca (semakin tinggi semakin mudah)
- Comprehensive Score: Kombinasi preservation dan clarity

REKOMENDASI PENGGUNAAN:
- Untuk sentiment analysis: Gunakan variant dengan preservation score tinggi
- Untuk readability analysis: Gunakan variant asli atau clean_only
- Untuk topic modeling: Gunakan variant yang mempertahankan economic terms
- Untuk machine learning: Gunakan variant terbaik berdasarkan comprehensive score

PERHATIAN:
- Reading ease scores yang sangat rendah/negatif menunjukkan teks yang sangat kompleks
- Pertimbangkan untuk filter teks yang terlalu pendek (<10 kata) sebelum analisis
"""
    
    with open(os.path.join(output_dir, 'preprocessing_recommendation.txt'), 'w', encoding='utf-8') as f:
        f.write(recommendation)
    
    logger.info(f"Preprocessing analysis completed. Best variant: {best_variant}")
    return best_variant, comprehensive_scores

def main():
    parser = argparse.ArgumentParser(description='Robust balanced preprocessing for merged annotation data')
    parser.add_argument('--input', required=True, help='Input Excel file')
    parser.add_argument('--output', required=True, help='Output Excel file')
    parser.add_argument('--text-col', required=True, help='Text column name')
    parser.add_argument('--output-dir', default='output/preprocessing_analysis', 
                       help='Output directory for analysis')
    parser.add_argument('--sheet-name', default=0, help='Sheet name or index')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_excel(args.input, sheet_name=args.sheet_name)
    logger.info(f"Loaded {len(df)} rows")
    
    if args.text_col not in df.columns:
        logger.error(f"Text column '{args.text_col}' not found")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Apply robust balanced preprocessing
    result_df, preservation_scores = robust_balanced_preprocessing(df, args.text_col)
    
    if result_df is None:
        logger.error("Preprocessing failed")
        return
    
    # Calculate clarity metrics
    processed_cols = [col for col in result_df.columns if col.startswith('processed_')]
    clarity_df = calculate_clarity_metrics_batch_robust(result_df, processed_cols + [args.text_col])
    
    # Create analysis
    best_variant, comprehensive_scores = create_robust_preprocessing_analysis(
        result_df, preservation_scores, clarity_df, args.output_dir
    )
    
    # Save processed data
    result_df.to_excel(args.output, index=False)
    clarity_df.to_excel(os.path.join(args.output_dir, 'clarity_metrics_detailed.xlsx'), index=False)
    
    logger.info(f"Robust preprocessing completed. Results saved to {args.output}")
    logger.info(f"Analysis saved to {args.output_dir}")
    logger.info(f"Recommended variant: {best_variant}")

if __name__ == "__main__":
    main()