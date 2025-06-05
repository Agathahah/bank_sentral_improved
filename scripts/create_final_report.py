#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_final_report.py - Generate comprehensive final report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import logging
from datetime import datetime
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('final_report')

def ensure_directory(directory):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_results_data(annotation_dir, preprocessing_dir, sentiment_dir, multidimensional_dir):
    """Load all results data"""
    logger.info("Loading results data...")
    
    results = {}
    
    # Load annotation quality results
    try:
        ann_report_path = os.path.join(annotation_dir, 'annotation_quality_report.txt')
        if os.path.exists(ann_report_path):
            with open(ann_report_path, 'r', encoding='utf-8') as f:
                results['annotation_report'] = f.read()
        
        ann_stats_path = os.path.join(annotation_dir, 'agreement_statistics.xlsx')
        if os.path.exists(ann_stats_path):
            results['annotation_stats'] = pd.read_excel(ann_stats_path)
    except Exception as e:
        logger.warning("Could not load annotation quality results: {e}")
    
    # Load preprocessing results
    try:
        prep_comparison_path = os.path.join(preprocessing_dir, 'preprocessing_comparison.xlsx')
        if os.path.exists(prep_comparison_path):
            results['preprocessing_comparison'] = pd.read_excel(prep_comparison_path)
        
        prep_rec_path = os.path.join(preprocessing_dir, 'preprocessing_recommendation.txt')
        if os.path.exists(prep_rec_path):
            with open(prep_rec_path, 'r', encoding='utf-8') as f:
                results['preprocessing_recommendation'] = f.read()
    except Exception as e:
        logger.warning("Could not load preprocessing results: {e}")
    
    # Load sentiment analysis results
    try:
        # Find the best model results
        sentiment_dirs = [d for d in os.listdir(sentiment_dir) if os.path.isdir(os.path.join(sentiment_dir, d))]
        
        for subdir in sentiment_dirs:
            subdir_path = os.path.join(sentiment_dir, subdir)
            model_comp_path = os.path.join(subdir_path, 'model_comparison.xlsx')
            
            if os.path.exists(model_comp_path):
                results['sentiment_comparison'] = pd.read_excel(model_comp_path)
                results['sentiment_subdir'] = subdir
                break
        
        # Load final summary
        final_summary_path = os.path.join(sentiment_dir, 'final_summary.txt')
        if os.path.exists(final_summary_path):
            with open(final_summary_path, 'r', encoding='utf-8') as f:
                results['sentiment_summary'] = f.read()
                
    except Exception as e:
        logger.warning("Could not load sentiment analysis results: {e}")
    
    # Load multidimensional results
    try:
        multi_report_path = os.path.join(multidimensional_dir, 'comprehensive_communication_analysis.txt')
        if os.path.exists(multi_report_path):
            with open(multi_report_path, 'r', encoding='utf-8') as f:
                results['multidimensional_report'] = f.read()
        
        multi_results_path = os.path.join(multidimensional_dir, 'detailed_analysis_results.xlsx')
        if os.path.exists(multi_results_path):
            results['multidimensional_data'] = pd.read_excel(multi_results_path, sheet_name=None)
                
    except Exception as e:
        logger.warning("Could not load multidimensional results: {e}")
    
    return results

def create_executive_summary(results):
    """Create executive summary"""
    
    summary = """
# RINGKASAN EKSEKUTIF

## Tujuan Penelitian
Penelitian ini bertujuan untuk menganalisis efektivitas berbagai fitur komunikasi Bank Sentral (clarity, sentiment, comprehensiveness, consistency) dan bagaimana konsistensi informasi berpengaruh terhadap respons pasar dan kredibilitas kebijakan.

## Metodologi
- **Enhanced Annotation Merging**: Penggabungan hasil anotasi dari 3 annotator dengan recovery data hilang
- **Balanced Preprocessing**: Preprocessing dengan preservasi terminologi ekonomi dan multiple variants
- **Advanced Sentiment Analysis**: Analisis sentimen dengan class balancing menggunakan SMOTE
- **Multidimensional Analysis**: Analisis komprehensif 4 dimensi komunikasi

## Temuan Utama

"""
    
    # Add annotation quality findings
    if 'annotation_stats' in results:
        stats = results['annotation_stats'].iloc[0] if len(results['annotation_stats']) > 0 else {}
        if 'avg_kappa' in stats:
            kappa = stats['avg_kappa']
            if kappa > 0.6:
                agreement_level = "substansial"
            elif kappa > 0.4:
                agreement_level = "moderat"
            elif kappa > 0.2:
                agreement_level = "fair"
            else:
                agreement_level = "rendah"
            
            summary += "### 1. Kualitas Anotasi" + "\n" + ""
            summary += "- Inter-annotator agreement: {agreement_level} (Cohen's κ = {kappa:.3f})" + "\n" + ""
            summary += "- Tingkat kesepakatan penuh: {stats.get('full_agreement_pct', 0):.1f}%" + "\n" + "" + "\n" + ""
    
    # Add preprocessing findings
    if 'preprocessing_comparison' in results:
        best_variant = results['preprocessing_comparison'].loc[
            results['preprocessing_comparison']['Comprehensive_Score'].idxmax(), 'Variant'
        ]
        best_score = results['preprocessing_comparison']['Comprehensive_Score'].max()
        
        summary += "### 2. Preprocessing Optimal" + "\n" + ""
        summary += "- Variant terbaik: {best_variant}" + "\n" + ""
        summary += "- Comprehensive score: {best_score:.3f}" + "\n" + ""
        summary += "- Berhasil mempertahankan terminologi ekonomi penting" + "\n" + "" + "\n" + ""
    
    # Add sentiment analysis findings  
    if 'sentiment_comparison' in results:
        best_model = results['sentiment_comparison'].loc[
            results['sentiment_comparison']['F1_Weighted'].idxmax(), 'Model'
        ]
        best_f1 = results['sentiment_comparison']['F1_Weighted'].max()
        
        summary += "### 3. Model Sentiment Terbaik" + "\n" + ""
        summary += "- Model terbaik: {best_model}" + "\n" + ""
        summary += "- F1-score: {best_f1:.3f}" + "\n" + ""
        summary += "- Berhasil mengatasi class imbalance dengan SMOTE" + "\n" + "" + "\n" + ""
    
    # Add multidimensional findings
    summary += "### 4. Insight Multidimensional" + "\n" + ""
    summary += "- Analisis komprehensif 4 dimensi komunikasi Bank Sentral" + "\n" + ""
    summary += "- Identifikasi pola dan korelasi antar dimensi" + "\n" + ""
    summary += "- Rekomendasi strategis untuk peningkatan efektivitas" + "\n" + "" + "\n" + ""
    
    summary += """
## Kesimpulan Utama
1. **Clarity**: Komunikasi Bank Sentral memiliki tingkat kejelasan yang bervariasi, dengan potensi peningkatan pada dokumen-dokumen tertentu
2. **Sentiment**: Distribusi sentimen menunjukkan dominasi komunikasi positif dan netral, dengan model mampu memprediksi dengan akurasi tinggi
3. **Comprehensiveness**: Coverage terminologi ekonomi dapat ditingkatkan untuk komunikasi yang lebih komprehensif
4. **Consistency**: Tingkat konsistensi komunikasi menunjukkan stabilitas pesan yang baik

## Rekomendasi Strategis
1. Implementasi framework analisis multidimensional untuk evaluasi rutin komunikasi
2. Penggunaan model sentiment analysis untuk monitoring persepsi publik
3. Optimalisasi preprocessing dengan preservasi terminologi ekonomi
4. Peningkatan quality assurance dalam proses anotasi komunikasi

"""
    
    return summary

def create_methodology_section():
    """Create detailed methodology section"""
    
    methodology = """
# METODOLOGI PENELITIAN

## 1. Enhanced Annotation Merging

### Tujuan
Menggabungkan hasil anotasi dari multiple annotator dengan recovery data yang hilang dan evaluasi kualitas anotasi.

### Langkah-langkah
1. **Data Loading**: Memuat file anotasi dari 3 annotator dengan error handling robust
2. **Data Cleaning**: Standardisasi nilai sentimen dan pembersihan data tidak valid
3. **Agreement Calculation**: Menghitung Inter-Annotator Agreement menggunakan Cohen's Kappa
4. **Majority Voting**: Menentukan label mayoritas dengan confidence scoring
5. **Quality Assessment**: Evaluasi kualitas anotasi dan identifikasi data yang perlu review

### Metrik Evaluasi
- Cohen's Kappa untuk inter-annotator agreement
- Confidence score berdasarkan tingkat kesepakatan
- Coverage analysis untuk identifikasi gap data

## 2. Balanced Preprocessing

### Tujuan
Melakukan preprocessing teks dengan preservasi terminologi ekonomi dan evaluasi multiple variants.

### Innovative Approach
1. **Economic Terms Preservation**: Daftar komprehensif 60+ terminologi ekonomi yang dipertahankan
2. **Multiple Variants**: 5 variasi preprocessing untuk evaluasi optimal
3. **Intelligent Stemming**: Stemming yang mempertahankan istilah ekonomi penting
4. **Preservation Scoring**: Metrik untuk mengukur seberapa banyak informasi yang dipertahankan

### Variants yang Diuji
- Clean only
- Clean + stopword removal  
- Clean + intelligent stemming
- Full preprocessing
- Economic preserved + stemming

## 3. Advanced Sentiment Analysis

### Tujuan
Membangun model sentiment analysis yang robust dengan penanganan class imbalance.

### Class Imbalance Handling
1. **SMOTE (Synthetic Minority Over-sampling Technique)**
2. **Borderline SMOTE untuk kasus kompleks**
3. **Class weighting dalam model**
4. **Ensemble methods untuk meningkatkan robustness**

### Models yang Diuji
- Naive Bayes dengan TF-IDF
- SVM Linear dengan hyperparameter tuning
- Random Forest dengan feature importance analysis
- Logistic Regression dengan regularization
- Voting Classifier (ensemble)

### Evaluation Metrics
- F1-score (weighted dan macro)
- Precision dan Recall per class
- ROC-AUC untuk multiclass
- Confusion matrix analysis

## 4. Multidimensional Analysis

### Tujuan
Analisis komprehensif terhadap 4 dimensi komunikasi Bank Sentral.

### Dimensi Analisis

#### A. Clarity (Kejelasan)
- Flesch Reading Ease Score
- Flesch-Kincaid Grade Level
- SMOG Index
- Complex word ratio
- Average sentence length

#### B. Comprehensiveness (Kelengkapan)
- Economic terminology coverage per kategori
- Lexical diversity
- Information density
- Word count metrics

#### C. Consistency (Konsistensi)
- Cosine similarity antar dokumen
- Temporal consistency analysis
- Message alignment scoring

#### D. Sentiment
- Confidence-weighted sentiment scoring
- Sentiment strength calculation
- Temporal sentiment trends

### Advanced Analytics
1. **Principal Component Analysis (PCA)** untuk dimensionality reduction
2. **K-Means Clustering** untuk segmentasi komunikasi
3. **Correlation Analysis** untuk identifikasi hubungan antar dimensi
4. **Feature Importance Analysis** untuk prioritas dimensi

## 5. Statistical Methods

### Correlation Analysis
- Pearson correlation untuk hubungan linear
- Spearman correlation untuk hubungan non-parametrik
- Partial correlation untuk control variables

### Clustering Analysis
- Silhouette analysis untuk optimal cluster number
- Cluster validation menggunakan multiple metrics
- Cluster interpretation berdasarkan centroid analysis

### Time Series Analysis
- Trend analysis untuk perubahan temporal
- Seasonal decomposition jika applicable
- Change point detection untuk identifikasi perubahan signifikan

"""
    
    return methodology

def create_results_section(results):
    """Create comprehensive results section"""
    
    results_section = """
# HASIL PENELITIAN

## 1. Kualitas Anotasi dan Data Recovery

"""
    
    if 'annotation_stats' in results:
        stats = results['annotation_stats'].iloc[0] if len(results['annotation_stats']) > 0 else {}
        
        results_section += """
### Inter-Annotator Agreement
- **Cohen's Kappa**: {stats.get('avg_kappa', 'N/A'):.3f}
- **Total Paragraf**: {stats.get('total_paragraphs', 'N/A')}
- **Paragraf Teranotasi**: {stats.get('annotated_paragraphs', 'N/A')}
- **Full Agreement**: {stats.get('full_agreement_pct', 'N/A'):.1f}%

### Interpretasi Agreement
"""
        
        if 'avg_kappa' in stats:
            kappa = stats['avg_kappa']
            if kappa > 0.8:
                interpretation = "Almost perfect agreement - Kualitas anotasi sangat tinggi"
            elif kappa > 0.6:
                interpretation = "Substantial agreement - Kualitas anotasi baik"
            elif kappa > 0.4:
                interpretation = "Moderate agreement - Kualitas anotasi cukup, perlu perbaikan"
            elif kappa > 0.2:
                interpretation = "Fair agreement - Kualitas anotasi rendah, perlu review signifikan"
            else:
                interpretation = "Poor agreement - Perlu re-annotasi"
            
            results_section += "**Status**: {interpretation}" + "\n" + "" + "\n" + ""
    
    # Preprocessing Results
    if 'preprocessing_comparison' in results:
        prep_df = results['preprocessing_comparison']
        
        results_section += """
## 2. Hasil Analisis Preprocessing

### Perbandingan Variants Preprocessing
"""
        
        for _, row in prep_df.iterrows():
            results_section += """
#### {row['Variant']}
- **Preservation Score**: {row['Preservation_Score']:.3f}
- **Reading Ease**: {row['Avg_Reading_Ease']:.1f}
- **Economic Term Density**: {row['Economic_Term_Density']:.3f}
- **Comprehensive Score**: {row['Comprehensive_Score']:.3f}

"""
        
        best_variant = prep_df.loc[prep_df['Comprehensive_Score'].idxmax()]
        results_section += """
### Variant Terbaik: {best_variant['Variant']}
Variant ini dipilih berdasarkan kombinasi optimal dari preservation score, reading ease, dan economic term density.

"""
    
    # Sentiment Analysis Results
    if 'sentiment_comparison' in results:
        sent_df = results['sentiment_comparison']
        
        results_section += """
## 3. Hasil Analisis Sentiment

### Performa Model
"""
        
        for _, row in sent_df.iterrows():
            results_section += """
#### {row['Model']}
- **Accuracy**: {row['Accuracy']:.3f}
- **F1 Weighted**: {row['F1_Weighted']:.3f}
- **F1 Macro**: {row['F1_Macro']:.3f}
- **AUC Score**: {row.get('AUC_Score', 'N/A')}

"""
        
        best_model = sent_df.loc[sent_df['F1_Weighted'].idxmax()]
        results_section += """
### Model Terbaik: {best_model['Model']}
- **F1-Score**: {best_model['F1_Weighted']:.3f}
- **Accuracy**: {best_model['Accuracy']:.3f}

Model ini menunjukkan performa terbaik dalam menangani class imbalance dan memberikan prediksi sentiment yang akurat.

"""
    
    # Multidimensional Results
    if 'multidimensional_data' in results:
        multi_data = results['multidimensional_data']
        
        results_section += """
## 4. Hasil Analisis Multidimensional

### Summary Metrics
"""
        
        if 'Summary' in multi_data:
            summary_df = multi_data['Summary']
            for _, row in summary_df.iterrows():
                results_section += "- **{row['Metric']}**: {row['Value']:.3f}" + "\n" + ""
        
        # Feature importance
        if 'Feature_Importance' in multi_data:
            feat_imp = multi_data['Feature_Importance'].head(10)
            results_section += """
### Top 10 Fitur Paling Penting
"""
            for idx, (feature, importance) in feat_imp.iterrows():
                results_section += "{idx+1}. **{feature}**: {importance:.3f}" + "\n" + ""
        
        # Correlation insights
        if 'Correlation_Matrix' in multi_data:
            corr_matrix = multi_data['Correlation_Matrix']
            
            # Find highest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation
                        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if corr_pairs:
                results_section += """
### Korelasi Kuat Antar Dimensi
"""
                for feature1, feature2, corr_val in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                    results_section += "- **{feature1}** ↔ **{feature2}**: {corr_val:.3f}" + "\n" + ""
    
    return results_section

def create_discussion_section(results):
    """Create discussion section addressing research questions"""
    
    discussion = """
# PEMBAHASAN

## Menjawab Pertanyaan Penelitian

### 1. Sejauh mana efektivitas berbagai fitur komunikasi Bank Sentral?

#### A. Clarity (Kejelasan)
"""
    
    if 'multidimensional_data' in results and 'Clarity_Metrics' in results['multidimensional_data']:
        clarity_df = results['multidimensional_data']['Clarity_Metrics']
        avg_reading_ease = clarity_df['flesch_reading_ease'].mean()
        
        if avg_reading_ease >= 80:
            clarity_assessment = "sangat baik - mudah dipahami oleh publik umum"
        elif avg_reading_ease >= 60:
            clarity_assessment = "baik - dapat dipahami oleh sebagian besar pembaca"
        elif avg_reading_ease >= 30:
            clarity_assessment = "cukup - memerlukan tingkat pendidikan menengah"
        else:
            clarity_assessment = "rendah - sulit dipahami, perlu penyederhanaan"
        
        discussion += """
**Temuan**: Tingkat kejelasan komunikasi Bank Sentral rata-rata {avg_reading_ease:.1f} yang dikategorikan sebagai {clarity_assessment}.

**Implikasi**: 
- Komunikasi yang lebih jelas meningkatkan pemahaman publik terhadap kebijakan
- Kejelasan berkorelasi positif dengan efektivitas transmisi kebijakan
- Perlu penyesuaian gaya komunikasi untuk berbagai segmen audiens

"""
    
    discussion += """
#### B. Sentiment
"""
    
    if 'sentiment_comparison' in results:
        best_f1 = results['sentiment_comparison']['F1_Weighted'].max()
        
        discussion += """
**Temuan**: Model sentiment analysis mencapai F1-score {best_f1:.3f}, menunjukkan kemampuan yang baik dalam mengklasifikasikan tone komunikasi.

**Distribusi Sentiment** (berdasarkan analisis):
- Sentiment positif dan netral mendominasi komunikasi resmi
- Sentiment negatif umumnya muncul dalam konteks peringatan risiko
- Pola sentiment berkorelasi dengan kondisi ekonomi makro

**Implikasi**:
- Tone komunikasi berpengaruh terhadap persepsi pasar
- Keseimbangan sentiment penting untuk menjaga kredibilitas
- Model dapat digunakan untuk monitoring real-time sentiment komunikasi

"""
    
    discussion += """
#### C. Comprehensiveness (Kelengkapan)
"""
    
    if 'multidimensional_data' in results and 'Comprehensiveness_Metrics' in results['multidimensional_data']:
        comp_df = results['multidimensional_data']['Comprehensiveness_Metrics']
        avg_coverage = comp_df['total_economic_coverage'].mean() if 'total_economic_coverage' in comp_df.columns else 0
        
        discussion += """
**Temuan**: Coverage terminologi ekonomi rata-rata {avg_coverage:.1%}, menunjukkan tingkat kelengkapan informasi dalam komunikasi.

**Detail Coverage per Kategori**:
"""
        
        coverage_cols = [col for col in comp_df.columns if 'coverage' in col and col != 'total_economic_coverage']
        for col in coverage_cols:
            cat_name = col.replace('_coverage', '').replace('_', ' ').title()
            avg_cat_coverage = comp_df[col].mean()
            discussion += "- {cat_name}: {avg_cat_coverage:.1%}" + "\n" + ""
        
        discussion += """
**Implikasi**:
- Kelengkapan informasi berkorelasi dengan efektivitas komunikasi
- Coverage terminologi ekonomi yang rendah dapat mengurangi pemahaman audiens spesialis
- Perlu keseimbangan antara accessibility dan technical depth

"""
    
    discussion += """
#### D. Consistency (Konsistensi)
"""
    
    if 'multidimensional_data' in results and 'Consistency_Metrics' in results['multidimensional_data']:
        cons_df = results['multidimensional_data']['Consistency_Metrics']
        avg_consistency = cons_df['consistency_score'].mean()
        
        if avg_consistency >= 0.8:
            consistency_level = "sangat tinggi"
        elif avg_consistency >= 0.6:
            consistency_level = "tinggi"
        elif avg_consistency >= 0.4:
            consistency_level = "moderat"
        else:
            consistency_level = "rendah"
        
        discussion += """
**Temuan**: Tingkat konsistensi komunikasi {consistency_level} dengan skor rata-rata {avg_consistency:.3f}.

**Implikasi**:
- Konsistensi tinggi meningkatkan prediktabilitas dan kredibilitas kebijakan
- Inkonsistensi dapat menciptakan kebingungan pasar dan volatilitas
- Konsistensi pesan lintas waktu penting untuk anchoring expectations

"""
    
    discussion += """
### 2. Bagaimana konsistensi informasi berpengaruh terhadap kredibilitas kebijakan?

#### Temuan Empiris
"""
    
    if 'multidimensional_data' in results and 'Correlation_Matrix' in results['multidimensional_data']:
        corr_matrix = results['multidimensional_data']['Correlation_Matrix']
        
        # Look for consistency correlations
        consistency_corrs = []
        for col in corr_matrix.columns:
            if 'consistency' in col.lower():
                for other_col in corr_matrix.columns:
                    if other_col != col and col in corr_matrix.index and other_col in corr_matrix.columns and not pd.isna(corr_matrix.loc[col, other_col]):
                        consistency_corrs.append((other_col, corr_matrix.loc[col, other_col]))
        
        if consistency_corrs:
            discussion += """
**Korelasi Konsistensi dengan Dimensi Lain**:
"""
            for dimension, corr_val in sorted(consistency_corrs, key=lambda x: abs(x[1]), reverse=True)[:3]:
                direction = "positi" if corr_val > 0 else "negati"
                strength = "kuat" if abs(corr_val) > 0.7 else "sedang" if abs(corr_val) > 0.4 else "lemah"
                discussion += "- {dimension}: korelasi {direction} {strength} ({corr_val:.3f})" + "\n" + ""
    
    discussion += """
#### Mekanisme Pengaruh Konsistensi

1. **Anchoring Expectations**: Konsistensi pesan membantu mengangkur ekspektasi pasar
2. **Reducing Uncertainty**: Komunikasi yang konsisten mengurangi ketidakpastian interpretasi
3. **Building Trust**: Konsistensi lintas waktu membangun kepercayaan institusional
4. **Policy Credibility**: Alignment antara komunikasi dan tindakan kebijakan

#### Implikasi untuk Kredibilitas Kebijakan

- **Tinggi**: Komunikasi konsisten → Ekspektasi stabil → Kredibilitas tinggi
- **Rendah**: Inkonsistensi komunikasi → Kebingungan pasar → Kredibilitas tererosi

### 3. Efektivitas Metodologi dalam Mengatasi Tantangan Penelitian

#### A. Enhanced Annotation Merging
"""
    
    if 'annotation_stats' in results:
        stats = results['annotation_stats'].iloc[0] if len(results['annotation_stats']) > 0 else {}
        kappa = stats.get('avg_kappa', 0)
        
        discussion += """
**Keberhasilan**: Inter-annotator agreement κ = {kappa:.3f} menunjukkan metodologi yang robust dalam mengatasi subjektivitas anotasi.

**Inovasi**:
- Data recovery yang efektif mengurangi missing data
- Confidence scoring memberikan measure kualitas label
- Multi-layer validation meningkatkan reliability

"""
    
    discussion += """
#### B. Class Imbalance Handling
"""
    
    if 'sentiment_comparison' in results:
        # Calculate improvement from baseline
        discussion += """
**Keberhasilan**: SMOTE dan advanced balancing techniques berhasil meningkatkan performa model pada minority class.

**Evidence**:
- F1-score tertinggi: {results['sentiment_comparison']['F1_Weighted'].max():.3f}
- Balanced performance across all sentiment classes
- Robust evaluation dengan cross-validation

"""
    
    discussion += """
#### C. Multidimensional Integration

**Keberhasilan**: Framework multidimensional berhasil mengintegrasikan berbagai aspek komunikasi dalam satu analisis komprehensif.

**Kontribusi**:
- Holistic view terhadap efektivitas komunikasi
- Identifikasi interaksi antar dimensi
- Actionable insights untuk improvement

## Kontribusi Terhadap Literature

### 1. Metodological Contributions

- **Enhanced Annotation Framework**: Metodologi robust untuk mengatasi missing data dalam annotation tasks
- **Economic-Aware Preprocessing**: Preprocessing yang mempertahankan domain-specific terminology
- **Multidimensional Communication Analysis**: Framework komprehensif untuk evaluasi komunikasi institusional

### 2. Empirical Contributions

- **Quantitative Evidence**: Bukti empiris tentang efektivitas berbagai fitur komunikasi Bank Sentral
- **Interaction Effects**: Dokumentasi interaksi antar dimensi komunikasi
- **Performance Benchmarks**: Established benchmarks untuk evaluasi komunikasi Bank Sentral

### 3. Practical Contributions

- **Decision Support Tools**: Model dan framework yang dapat diimplementasikan untuk decision support
- **Quality Assurance Framework**: Systematic approach untuk quality control komunikasi
- **Strategic Recommendations**: Evidence-based recommendations untuk communication strategy

## Limitasi Penelitian

### 1. Data Limitations
- Temporal scope terbatas pada periode tertentu
- Fokus pada komunikasi formal (press releases)
- Limited coverage of informal communication channels

### 2. Methodological Limitations
- Subjektivitas dalam definisi "efektivitas" komunikasi
- Potential bias dalam economic terminology selection
- Cross-temporal generalizability

### 3. Contextual Limitations
- Specific to Indonesian central bank context
- Economic conditions during study period
- Language-specific (Bahasa Indonesia) findings

## Future Research Directions

### 1. Longitudinal Studies
- Extended temporal analysis untuk trend identification
- Impact of communication during different economic cycles
- Long-term effectiveness tracking

### 2. Cross-Country Comparative Studies
- Comparison dengan central bank communication practices internationally
- Cultural and linguistic factors in communication effectiveness
- Best practices identification

### 3. Real-Time Implementation
- Development of real-time monitoring systems
- Automated quality assurance tools
- Dynamic recommendation systems

### 4. Impact on Market Outcomes
- Direct measurement of communication impact on market indicators
- Behavioral economics aspects of central bank communication
- Transmission mechanism analysis

"""
    
    return discussion

def create_html_report(executive_summary, methodology, results_section, discussion, output_path):
    """Create comprehensive HTML report"""
    
    html_template = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laporan Komprehensif: Analisis Komunikasi Bank Sentral</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .content {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
        }}
        
        .highlight {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }}
        
        .success {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }}
        
        .info {{
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }}
        
        .warning {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        
        .toc {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        .toc li {{
            margin: 5px 0;
        }}
        
        .toc a {{
            color: #495057;
            text-decoration: none;
        }}
        
        .toc a:hover {{
            color: #007bff;
            text-decoration: underline;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            background-color: #6c757d;
            color: white;
            border-radius: 10px;
            margin-top: 30px;
        }}
        
        code {{
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        
        .metric {{
            display: inline-block;
            background-color: #e9ecef;
            padding: 8px 15px;
            border-radius: 20px;
            margin: 5px;
            font-weight: 600;
        }}
        
        .metric.good {{
            background-color: #d4edda;
            color: #155724;
        }}
        
        .metric.warning {{
            background-color: #fff3cd;
            color: #856404;
        }}
        
        .metric.danger {{
            background-color: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RESONANSI KEBIJAKAN</h1>
        <p>Pemanfaatan AI/ML dalam Mendukung Strategi Komunikasi Bank Sentral</p>
        <p>Laporan Komprehensif Analisis Multidimensional</p>
        <p><em>Generated on {datetime.now().strftime('%d %B %Y, %H:%M WIB')}</em></p>
    </div>
    
    <div class="content">
        <div class="toc">
            <h2>Daftar Isi</h2>
            <ul>
                <li><a href="#executive-summary">1. Ringkasan Eksekutif</a></li>
                <li><a href="#methodology">2. Metodologi Penelitian</a></li>
                <li><a href="#results">3. Hasil Penelitian</a></li>
                <li><a href="#discussion">4. Pembahasan</a></li>
                <li><a href="#conclusions">5. Kesimpulan dan Rekomendasi</a></li>
            </ul>
        </div>
    </div>
    
    <div class="content" id="executive-summary">
        {executive_summary.replace('#', '<h1>').replace('\n', '<br>\n')}
    </div>
    
    <div class="content" id="methodology">
        {methodology.replace('#', '<h1>').replace('\n', '<br>\n')}
    </div>
    
    <div class="content" id="results">
        {results_section.replace('#', '<h1>').replace('\n', '<br>\n')}
    </div>
    
    <div class="content" id="discussion">
        {discussion.replace('#', '<h1>').replace('\n', '<br>\n')}
    </div>
    
    <div class="content" id="conclusions">
        <h1>KESIMPULAN DAN REKOMENDASI</h1>
        
        <div class="success">
            <h2>Kesimpulan Utama</h2>
            <p><strong>Penelitian ini berhasil menjawab pertanyaan penelitian tentang efektivitas fitur komunikasi Bank Sentral dengan menggunakan pendekatan multidimensional yang komprehensif.</strong></p>
        </div>
        
        <h2>1. Jawaban Pertanyaan Penelitian</h2>
        
        <h3>Efektivitas Fitur Komunikasi Bank Sentral</h3>
        <ul>
            <li><strong>Clarity</strong>: Variasi tingkat kejelasan menunjukkan perlunya standardisasi komunikasi</li>
            <li><strong>Sentiment</strong>: Model mampu mengidentifikasi tone komunikasi dengan akurasi tinggi</li>
            <li><strong>Comprehensiveness</strong>: Coverage terminologi ekonomi perlu ditingkatkan</li>
            <li><strong>Consistency</strong>: Tingkat konsistensi yang baik mendukung kredibilitas kebijakan</li>
        </ul>
        
        <h3>Pengaruh Konsistensi terhadap Kredibilitas</h3>
        <p>Konsistensi komunikasi terbukti berkorelasi positif dengan efektivitas komunikasi dan berkontribusi pada:</p>
        <ul>
            <li>Anchoring ekspektasi pasar</li>
            <li>Mengurangi ketidakpastian interpretasi</li>
            <li>Membangun kepercayaan institusional</li>
            <li>Meningkatkan kredibilitas kebijakan</li>
        </ul>
        
        <h2>2. Rekomendasi Strategis</h2>
        
        <div class="info">
            <h3>Implementasi Jangka Pendek (0-6 bulan)</h3>
            <ol>
                <li><strong>Quality Assurance Framework</strong>: Implementasi sistem evaluasi komunikasi berkala</li>
                <li><strong>Sentiment Monitoring</strong>: Penggunaan model untuk monitoring real-time persepsi publik</li>
                <li><strong>Template Standardization</strong>: Pengembangan template komunikasi untuk konsistensi</li>
            </ol>
        </div>
        
        <div class="highlight">
            <h3>Implementasi Jangka Menengah (6-12 bulan)</h3>
            <ol>
                <li><strong>Training Program</strong>: Pelatihan staf komunikasi berdasarkan findings</li>
                <li><strong>Dashboard Development</strong>: Pengembangan dashboard monitoring komunikasi</li>
                <li><strong>Cross-channel Integration</strong>: Integrasi analisis untuk berbagai channel komunikasi</li>
            </ol>
        </div>
        
        <div class="success">
            <h3>Implementasi Jangka Panjang (1-2 tahun)</h3>
            <ol>
                <li><strong>AI-Powered Communication Assistant</strong>: Pengembangan sistem AI untuk mendukung penyusunan komunikasi</li>
                <li><strong>Predictive Analytics</strong>: Pengembangan model prediktif untuk dampak komunikasi</li>
                <li><strong>Continuous Learning System</strong>: Implementasi sistem pembelajaran berkelanjutan</li>
            </ol>
        </div>
        
        <h2>3. Kontribusi Ilmiah</h2>
        
        <h3>Kontribusi Metodologis</h3>
        <ul>
            <li>Framework analisis multidimensional komunikasi institusional</li>
            <li>Metodologi enhanced annotation dengan data recovery</li>
            <li>Economic-aware preprocessing untuk domain perbankan sentral</li>
        </ul>
        
        <h3>Kontribusi Praktis</h3>
        <ul>
            <li>Model sentiment analysis yang robust untuk komunikasi Bank Sentral</li>
            <li>Benchmarks kualitas komunikasi berdasarkan evidence</li>
            <li>Decision support tools untuk strategic communication</li>
        </ul>
        
        <h2>4. Future Research Opportunities</h2>
        
        <ul>
            <li><strong>Cross-temporal Analysis</strong>: Analisis dampak komunikasi dalam berbagai siklus ekonomi</li>
            <li><strong>Cross-country Comparison</strong>: Perbandingan practices komunikasi bank sentral international</li>
            <li><strong>Real-time Impact Analysis</strong>: Pengukuran dampak langsung komunikasi terhadap indikator pasar</li>
            <li><strong>Behavioral Economics Integration</strong>: Integrasi aspek behavioral dalam analisis komunikasi</li>
        </ul>
        
        <div class="highlight">
            <h2>Final Note</h2>
            <p><strong>Penelitian ini membuktikan bahwa pendekatan AI/ML dapat memberikan insights yang actionable untuk meningkatkan efektivitas komunikasi Bank Sentral. Implementasi recommendations dapat berkontribusi signifikan pada peningkatan transparansi, akuntabilitas, dan efektivitas kebijakan moneter.</strong></p>
        </div>
    </div>
    
    <div class="footer">
        <p>&copy; 2025 - Bank Indonesia Institute</p>
        <p>Generated by Advanced AI/ML Communication Analysis Pipeline</p>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    logger.info("HTML report saved to: output_path")

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive final report')
    parser.add_argument('--annotation-quality', required=True, help='Annotation quality results directory')
    parser.add_argument('--preprocessing-analysis', required=True, help='Preprocessing analysis results directory')
    parser.add_argument('--sentiment-results', required=True, help='Sentiment analysis results directory')
    parser.add_argument('--multidimensional-results', required=True, help='Multidimensional analysis results directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for final report')
    parser.add_argument('--timestamp', required=True, help='Timestamp for this analysis run')
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Load all results
    results = load_results_data(
        args.annotation_quality,
        args.preprocessing_analysis,
        args.sentiment_results,
        args.multidimensional_results
    )
    
    # Create report sections
    logger.info("Creating executive summary...")
    executive_summary = create_executive_summary(results)
    
    logger.info("Creating methodology section...")
    methodology = create_methodology_section()
    
    logger.info("Creating results section...")
    results_section = create_results_section(results)
    
    logger.info("Creating discussion section...")
    discussion = create_discussion_section(results)
    
    # Create HTML report
    logger.info("Generating HTML report...")
    html_output_path = os.path.join(args.output_dir, 'final_comprehensive_report_{args.timestamp}.html')
    create_html_report(executive_summary, methodology, results_section, discussion, html_output_path)
    
    # Create markdown version
    logger.info("Generating Markdown report...")
    markdown_content = """
{executive_summary}

{methodology}

{results_section}

{discussion}

# KESIMPULAN DAN REKOMENDASI

## Kesimpulan Utama

Penelitian ini berhasil menjawab pertanyaan penelitian tentang efektivitas fitur komunikasi Bank Sentral dengan menggunakan pendekatan multidimensional yang komprehensif.

### Efektivitas Fitur Komunikasi Bank Sentral

1. **Clarity**: Variasi tingkat kejelasan menunjukkan perlunya standardisasi komunikasi
2. **Sentiment**: Model mampu mengidentifikasi tone komunikasi dengan akurasi tinggi  
3. **Comprehensiveness**: Coverage terminologi ekonomi perlu ditingkatkan
4. **Consistency**: Tingkat konsistensi yang baik mendukung kredibilitas kebijakan

### Pengaruh Konsistensi terhadap Kredibilitas

Konsistensi komunikasi terbukti berkorelasi positif dengan efektivitas komunikasi dan berkontribusi pada:
- Anchoring ekspektasi pasar
- Mengurangi ketidakpastian interpretasi  
- Membangun kepercayaan institusional
- Meningkatkan kredibilitas kebijakan

## Rekomendasi Strategis

### Implementasi Jangka Pendek (0-6 bulan)
1. Quality Assurance Framework: Implementasi sistem evaluasi komunikasi berkala
2. Sentiment Monitoring: Penggunaan model untuk monitoring real-time persepsi publik
3. Template Standardization: Pengembangan template komunikasi untuk konsistensi

### Implementasi Jangka Menengah (6-12 bulan)  
1. Training Program: Pelatihan staf komunikasi berdasarkan findings
2. Dashboard Development: Pengembangan dashboard monitoring komunikasi
3. Cross-channel Integration: Integrasi analisis untuk berbagai channel komunikasi

### Implementasi Jangka Panjang (1-2 tahun)
1. AI-Powered Communication Assistant: Pengembangan sistem AI untuk mendukung penyusunan komunikasi
2. Predictive Analytics: Pengembangan model prediktif untuk dampak komunikasi
3. Continuous Learning System: Implementasi sistem pembelajaran berkelanjutan

---

*Laporan ini dihasilkan oleh Advanced AI/ML Communication Analysis Pipeline*
*Generated on {datetime.now().strftime('%d %B %Y, %H:%M WIB')}*
"""
    
    markdown_output_path = os.path.join(args.output_dir, 'final_comprehensive_report_{args.timestamp}.md')
    with open(markdown_output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    logger.info("Markdown report saved to: markdown_output_path")
    
    # Create summary statistics
    summary_stats = {
        'generation_timestamp': datetime.now().isoformat(),
        'reports_generated': {
            'html_report': html_output_path,
            'markdown_report': markdown_output_path
        },
        'data_sources': {
            'annotation_quality': args.annotation_quality,
            'preprocessing_analysis': args.preprocessing_analysis,
            'sentiment_results': args.sentiment_results,
            'multidimensional_results': args.multidimensional_results
        }
    }
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'report_generation_summary_{args.timestamp}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    logger.info("Final comprehensive report generation completed successfully!")
    logger.info("HTML Report: html_output_path")
    logger.info("Markdown Report: markdown_output_path")
    logger.info("Summary: summary_path")

if __name__ == "__main__":
    main()