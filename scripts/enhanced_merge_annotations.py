#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_merge_annotations.py - Fixed version with proper sentiment merging
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/enhanced_merge_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('enhanced_merge_annotations')

def ensure_directory(directory):
    """Pastikan direktori ada, buat jika tidak ada."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Direktori dibuat: {directory}")

def load_and_separate_annotations(file_path):
    """
    Load file anotasi dan pisahkan data unique vs overlap
    """
    logger.info(f"Memuat file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File tidak ditemukan: {file_path}")
        return None, None
    
    try:
        # Baca file Excel
        xls = pd.ExcelFile(file_path)
        logger.info(f"Sheet yang tersedia: {xls.sheet_names}")
        
        # Gunakan sheet 'Anotasi'
        if 'Anotasi' in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name='Anotasi')
            logger.info(f"Menggunakan sheet 'Anotasi'")
        else:
            df = pd.read_excel(file_path, sheet_name=0)
            logger.info(f"Menggunakan sheet pertama: '{xls.sheet_names[0]}'")
        
        logger.info(f"Data dimuat: {len(df)} baris, {len(df.columns)} kolom")
        logger.info(f"Kolom: {list(df.columns)}")
        
        # Handle missing text column for Annotator_3
        if 'Teks_Paragraf' not in df.columns:
            # For Annotator_3, the text is in 'Rapat Dewan Gubernur (RDG) Bank Indonesia' column
            text_col = 'Rapat Dewan Gubernur (RDG) Bank Indonesia'
            if text_col in df.columns:
                logger.info(f"Menggunakan kolom '{text_col}' sebagai Teks_Paragraf")
                df['Teks_Paragraf'] = df[text_col]
            else:
                logger.warning("Tidak dapat menemukan kolom teks yang sesuai")
                df['Teks_Paragraf'] = ""
        
        # Tambahkan source info
        annotator_name = os.path.basename(file_path).replace('.xlsx', '')
        df['Source_File'] = annotator_name
        
        # Clean data BEFORE separating
        df = clean_annotation_data(df)
        
        # Debug: Check sentiment values after cleaning
        if 'Sentimen' in df.columns:
            logger.info(f"Sentiment distribution for {annotator_name}: {df['Sentimen'].value_counts().to_dict()}")
            logger.info(f"Null sentiments: {df['Sentimen'].isna().sum()}")
        
        # Pisahkan data berdasarkan Is_Overlap
        if 'Is_Overlap' in df.columns:
            unique_data = df[df['Is_Overlap'] == False].copy()
            overlap_data = df[df['Is_Overlap'] == True].copy()
            
            logger.info(f"Data unik: {len(unique_data)} baris")
            logger.info(f"Data overlap: {len(overlap_data)} baris")
            
            # Debug: Check if sentiments are preserved
            if 'Sentimen' in unique_data.columns:
                logger.info(f"Unique data sentiments: {unique_data['Sentimen'].value_counts().to_dict()}")
            
            return unique_data, overlap_data
        else:
            logger.warning("Kolom 'Is_Overlap' tidak ditemukan, menggunakan semua data sebagai unique")
            return df, pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error memuat file {file_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def clean_annotation_data(df):
    """Membersihkan data anotasi dengan lebih hati-hati"""
    logger.info("Membersihkan data anotasi...")
    
    # First, let's see what we have
    if 'Sentimen' in df.columns:
        logger.info(f"Raw sentiment values before cleaning: {df['Sentimen'].value_counts(dropna=False).to_dict()}")
    
    # Bersihkan whitespace untuk semua object columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Don't convert to string first - this might convert NaN to 'nan'
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Standarisasi nilai sentimen
    if 'Sentimen' in df.columns:
        # First, handle actual string values
        df['Sentimen'] = df['Sentimen'].apply(lambda x: x.strip().replace('\n', '').title() if isinstance(x, str) else x)
        
        # Koreksi typo umum
        df['Sentimen'] = df['Sentimen'].replace({
            'Postif': 'Positif',
            'Negtif': 'Negatif',
            'Netrl': 'Netral',
            'Neural': 'Netral',
            'nan': np.nan,
            'None': np.nan,
            '': np.nan
        })
        
        # Valid sentiments
        valid_sentiments = ['Positif', 'Netral', 'Negatif']
        
        # Check for invalid values
        mask_valid = df['Sentimen'].isin(valid_sentiments)
        mask_null = df['Sentimen'].isna()
        invalid_mask = ~(mask_valid | mask_null)
        
        if invalid_mask.any():
            invalid_values = df.loc[invalid_mask, 'Sentimen'].value_counts()
            logger.warning(f"Nilai sentimen tidak valid: {invalid_values.to_dict()}")
            df.loc[invalid_mask, 'Sentimen'] = np.nan
        
        # Log cleaned sentiment distribution
        logger.info(f"Sentiment values after cleaning: {df['Sentimen'].value_counts(dropna=False).to_dict()}")
    
    # Remove rows with null sentiment only if all sentiment values are null
    if 'Sentimen' in df.columns:
        before_count = len(df)
        null_sentiment_count = df['Sentimen'].isna().sum()
        
        # Only remove if it's a significant portion and not all data
        if null_sentiment_count > 0 and null_sentiment_count < len(df):
            df = df[df['Sentimen'].notna()].copy()
            after_count = len(df)
            logger.info(f"Removed {before_count - after_count} rows with null sentiment")
        elif null_sentiment_count == len(df):
            logger.error("All sentiment values are null! Keeping data but results will be limited.")
    
    logger.info(f"Data dibersihkan: {len(df)} baris tersisa")
    return df

def analyze_overlap_data(overlap_dfs):
    """Analisis data overlap untuk quality check"""
    logger.info("Menganalisis data overlap...")
    
    if not overlap_dfs or all(df.empty for df in overlap_dfs):
        logger.warning("Tidak ada data overlap untuk dianalisis")
        return None, 0
    
    # Combine all overlap data
    all_overlap = pd.concat([df for df in overlap_dfs if not df.empty], ignore_index=True)
    
    if len(all_overlap) > 0:
        logger.info(f"Total data overlap: {len(all_overlap)} baris")
        
        # Debug overlap sentiment distribution
        if 'Sentimen' in all_overlap.columns:
            logger.info(f"Overlap sentiment distribution: {all_overlap['Sentimen'].value_counts().to_dict()}")
        
        # Check agreement on overlap data
        if len([df for df in overlap_dfs if not df.empty]) >= 2:
            try:
                overlap_pivot = all_overlap.pivot_table(
                    index='ID_Paragraf',
                    columns='Source_File', 
                    values='Sentimen',
                    aggfunc='first'
                )
                
                # Calculate agreement on overlap
                def check_overlap_agreement(row):
                    values = row.dropna().unique()
                    return len(values) == 1 if len(values) > 0 else False
                
                overlap_agreement = overlap_pivot.apply(check_overlap_agreement, axis=1)
                agreement_pct = (overlap_agreement.sum() / len(overlap_agreement)) * 100 if len(overlap_agreement) > 0 else 0
                
                logger.info(f"Agreement pada overlap data: {agreement_pct:.1f}%")
                
                return overlap_pivot, agreement_pct
            except Exception as e:
                logger.error(f"Error in overlap analysis: {e}")
                return None, 0
    
    return None, 0

def merge_all_annotations(unique_dfs, overlap_dfs, id_col='ID_Paragraf', sentiment_col='Sentimen'):
    """Merge ALL annotations (both unique and overlap) dengan benar"""
    logger.info("Menggabungkan semua annotations (unique + overlap)...")
    
    # Combine all data (unique + overlap)
    all_dfs = []
    
    # Add unique data
    for df in unique_dfs:
        if not df.empty:
            all_dfs.append(df)
    
    # Add overlap data
    for df in overlap_dfs:
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        logger.error("Tidak ada data untuk digabungkan")
        return None, None
    
    # Combine all data
    all_data = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total combined data: {len(all_data)} baris")
    
    # Debug: Check sentiment in combined data
    if sentiment_col in all_data.columns:
        logger.info(f"Combined data sentiment distribution: {all_data[sentiment_col].value_counts().to_dict()}")
        logger.info(f"Combined data sources: {all_data['Source_File'].value_counts().to_dict()}")
    
    # Create pivot untuk sentiment analysis
    try:
        sentiment_pivot = all_data.pivot_table(
            index=id_col,
            columns='Source_File',
            values=sentiment_col,
            aggfunc='first'
        )
        
        logger.info(f"Pivot table created: {sentiment_pivot.shape}")
        logger.info(f"Pivot columns: {list(sentiment_pivot.columns)}")
        
        # Debug: Check if pivot has data
        for col in sentiment_pivot.columns:
            non_null = sentiment_pivot[col].notna().sum()
            logger.info(f"Non-null values in {col}: {non_null}")
        
    except Exception as e:
        logger.error(f"Error creating pivot table: {e}")
        return None, None
    
    # Calculate Cohen's Kappa for overlap data only
    kappa_scores = {}
    if overlap_dfs:
        overlap_data = pd.concat([df for df in overlap_dfs if not df.empty], ignore_index=True)
        if not overlap_data.empty:
            try:
                overlap_pivot = overlap_data.pivot_table(
                    index=id_col,
                    columns='Source_File',
                    values=sentiment_col,
                    aggfunc='first'
                )
                
                annotators = [col for col in overlap_pivot.columns if 'Annotator' in col]
                
                for i in range(len(annotators)):
                    for j in range(i+1, len(annotators)):
                        ann1, ann2 = annotators[i], annotators[j]
                        
                        valid_mask = overlap_pivot[ann1].notna() & overlap_pivot[ann2].notna()
                        
                        if valid_mask.sum() > 0:
                            y1 = overlap_pivot.loc[valid_mask, ann1]
                            y2 = overlap_pivot.loc[valid_mask, ann2]
                            
                            try:
                                kappa = cohen_kappa_score(y1, y2)
                                kappa_scores[f"{ann1}_vs_{ann2}"] = kappa
                                logger.info(f"Kappa score for {ann1} vs {ann2}: {kappa:.3f}")
                            except Exception as e:
                                logger.warning(f"Error calculating kappa for {ann1} vs {ann2}: {e}")
            except Exception as e:
                logger.warning(f"Error in kappa calculation: {e}")
    
    # Calculate majority sentiment
    def get_majority_sentiment(row):
        values = row.dropna()
        if len(values) == 0:
            return np.nan
        
        counter = Counter(values)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else np.nan
    
    sentiment_pivot['majority_sentiment'] = sentiment_pivot.apply(get_majority_sentiment, axis=1)
    logger.info(f"Majority sentiment calculated: {sentiment_pivot['majority_sentiment'].value_counts().to_dict()}")
    
    # Calculate agreement
    annotator_cols = [col for col in sentiment_pivot.columns if 'Annotator' in col]
    
    def check_full_agreement(row):
        values = [row[col] for col in annotator_cols if pd.notna(row[col])]
        return len(set(values)) == 1 if len(values) > 1 else False
    
    sentiment_pivot['full_agreement'] = sentiment_pivot.apply(check_full_agreement, axis=1)
    
    # Calculate confidence
    def calculate_confidence(row):
        values = [row[col] for col in annotator_cols if pd.notna(row[col])]
        
        if not values:
            return 0.0
        elif len(values) == 1:
            return 1.0  # Only one annotator, so 100% "agreement"
        
        counter = Counter(values)
        most_common_count = counter.most_common(1)[0][1] if counter else 0
        confidence = most_common_count / len(values)
        return confidence
    
    sentiment_pivot['confidence_score'] = sentiment_pivot.apply(calculate_confidence, axis=1)
    
    # Calculate statistics
    full_agreement_count = sentiment_pivot['full_agreement'].sum()
    avg_kappa = np.mean(list(kappa_scores.values())) if kappa_scores else 0.0
    
    # Count how many paragraphs have at least one annotation
    annotated_mask = sentiment_pivot[annotator_cols].notna().any(axis=1)
    annotated_count = annotated_mask.sum()
    
    stats = {
        'total_paragraphs': len(sentiment_pivot),
        'annotated_paragraphs': annotated_count,
        'full_agreement_count': full_agreement_count,
        'full_agreement_pct': (full_agreement_count / annotated_count * 100) if annotated_count > 0 else 0,
        'kappa_scores': kappa_scores,
        'avg_kappa': avg_kappa
    }
    
    logger.info(f"Agreement Statistics:")
    logger.info(f"- Total paragraf: {stats['total_paragraphs']}")
    logger.info(f"- Paragraf teranotasi: {stats['annotated_paragraphs']}")
    logger.info(f"- Full agreement: {stats['full_agreement_count']} ({stats['full_agreement_pct']:.2f}%)")
    logger.info(f"- Average Cohen's Kappa: {stats['avg_kappa']:.3f}")
    
    return sentiment_pivot, stats

def create_final_merged_dataframe(sentiment_pivot, all_data, 
                                id_col='ID_Paragraf', 
                                sentiment_col='Sentimen',
                                topic_col='Topik_Utama',
                                text_col='Teks_Paragraf'):
    """Buat dataframe final yang merged dengan benar"""
    logger.info("Membuat dataframe final...")
    
    # Start with sentiment pivot data
    result = sentiment_pivot.reset_index()
    
    # Rename columns appropriately
    annotator_cols = [col for col in sentiment_pivot.columns if 'Annotator' in col]
    
    # Keep the sentiment data from pivot
    result = result.rename(columns={
        'majority_sentiment': f'{sentiment_col}_Majority',
        'confidence_score': 'Confidence_Score'
    })
    
    # Add individual annotator sentiments with proper naming
    for col in annotator_cols:
        result = result.rename(columns={col: f'{sentiment_col}_{col}'})
    
    # Add agreement status
    result['Agreement_Status'] = result.apply(
        lambda row: 'Full' if row.get('full_agreement', False) else 
                   'Partial' if pd.notna(row[f'{sentiment_col}_Majority']) else 'None',
        axis=1
    )
    
    # Remove the temporary full_agreement column
    if 'full_agreement' in result.columns:
        result = result.drop('full_agreement', axis=1)
    
    # Get metadata from the original data
    metadata_cols = [id_col, text_col, topic_col, 'Tanggal', 'Judul', 'URL', 
                     'ID_Dokumen', 'Paragraf_Ke', 'Total_Paragraf']
    existing_metadata_cols = [col for col in metadata_cols if col in all_data.columns]
    
    # Get first occurrence of each paragraph
    metadata_df = all_data[existing_metadata_cols].drop_duplicates(subset=[id_col])
    
    # Merge metadata
    result = result.merge(metadata_df, on=id_col, how='left')
    
    # Add topic majority if available
    if topic_col in all_data.columns:
        try:
            topic_pivot = all_data.pivot_table(
                index=id_col,
                columns='Source_File',
                values=topic_col,
                aggfunc='first'
            )
            
            # Majority topic
            def get_majority_topic(row):
                values = row.dropna()
                if len(values) == 0:
                    return np.nan
                counter = Counter(values)
                most_common = counter.most_common(1)
                return most_common[0][0] if most_common else np.nan
            
            topic_pivot['majority_topic'] = topic_pivot.apply(get_majority_topic, axis=1)
            result = result.merge(
                topic_pivot[['majority_topic']].rename(columns={'majority_topic': f'{topic_col}_Majority'}),
                left_on=id_col,
                right_index=True,
                how='left'
            )
            
            # Individual topics
            topic_annotator_cols = [col for col in topic_pivot.columns if 'Annotator' in col]
            for col in topic_annotator_cols:
                result = result.merge(
                    topic_pivot[[col]].rename(columns={col: f'{topic_col}_{col}'}),
                    left_on=id_col,
                    right_index=True,
                    how='left'
                )
        except Exception as e:
            logger.warning(f"Error processing topic data: {e}")
    
    # Final check
    logger.info(f"Final merged dataframe: {len(result)} rows, {len(result.columns)} columns")
    logger.info(f"Columns in final dataframe: {list(result.columns)}")
    
    # Debug: Check if sentiment columns have data
    sentiment_cols = [col for col in result.columns if 'Sentimen' in col]
    for col in sentiment_cols:
        non_null = result[col].notna().sum()
        logger.info(f"Non-null values in {col}: {non_null}")
    
    return result

def create_quality_report(merged_df, agreement_stats, output_dir):
    """Membuat laporan kualitas dengan error handling"""
    logger.info("Membuat laporan kualitas anotasi...")
    
    ensure_directory(output_dir)
    
    # Debug: Check what columns we have
    logger.info(f"Columns in merged_df: {list(merged_df.columns)}")
    
    # Check if we have valid sentiment data
    sentiment_majority_col = [col for col in merged_df.columns if 'Sentimen_Majority' in col]
    has_sentiment = sentiment_majority_col and not merged_df[sentiment_majority_col[0]].dropna().empty
    
    logger.info(f"Has sentiment data: {has_sentiment}")
    if has_sentiment:
        logger.info(f"Sentiment distribution: {merged_df[sentiment_majority_col[0]].value_counts().to_dict()}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Agreement Status
    plt.subplot(2, 3, 1)
    if 'Agreement_Status' in merged_df.columns:
        agreement_counts = merged_df['Agreement_Status'].value_counts()
        if not agreement_counts.empty:
            plt.pie(agreement_counts.values, labels=agreement_counts.index, autopct='%1.1f%%')
            plt.title('Agreement Status Distribution')
        else:
            plt.text(0.5, 0.5, 'No Agreement\nData Available', ha='center', va='center')
            plt.title('Agreement Status Distribution')
    
    # Confidence Scores
    plt.subplot(2, 3, 2)
    if 'Confidence_Score' in merged_df.columns:
        confidence_scores = merged_df['Confidence_Score'].dropna()
        if len(confidence_scores) > 0:
            plt.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Confidence Score Distribution')
        else:
            plt.text(0.5, 0.5, 'No Valid\nConfidence Scores', ha='center', va='center')
            plt.title('Confidence Score Distribution')
    
    # Sentiment Distribution
    plt.subplot(2, 3, 3)
    if has_sentiment:
        sentiment_counts = merged_df[sentiment_majority_col[0]].value_counts()
        plt.bar(sentiment_counts.index, sentiment_counts.values)
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Sentiment Distribution')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No Valid\nSentiment Data', ha='center', va='center')
        plt.title('Sentiment Distribution')
    
    # Kappa Scores
    plt.subplot(2, 3, 4)
    if agreement_stats['kappa_scores']:
        kappa_pairs = list(agreement_stats['kappa_scores'].keys())
        kappa_values = list(agreement_stats['kappa_scores'].values())
        plt.bar(range(len(kappa_pairs)), kappa_values)
        plt.xlabel('Annotator Pairs')
        plt.ylabel("Cohen's Kappa")
        plt.title('Inter-Annotator Agreement')
        plt.xticks(range(len(kappa_pairs)), [pair.replace('_vs_', '\nvs ') for pair in kappa_pairs])
        
        # Add interpretation line
        plt.axhline(y=0.6, color='g', linestyle='--', label='Substantial')
        plt.axhline(y=0.4, color='y', linestyle='--', label='Moderate')
        plt.axhline(y=0.2, color='r', linestyle='--', label='Fair')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No Overlap Data\nfor Kappa Calculation', ha='center', va='center')
        plt.title('Inter-Annotator Agreement')
    
    # Data Coverage
    plt.subplot(2, 3, 5)
    annotator_cols = [col for col in merged_df.columns if 'Sentimen_Annotator' in col]
    if annotator_cols:
        coverage_data = []
        for col in annotator_cols:
            coverage = (merged_df[col].notna().sum() / len(merged_df)) * 100
            annotator_name = col.replace('Sentimen_', '')
            coverage_data.append((annotator_name, coverage))
        
        if coverage_data:
            annotators, coverages = zip(*coverage_data)
            plt.bar(annotators, coverages)
            plt.xlabel('Annotator')
            plt.ylabel('Coverage (%)')
            plt.title('Data Coverage by Annotator')
            plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No Coverage\nData Available', ha='center', va='center')
        plt.title('Data Coverage')
    
    # Summary Stats
    plt.subplot(2, 3, 6)
    
    # Calculate actual values with proper checking
    avg_confidence = merged_df['Confidence_Score'].mean() if 'Confidence_Score' in merged_df.columns else 0
    
    stats_text = f"""
Total Paragraphs: {len(merged_df)}
Annotated: {agreement_stats.get('annotated_paragraphs', 0)}
Full Agreement: {agreement_stats.get('full_agreement_pct', 0):.1f}%
Avg Kappa: {agreement_stats.get('avg_kappa', 0):.3f}
Avg Confidence: {avg_confidence:.3f}
"""
    plt.text(0.1, 0.5, stats_text, ha='left', va='center', fontsize=10)
    plt.title('Summary Statistics')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'annotation_quality_report.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate text report
    report_content = f"""
LAPORAN KUALITAS ANOTASI (OVERLAP-AWARE)
========================================

1. STATISTIK KESELURUHAN
   - Total Paragraf: {len(merged_df)}
   - Paragraf Teranotasi: {agreement_stats.get('annotated_paragraphs', 0)}
   - Coverage: {(agreement_stats.get('annotated_paragraphs', 0)/len(merged_df)*100):.2f}% if len(merged_df) > 0 else 0

2. INTER-ANNOTATOR AGREEMENT
   - Full Agreement: {agreement_stats.get('full_agreement_count', 0)} ({agreement_stats.get('full_agreement_pct', 0):.2f}%)
   - Average Cohen's Kappa: {agreement_stats.get('avg_kappa', 0):.3f}

3. KAPPA SCORES PER PASANGAN ANNOTATOR
"""
    
    if agreement_stats.get('kappa_scores'):
        for pair, kappa in agreement_stats['kappa_scores'].items():
            report_content += f"   - {pair}: {kappa:.3f}\n"
    else:
        report_content += "   - Tidak ada data overlap untuk menghitung Kappa\n"
    
    report_content += f"""
4. DISTRIBUSI CONFIDENCE SCORE
"""
    
    if 'Confidence_Score' in merged_df.columns and merged_df['Confidence_Score'].notna().any():
        report_content += f"""   - Mean: {merged_df['Confidence_Score'].mean():.3f}
   - Median: {merged_df['Confidence_Score'].median():.3f}
   - Std: {merged_df['Confidence_Score'].std():.3f}
"""
    else:
        report_content += "   - Tidak ada data confidence score\n"
    
    report_content += """
5. DISTRIBUSI SENTIMEN (MAJORITY)
"""
    
    if has_sentiment:
        sentiment_dist = merged_df[sentiment_majority_col[0]].value_counts()
        total_sentiment = sentiment_dist.sum()
        for sentiment, count in sentiment_dist.items():
            pct = (count / total_sentiment) * 100 if total_sentiment > 0 else 0
            report_content += f"   - {sentiment}: {count} ({pct:.2f}%)\n"
    else:
        report_content += "   - Tidak ada data sentimen yang valid\n"
    
    # Add interpretation for Kappa score
    if agreement_stats.get('avg_kappa', 0) > 0:
        kappa_val = agreement_stats['avg_kappa']
        if kappa_val >= 0.81:
            interpretation = "Almost perfect agreement"
        elif kappa_val >= 0.61:
            interpretation = "Substantial agreement"
        elif kappa_val >= 0.41:
            interpretation = "Moderate agreement"
        elif kappa_val >= 0.21:
            interpretation = "Fair agreement"
        else:
            interpretation = "Poor agreement"
        
        report_content += f"""
6. INTERPRETASI AGREEMENT
   - Cohen's Kappa: {kappa_val:.3f} ({interpretation})
"""
    
    with open(os.path.join(output_dir, 'annotation_quality_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info("Laporan kualitas selesai dibuat")

def main():
    parser = argparse.ArgumentParser(description='Enhanced annotation merger with overlap handling')
    parser.add_argument('--input-files', nargs='+', required=True, help='List of annotation files')
    parser.add_argument('--output-file', required=True, help='Output merged annotations file')
    parser.add_argument('--output-dir', default='output/enhanced_merge', help='Output directory')
    parser.add_argument('--id-col', default='ID_Paragraf', help='ID column name')
    parser.add_argument('--sentiment-col', default='Sentimen', help='Sentiment column name')
    parser.add_argument('--topic-col', default='Topik_Utama', help='Topic column name')
    parser.add_argument('--text-col', default='Teks_Paragraf', help='Text column name')
    
    args = parser.parse_args()
    
    # Create directories
    ensure_directory('logs')
    ensure_directory(os.path.dirname(args.output_file))
    ensure_directory(args.output_dir)
    
    # Load and separate data
    unique_dfs = []
    overlap_dfs = []
    
    for file_path in args.input_files:
        unique_data, overlap_data = load_and_separate_annotations(file_path)
        if unique_data is not None and len(unique_data) > 0:
            unique_dfs.append(unique_data)
        if overlap_data is not None and len(overlap_data) > 0:
            overlap_dfs.append(overlap_data)
    
    if not unique_dfs and not overlap_dfs:
        logger.error("Tidak ada data yang berhasil dimuat")
        return
    
    logger.info(f"Berhasil memuat data dari {len(unique_dfs)} file (unique) dan {len(overlap_dfs)} file (overlap)")
    
    # Merge ALL annotations (unique + overlap)
    sentiment_pivot, agreement_stats = merge_all_annotations(
        unique_dfs, overlap_dfs, args.id_col, args.sentiment_col
    )
    
    if sentiment_pivot is None:
        logger.error("Gagal membuat sentiment pivot")
        return
    
    # Combine all data for metadata
    all_data_list = []
    for df in unique_dfs:
        if not df.empty:
            all_data_list.append(df)
    for df in overlap_dfs:
        if not df.empty:
            all_data_list.append(df)
    
    all_data = pd.concat(all_data_list, ignore_index=True)
    
    # Create final dataframe
    merged_df = create_final_merged_dataframe(
        sentiment_pivot, all_data,
        args.id_col, args.sentiment_col, args.topic_col, args.text_col
    )
    
    # Save results
    merged_df.to_excel(args.output_file, index=False)
    logger.info(f"Hasil gabungan disimpan di: {args.output_file}")
    
    # Debug: Check saved file
    logger.info(f"Saved columns: {list(merged_df.columns)}")
    sentiment_cols = [col for col in merged_df.columns if 'Sentimen' in col]
    for col in sentiment_cols:
        non_null = merged_df[col].notna().sum()
        logger.info(f"Non-null values in saved {col}: {non_null}")
    
    # Create quality report
    create_quality_report(merged_df, agreement_stats, args.output_dir)
    
    # Save agreement statistics
    stats_df = pd.DataFrame([agreement_stats])
    stats_df.to_excel(os.path.join(args.output_dir, 'agreement_statistics.xlsx'), index=False)
    
    logger.info("Enhanced merge annotations (overlap-aware) selesai!")

if __name__ == "__main__":
    main()