#!/usr/bin/env python3
"""
comprehensive_time_series_analysis.py - Time series analysis for central bank communication
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import signal
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

class CommunicationTimeSeriesAnalyzer:
    """Comprehensive time series analysis for central bank communication effectiveness"""
    
    def __init__(self, df, date_col='Tanggal', text_col='Teks_Paragraf'):
        self.df = df.copy()
        self.date_col = date_col
        self.text_col = text_col
        self.prepare_temporal_data()
        
    def prepare_temporal_data(self):
        """Prepare data for time series analysis"""
        
        # Convert date column
        if self.date_col in self.df.columns:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
            self.df = self.df.dropna(subset=[self.date_col])
            self.df = self.df.sort_values(self.date_col)
        else:
            # Create synthetic dates if no date column
            print("Warning: No date column found, creating synthetic temporal sequence")
            self.df['synthetic_date'] = pd.date_range(
                start='2019-01-01', 
                periods=len(self.df), 
                freq='W'
            )
            self.date_col = 'synthetic_date'
    
    def calculate_temporal_metrics(self):
        """Calculate time-varying communication metrics"""
        
        metrics_df = self.df.copy()
        
        # 1. CLARITY METRICS OVER TIME
        try:
            import textstat
        except ImportError:
            print("Warning: textstat not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'textstat'])
            import textstat
        
        clarity_metrics = []
        for idx, row in metrics_df.iterrows():
            text = row[self.text_col] if pd.notna(row[self.text_col]) else ""
            
            if len(text.split()) >= 10:
                clarity_metrics.append({
                    'date': row[self.date_col],
                    'flesch_reading_ease': textstat.flesch_reading_ease(text),
                    'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                    'avg_sentence_length': len(text.split()) / max(len(text.split('.')), 1),
                    'complex_word_ratio': len([w for w in text.split() if len(w) > 7]) / len(text.split()),
                    'word_count': len(text.split())
                })
            else:
                clarity_metrics.append({
                    'date': row[self.date_col],
                    'flesch_reading_ease': np.nan,
                    'flesch_kincaid_grade': np.nan,
                    'avg_sentence_length': np.nan,
                    'complex_word_ratio': np.nan,
                    'word_count': 0
                })
        
        clarity_ts = pd.DataFrame(clarity_metrics).set_index('date')
        
        # 2. COMPREHENSIVENESS METRICS OVER TIME
        economic_terms = {
            'monetary_policy': ['inflasi', 'deflasi', 'moneter', 'kebijakan', 'transmisi'],
            'banking': ['bank', 'sentral', 'kredit', 'likuiditas', 'cadangan'],
            'interest_rates': ['suku', 'bunga', 'rate', 'repo', 'fasbi'],
            'exchange_rates': ['nilai', 'tukar', 'kurs', 'rupiah', 'valuta'],
            'economic_indicators': ['pertumbuhan', 'pdb', 'produksi', 'konsumsi'],
            'financial_markets': ['pasar', 'modal', 'saham', 'obligasi', 'yield']
        }
        
        comprehensiveness_metrics = []
        for idx, row in metrics_df.iterrows():
            text = row[self.text_col] if pd.notna(row[self.text_col]) else ""
            words = text.lower().split()
            
            total_coverage = 0
            category_metrics = {'date': row[self.date_col]}
            
            for category, terms in economic_terms.items():
                covered_terms = sum(1 for term in terms if term in words)
                coverage = covered_terms / len(terms)
                category_metrics[f'{category}_coverage'] = coverage
                total_coverage += covered_terms
            
            category_metrics.update({
                'total_economic_coverage': total_coverage / sum(len(terms) for terms in economic_terms.values()),
                'lexical_diversity': len(set(words)) / len(words) if words else 0,
                'unique_economic_terms': len(set(words) & set().union(*economic_terms.values()))
            })
            
            comprehensiveness_metrics.append(category_metrics)
        
        comprehensiveness_ts = pd.DataFrame(comprehensiveness_metrics).set_index('date')
        
        # 3. SENTIMENT METRICS OVER TIME
        sentiment_metrics = []
        sentiment_cols = [col for col in metrics_df.columns if 'sentimen' in col.lower() and 'majority' in col.lower()]
        
        if sentiment_cols:
            sentiment_col = sentiment_cols[0]
            sentiment_encoding = {'Positif': 1, 'Netral': 0, 'Negatif': -1}
            
            for idx, row in metrics_df.iterrows():
                sentiment = row[sentiment_col] if pd.notna(row[sentiment_col]) else "Netral"
                confidence_cols = [col for col in metrics_df.columns if 'confidence' in col.lower()]
                confidence = row[confidence_cols[0]] if confidence_cols and pd.notna(row[confidence_cols[0]]) else 0.5
                
                sentiment_metrics.append({
                    'date': row[self.date_col],
                    'sentiment_numeric': sentiment_encoding.get(sentiment, 0),
                    'sentiment_confidence': confidence,
                    'sentiment_strength': abs(sentiment_encoding.get(sentiment, 0)) * confidence,
                    'is_positive': 1 if sentiment == 'Positif' else 0,
                    'is_negative': 1 if sentiment == 'Negatif' else 0,
                    'is_neutral': 1 if sentiment == 'Netral' else 0
                })
        
        sentiment_ts = pd.DataFrame(sentiment_metrics).set_index('date') if sentiment_metrics else pd.DataFrame()
        
        return clarity_ts, comprehensiveness_ts, sentiment_ts
    
    def create_aggregated_time_series(self, clarity_ts, comprehensiveness_ts, sentiment_ts, freq='M'):
        """Create aggregated time series at specified frequency"""
        
        # Resample to specified frequency
        clarity_agg = clarity_ts.resample(freq).agg({
            'flesch_reading_ease': 'mean',
            'flesch_kincaid_grade': 'mean',
            'avg_sentence_length': 'mean',
            'complex_word_ratio': 'mean',
            'word_count': ['mean', 'sum', 'count']
        }).round(3)
        
        comprehensiveness_agg = comprehensiveness_ts.resample(freq).agg({
            'total_economic_coverage': 'mean',
            'lexical_diversity': 'mean',
            'unique_economic_terms': 'mean',
            'monetary_policy_coverage': 'mean',
            'banking_coverage': 'mean',
            'interest_rates_coverage': 'mean'
        }).round(3)
        
        if not sentiment_ts.empty:
            sentiment_agg = sentiment_ts.resample(freq).agg({
                'sentiment_numeric': 'mean',
                'sentiment_confidence': 'mean',
                'sentiment_strength': 'mean',
                'is_positive': 'sum',
                'is_negative': 'sum',
                'is_neutral': 'sum'
            }).round(3)
            
            # Calculate sentiment ratios
            total_docs = sentiment_agg[['is_positive', 'is_negative', 'is_neutral']].sum(axis=1)
            sentiment_agg['positive_ratio'] = sentiment_agg['is_positive'] / total_docs
            sentiment_agg['negative_ratio'] = sentiment_agg['is_negative'] / total_docs
            sentiment_agg['neutral_ratio'] = sentiment_agg['is_neutral'] / total_docs
        else:
            sentiment_agg = pd.DataFrame()
        
        return clarity_agg, comprehensiveness_agg, sentiment_agg
    
    def perform_trend_analysis(self, ts_data, metric_name):
        """Perform comprehensive trend analysis"""
        
        if ts_data.empty or ts_data.isna().all():
            return None
        
        # Fill missing values
        ts_clean = ts_data.interpolate().fillna(method='bfill').fillna(method='ffill')
        
        analysis_results = {
            'metric_name': metric_name,
            'data_points': len(ts_clean),
            'date_range': (ts_clean.index.min(), ts_clean.index.max())
        }
        
        # 1. Basic Statistics
        analysis_results['descriptive_stats'] = {
            'mean': ts_clean.mean(),
            'std': ts_clean.std(),
            'min': ts_clean.min(),
            'max': ts_clean.max(),
            'trend_direction': 'increasing' if ts_clean.iloc[-1] > ts_clean.iloc[0] else 'decreasing'
        }
        
        # 2. Stationarity Test
        try:
            adf_result = adfuller(ts_clean.dropna())
            analysis_results['stationarity'] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            }
        except:
            analysis_results['stationarity'] = {'is_stationary': None}
        
        # 3. Seasonal Decomposition (if enough data)
        if len(ts_clean) >= 24:  # Need at least 2 years of monthly data
            try:
                decomposition = seasonal_decompose(ts_clean, model='additive', period=12)
                analysis_results['seasonality'] = {
                    'seasonal_strength': np.std(decomposition.seasonal) / np.std(ts_clean),
                    'trend_strength': np.std(decomposition.trend.dropna()) / np.std(ts_clean),
                    'has_seasonality': np.std(decomposition.seasonal) / np.std(ts_clean) > 0.1
                }
            except:
                analysis_results['seasonality'] = {'has_seasonality': None}
        
        # 4. Change Point Detection
        if len(ts_clean) >= 10:
            # Simple change point detection using gradient
            gradient = np.gradient(ts_clean.values)
            change_points = signal.find_peaks(np.abs(gradient), height=np.std(gradient))[0]
            
            analysis_results['change_points'] = {
                'count': len(change_points),
                'dates': [ts_clean.index[i] for i in change_points] if len(change_points) > 0 else [],
                'magnitude': [gradient[i] for i in change_points] if len(change_points) > 0 else []
            }
        
        # 5. Volatility Analysis
        if len(ts_clean) >= 5:
            rolling_std = ts_clean.rolling(window=min(6, len(ts_clean)//2)).std()
            analysis_results['volatility'] = {
                'mean_volatility': rolling_std.mean(),
                'max_volatility': rolling_std.max(),
                'volatility_trend': 'increasing' if rolling_std.iloc[-1] > rolling_std.iloc[0] else 'decreasing'
            }
        
        return analysis_results
    
    def create_comprehensive_visualizations(self, clarity_agg, comprehensiveness_agg, sentiment_agg, output_dir):
        """Create comprehensive time series visualizations"""
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(4, 3, figsize=(20, 16))
        fig.suptitle('Central Bank Communication: Time Series Analysis', fontsize=16, fontweight='bold')
        
        # 1. CLARITY TRENDS
        if not clarity_agg.empty:
            # Reading Ease over time
            axes[0, 0].plot(clarity_agg.index, clarity_agg[('flesch_reading_ease', 'mean')], 
                           marker='o', linewidth=2, color='blue')
            axes[0, 0].set_title('Clarity: Reading Ease Over Time')
            axes[0, 0].set_ylabel('Flesch Reading Ease')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Easy Threshold')
            axes[0, 0].legend()
            
            # Word Count Trends
            axes[0, 1].plot(clarity_agg.index, clarity_agg[('word_count', 'mean')], 
                           marker='s', linewidth=2, color='purple')
            axes[0, 1].set_title('Communication Length Over Time')
            axes[0, 1].set_ylabel('Average Word Count')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Complex Word Ratio
            axes[0, 2].plot(clarity_agg.index, clarity_agg[('complex_word_ratio', 'mean')], 
                           marker='^', linewidth=2, color='red')
            axes[0, 2].set_title('Complexity: Complex Word Ratio Over Time')
            axes[0, 2].set_ylabel('Complex Word Ratio')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 2. COMPREHENSIVENESS TRENDS
        if not comprehensiveness_agg.empty:
            # Total Economic Coverage
            axes[1, 0].plot(comprehensiveness_agg.index, comprehensiveness_agg[('total_economic_coverage', 'mean')], 
                           marker='o', linewidth=2, color='green')
            axes[1, 0].set_title('Comprehensiveness: Economic Coverage Over Time')
            axes[1, 0].set_ylabel('Economic Term Coverage')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% Threshold')
            axes[1, 0].legend()
            
            # Category Coverage Comparison
            category_cols = ['monetary_policy_coverage', 'banking_coverage', 'interest_rates_coverage']
            for i, col in enumerate(category_cols):
                if (col, 'mean') in comprehensiveness_agg.columns:
                    axes[1, 1].plot(comprehensiveness_agg.index, comprehensiveness_agg[(col, 'mean')], 
                                   marker='o', linewidth=2, label=col.replace('_coverage', '').replace('_', ' ').title())
            axes[1, 1].set_title('Coverage by Economic Category')
            axes[1, 1].set_ylabel('Coverage Ratio')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            # Lexical Diversity
            axes[1, 2].plot(comprehensiveness_agg.index, comprehensiveness_agg[('lexical_diversity', 'mean')], 
                           marker='s', linewidth=2, color='purple')
            axes[1, 2].set_title('Lexical Diversity Over Time')
            axes[1, 2].set_ylabel('Lexical Diversity Ratio')
            axes[1, 2].grid(True, alpha=0.3)
        
        # 3. SENTIMENT TRENDS
        if not sentiment_agg.empty:
            # Sentiment Score Over Time
            axes[2, 0].plot(sentiment_agg.index, sentiment_agg[('sentiment_numeric', 'mean')], 
                           marker='o', linewidth=2, color='blue')
            axes[2, 0].set_title('Sentiment Score Over Time')
            axes[2, 0].set_ylabel('Average Sentiment (-1 to 1)')
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[2, 0].axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Positive Threshold')
            axes[2, 0].axhline(y=-0.2, color='red', linestyle='--', alpha=0.5, label='Negative Threshold')
            axes[2, 0].legend()
            
            # Sentiment Distribution Over Time (Stacked)
            axes[2, 1].stackplot(sentiment_agg.index, 
                               sentiment_agg['positive_ratio'], 
                               sentiment_agg['neutral_ratio'], 
                               sentiment_agg['negative_ratio'],
                               labels=['Positive', 'Neutral', 'Negative'],
                               colors=['green', 'gray', 'red'], alpha=0.7)
            axes[2, 1].set_title('Sentiment Distribution Over Time')
            axes[2, 1].set_ylabel('Proportion')
            axes[2, 1].legend(loc='upper right')
            axes[2, 1].grid(True, alpha=0.3)
            
            # Sentiment Confidence Over Time
            axes[2, 2].plot(sentiment_agg.index, sentiment_agg[('sentiment_confidence', 'mean')], 
                           marker='^', linewidth=2, color='orange')
            axes[2, 2].set_title('Sentiment Confidence Over Time')
            axes[2, 2].set_ylabel('Average Confidence')
            axes[2, 2].grid(True, alpha=0.3)
            axes[2, 2].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High Confidence')
            axes[2, 2].legend()
        
        # 4. INTEGRATED ANALYSIS
        # Communication Effectiveness Index (combine all metrics)
        if not clarity_agg.empty and not comprehensiveness_agg.empty:
            # Normalize metrics to 0-1 scale
            clarity_norm = (clarity_agg[('flesch_reading_ease', 'mean')] - 
                           clarity_agg[('flesch_reading_ease', 'mean')].min()) / \
                          (clarity_agg[('flesch_reading_ease', 'mean')].max() - 
                           clarity_agg[('flesch_reading_ease', 'mean')].min())
            
            comprehensiveness_norm = comprehensiveness_agg[('total_economic_coverage', 'mean')]
            
            if not sentiment_agg.empty:
                sentiment_norm = (sentiment_agg[('sentiment_numeric', 'mean')] + 1) / 2  # Scale to 0-1
                effectiveness_index = (clarity_norm + comprehensiveness_norm + sentiment_norm) / 3
            else:
                effectiveness_index = (clarity_norm + comprehensiveness_norm) / 2
            
            axes[3, 0].plot(effectiveness_index.index, effectiveness_index.values, 
                           marker='o', linewidth=3, color='darkblue')
            axes[3, 0].fill_between(effectiveness_index.index, effectiveness_index.values, 
                                   alpha=0.3, color='lightblue')
            axes[3, 0].set_title('Communication Effectiveness Index')
            axes[3, 0].set_ylabel('Effectiveness Score (0-1)')
            axes[3, 0].grid(True, alpha=0.3)
            axes[3, 0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target Level')
            axes[3, 0].legend()
        
        # Communication Volume Over Time
        if not clarity_agg.empty:
            axes[3, 1].bar(clarity_agg.index, clarity_agg[('word_count', 'count')], 
                          color='lightcoral', alpha=0.7)
            axes[3, 1].set_title('Communication Volume (Number of Documents)')
            axes[3, 1].set_ylabel('Document Count')
            axes[3, 1].grid(True, alpha=0.3)
        
        # Volatility Analysis (Rolling Standard Deviation)
        if not clarity_agg.empty and len(clarity_agg) >= 6:
            rolling_clarity = clarity_agg[('flesch_reading_ease', 'mean')].rolling(window=6).std()
            rolling_comprehensiveness = comprehensiveness_agg[('total_economic_coverage', 'mean')].rolling(window=6).std()
            
            axes[3, 2].plot(rolling_clarity.index, rolling_clarity.values, 
                           marker='o', linewidth=2, label='Clarity Volatility', color='blue')
            axes[3, 2].plot(rolling_comprehensiveness.index, rolling_comprehensiveness.values, 
                           marker='s', linewidth=2, label='Comprehensiveness Volatility', color='green')
            axes[3, 2].set_title('Communication Consistency (6-Month Rolling Volatility)')
            axes[3, 2].set_ylabel('Standard Deviation')
            axes[3, 2].grid(True, alpha=0.3)
            axes[3, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comprehensive_time_series_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_time_series_insights(self, clarity_analysis, comprehensiveness_analysis, 
                                     sentiment_analysis, output_dir):
        """Generate comprehensive insights report"""
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        insights_report = f"""
COMPREHENSIVE TIME SERIES ANALYSIS REPORT
=========================================
Central Bank Communication Effectiveness Over Time

EXECUTIVE SUMMARY
================

This report provides temporal analysis of central bank communication across three key dimensions:
1. CLARITY - How easily the communication can be understood
2. COMPREHENSIVENESS - How thoroughly economic topics are covered  
3. SENTIMENT - The emotional tone and confidence of communication

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: Full dataset temporal range
"""
        
        # Save insights report
        with open(f'{output_dir}/time_series_insights_report.txt', 'w', encoding='utf-8') as f:
            f.write(insights_report)
        
        return insights_report

def main():
    """Main function with proper argument parsing"""
    parser = argparse.ArgumentParser(description='Comprehensive Time Series Analysis for Central Bank Communication')
    parser.add_argument('--merged-file', required=True, help='Path to merged annotations file')
    parser.add_argument('--processed-file', help='Path to processed file (optional)')
    parser.add_argument('--sentiment-file', help='Path to sentiment results file (optional)')
    parser.add_argument('--text-col', default='Teks_Paragraf', help='Text column name')
    parser.add_argument('--date-col', default='Tanggal', help='Date column name')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--freq', default='M', choices=['D', 'W', 'M', 'Q', 'Y'], 
                       help='Aggregation frequency (Daily, Weekly, Monthly, Quarterly, Yearly)')
    
    args = parser.parse_args()
    
    try:
        # Validate input file exists
        if not os.path.exists(args.merged_file):
            raise FileNotFoundError(f"Input file not found: {args.merged_file}")
        
        print(f"Loading data from: {args.merged_file}")
        
        # Load data
        df = pd.read_excel(args.merged_file)
        print(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        # Initialize analyzer with proper parameters
        analyzer = CommunicationTimeSeriesAnalyzer(
            df, 
            date_col=args.date_col, 
            text_col=args.text_col
        )
        
        # Calculate temporal metrics
        print("Calculating temporal metrics...")
        clarity_ts, comprehensiveness_ts, sentiment_ts = analyzer.calculate_temporal_metrics()
        
        # Create aggregated time series
        print(f"Creating aggregated time series with frequency: {args.freq}...")
        clarity_agg, comprehensiveness_agg, sentiment_agg = analyzer.create_aggregated_time_series(
            clarity_ts, comprehensiveness_ts, sentiment_ts, freq=args.freq
        )
        
        # Perform trend analysis
        print("Performing trend analysis...")
        clarity_analysis = analyzer.perform_trend_analysis(
            clarity_agg[('flesch_reading_ease', 'mean')] if not clarity_agg.empty else pd.Series(), 
            'Clarity'
        )
        
        comprehensiveness_analysis = analyzer.perform_trend_analysis(
            comprehensiveness_agg[('total_economic_coverage', 'mean')] if not comprehensiveness_agg.empty else pd.Series(),
            'Comprehensiveness'
        )
        
        sentiment_analysis = analyzer.perform_trend_analysis(
            sentiment_agg[('sentiment_numeric', 'mean')] if not sentiment_agg.empty else pd.Series(),
            'Sentiment'
        )
        
        # Create visualizations
        print("Creating comprehensive visualizations...")
        analyzer.create_comprehensive_visualizations(
            clarity_agg, comprehensiveness_agg, sentiment_agg, args.output_dir
        )
        
        # Generate insights
        print("Generating insights report...")
        insights = analyzer.generate_time_series_insights(
            clarity_analysis, comprehensiveness_analysis, sentiment_analysis, args.output_dir
        )
        
        print("\nAnalysis Summary:")
        print("=" * 60)
        print(f"Input file: {args.merged_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Analysis completed successfully!")
        
        return analyzer, insights
        
    except FileNotFoundError as e:
        print(f"ERROR: {str(e)}")
        return None, None
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    """Main execution block"""
    try:
        print("="*80)
        print("COMPREHENSIVE TIME SERIES ANALYSIS FOR CENTRAL BANK COMMUNICATION")
        print("="*80)
        
        analyzer, insights = main()
        
        if analyzer is not None:
            print("\n🎉 TIME SERIES ANALYSIS COMPLETED SUCCESSFULLY!")
        else:
            print("\n❌ TIME SERIES ANALYSIS FAILED!")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user (Ctrl+C)")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        exit(1)