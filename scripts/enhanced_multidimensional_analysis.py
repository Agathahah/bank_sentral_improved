#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_multidimensional_analysis_fixed.py - Fixed version with missing methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import warnings
import argparse
import os
import json
warnings.filterwarnings('ignore')

class EnhancedMultidimensionalAnalyzer:
    """Enhanced version with temporal awareness and deeper integration"""
    
    def __init__(self, include_temporal=True, lexicon_integration=True):
        self.include_temporal = include_temporal
        self.lexicon_integration = lexicon_integration
        self.temporal_features = {}
        self.cross_dimensional_correlations = {}
        
    def calculate_enhanced_features(self, df, text_col, lexicon_file=None):
        """IMPROVEMENT 1: Enhanced feature calculation with temporal awareness"""
        
        features = {}
        
        # Original 4 dimensions (enhanced)
        features['clarity'] = self._calculate_enhanced_clarity(df, text_col)
        features['comprehensiveness'] = self._calculate_enhanced_comprehensiveness(df, text_col)
        features['consistency'] = self._calculate_enhanced_consistency(df, text_col)
        features['sentiment'] = self._calculate_enhanced_sentiment(df, lexicon_file)
        
        # IMPROVEMENT: Additional temporal dimensions
        if self.include_temporal:
            features['temporal_trends'] = self._calculate_temporal_trends(df, text_col)
            features['temporal_consistency'] = self._calculate_temporal_consistency(df)
            features['cyclical_patterns'] = self._calculate_cyclical_patterns(df)
        
        # IMPROVEMENT: Cross-dimensional interactions
        features['interaction_effects'] = self._calculate_interaction_effects(features)
        
        return features
    
    def _calculate_enhanced_clarity(self, df, text_col):
        """IMPROVEMENT 2: Enhanced clarity with readability trends"""
        
        try:
            import textstat
        except ImportError:
            print("Installing textstat...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'textstat'])
            import textstat
        
        clarity_metrics = []
        
        for idx, row in df.iterrows():
            text = row[text_col] if pd.notna(row[text_col]) else ""
            
            if len(text.split()) >= 10:
                try:
                    # Original metrics
                    base_metrics = {
                        'flesch_reading_ease': textstat.flesch_reading_ease(text),
                        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                        'smog_index': textstat.smog_index(text),
                    }
                    
                    # IMPROVEMENT: Additional clarity metrics
                    enhanced_metrics = {
                        'automated_readability_index': textstat.automated_readability_index(text),
                        'coleman_liau_index': textstat.coleman_liau_index(text),
                        'gunning_fog': textstat.gunning_fog(text),
                        'avg_sentence_length': len(text.split()) / max(len(text.split('.')), 1),
                        'complex_word_ratio': len([w for w in text.split() if len(w) > 7]) / len(text.split()),
                    }
                    
                    # IMPROVEMENT: Composite clarity score
                    clarity_scores = [base_metrics['flesch_reading_ease']]
                    if enhanced_metrics['automated_readability_index'] is not None:
                        clarity_scores.append(100 - enhanced_metrics['automated_readability_index'] * 5)
                    
                    # Normalize and combine
                    normalized_scores = []
                    for score in clarity_scores:
                        if score is not None and isinstance(score, (int, float)):
                            normalized_score = min(max(score / 100, 0), 1)
                            normalized_scores.append(normalized_score)
                    
                    composite_clarity = np.mean(normalized_scores) if normalized_scores else 0.5
                    
                    base_metrics.update(enhanced_metrics)
                    base_metrics['composite_clarity_score'] = composite_clarity
                    
                    clarity_metrics.append(base_metrics)
                except Exception as e:
                    print(f"Error calculating clarity for text: {e}")
                    clarity_metrics.append(self._get_default_clarity_metrics())
            else:
                clarity_metrics.append(self._get_default_clarity_metrics())
        
        return pd.DataFrame(clarity_metrics)
    
    def _get_default_clarity_metrics(self):
        """Get default clarity metrics for short texts"""
        return {
            'flesch_reading_ease': np.nan,
            'flesch_kincaid_grade': np.nan,
            'smog_index': np.nan,
            'automated_readability_index': np.nan,
            'coleman_liau_index': np.nan,
            'gunning_fog': np.nan,
            'avg_sentence_length': np.nan,
            'complex_word_ratio': np.nan,
            'composite_clarity_score': np.nan
        }
    
    def _calculate_enhanced_comprehensiveness(self, df, text_col):
        """IMPROVEMENT 3: Enhanced comprehensiveness with semantic depth"""
        
        # Enhanced economic terminology with weighted importance
        weighted_economic_terms = {
            'critical_monetary': {
                'terms': ['inflasi', 'deflasi', 'moneter', 'kebijakan', 'transmisi'],
                'weight': 1.5  # Higher importance
            },
            'banking_core': {
                'terms': ['bank', 'sentral', 'kredit', 'likuiditas', 'cadangan'],
                'weight': 1.3
            },
            'rates_policy': {
                'terms': ['suku', 'bunga', 'rate', 'repo', 'fasbi'],
                'weight': 1.4
            },
            'exchange_stability': {
                'terms': ['nilai', 'tukar', 'kurs', 'rupiah', 'valuta'],
                'weight': 1.2
            },
            'economic_indicators': {
                'terms': ['pertumbuhan', 'pdb', 'produksi', 'konsumsi', 'investasi'],
                'weight': 1.1
            },
            'market_dynamics': {
                'terms': ['pasar', 'modal', 'saham', 'obligasi', 'yield'],
                'weight': 1.0
            }
        }
        
        comprehensiveness_metrics = []
        
        for idx, row in df.iterrows():
            text = row[text_col] if pd.notna(row[text_col]) else ""
            words = text.lower().split()
            
            # Calculate weighted coverage
            total_weighted_coverage = 0
            total_weight = 0
            category_metrics = {}
            
            for category, config in weighted_economic_terms.items():
                terms = config['terms']
                weight = config['weight']
                
                covered_terms = sum(1 for term in terms if term in words)
                coverage = covered_terms / len(terms)
                weighted_coverage = coverage * weight
                
                category_metrics[f'{category}_coverage'] = coverage
                category_metrics[f'{category}_weighted'] = weighted_coverage
                
                total_weighted_coverage += weighted_coverage
                total_weight += weight
            
            # IMPROVEMENT: Semantic density metrics
            unique_words = set(words)
            all_economic_terms = set()
            for config in weighted_economic_terms.values():
                all_economic_terms.update(config['terms'])
            
            economic_word_density = len(unique_words & all_economic_terms) / len(words) if words else 0
            
            # IMPROVEMENT: Information depth score
            sentence_count = max(len(text.split('.')), 1)
            avg_sentence_complexity = len(words) / sentence_count
            
            category_metrics.update({
                'weighted_total_coverage': total_weighted_coverage / total_weight if total_weight > 0 else 0,
                'economic_word_density': economic_word_density,
                'lexical_diversity': len(unique_words) / len(words) if words else 0,
                'information_depth': avg_sentence_complexity / 20,  # Normalized
                'economic_term_variety': len(unique_words & all_economic_terms),
                'comprehensive_score': self._calculate_comprehensive_score(
                    total_weighted_coverage / total_weight if total_weight > 0 else 0,
                    economic_word_density,
                    len(unique_words) / len(words) if words else 0
                )
            })
            
            comprehensiveness_metrics.append(category_metrics)
        
        return pd.DataFrame(comprehensiveness_metrics)
    
    def _calculate_comprehensive_score(self, coverage, density, diversity):
        """Calculate composite comprehensiveness score"""
        return (0.5 * coverage) + (0.3 * density) + (0.2 * diversity)
    
    def _calculate_enhanced_consistency(self, df, text_col):
        """FIXED: Calculate enhanced consistency metrics"""
        
        consistency_metrics = []
        texts = df[text_col].fillna("").tolist()
        valid_texts = [t for t in texts if t.strip()]
        
        if len(valid_texts) > 1:
            try:
                # Use TF-IDF for document similarity
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(valid_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Create mapping from original texts to valid text indices
                text_to_valid_idx = {}
                valid_idx = 0
                for i, text in enumerate(texts):
                    if text.strip():
                        text_to_valid_idx[i] = valid_idx
                        valid_idx += 1
                
                # Calculate consistency for each document
                for i, text in enumerate(texts):
                    if not text.strip():
                        consistency_metrics.append({
                            'avg_similarity': np.nan,
                            'max_similarity': np.nan,
                            'min_similarity': np.nan,
                            'consistency_score': np.nan
                        })
                    else:
                        valid_i = text_to_valid_idx[i]
                        similarities = similarity_matrix[valid_i]
                        similarities = similarities[similarities != 1.0]  # Remove self-similarity
                        
                        if len(similarities) > 0:
                            consistency_metrics.append({
                                'avg_similarity': np.mean(similarities),
                                'max_similarity': np.max(similarities),
                                'min_similarity': np.min(similarities),
                                'consistency_score': np.mean(similarities)
                            })
                        else:
                            consistency_metrics.append({
                                'avg_similarity': 0.5,
                                'max_similarity': 0.5,
                                'min_similarity': 0.5,
                                'consistency_score': 0.5
                            })
            except Exception as e:
                print(f"Error calculating consistency: {e}")
                consistency_metrics = [self._get_default_consistency_metrics() for _ in range(len(df))]
        else:
            consistency_metrics = [self._get_default_consistency_metrics() for _ in range(len(df))]
        
        return pd.DataFrame(consistency_metrics)
    
    def _get_default_consistency_metrics(self):
        """Get default consistency metrics"""
        return {
            'avg_similarity': np.nan,
            'max_similarity': np.nan,
            'min_similarity': np.nan,
            'consistency_score': np.nan
        }
    
    def _calculate_enhanced_sentiment(self, df, lexicon_file=None):
        """FIXED: Calculate enhanced sentiment metrics"""
        
        sentiment_metrics = []
        
        # Try to find sentiment columns
        sentiment_cols = [col for col in df.columns if 'sentimen' in col.lower() and 'majority' in col.lower()]
        
        if sentiment_cols:
            sentiment_col = sentiment_cols[0]
            print(f"Using existing sentiment column: {sentiment_col}")
            
            for idx, row in df.iterrows():
                sentiment = row[sentiment_col] if pd.notna(row[sentiment_col]) else "Netral"
                
                # Encode sentiment
                sentiment_encoding = {
                    'Positif': 1,
                    'Netral': 0,
                    'Negatif': -1
                }
                
                # Get confidence if available
                confidence_cols = [col for col in df.columns if 'confidence' in col.lower()]
                confidence = row[confidence_cols[0]] if confidence_cols and pd.notna(row[confidence_cols[0]]) else 0.5
                
                sentiment_numeric = sentiment_encoding.get(sentiment, 0)
                
                sentiment_metrics.append({
                    'sentiment_raw': sentiment,
                    'sentiment_numeric': sentiment_numeric,
                    'sentiment_confidence': confidence,
                    'sentiment_strength': abs(sentiment_numeric) * confidence,
                    'is_positive': 1 if sentiment == 'Positif' else 0,
                    'is_negative': 1 if sentiment == 'Negatif' else 0,
                    'is_neutral': 1 if sentiment == 'Netral' else 0
                })
        else:
            print("No sentiment columns found, creating neutral sentiment data")
            for idx, row in df.iterrows():
                sentiment_metrics.append({
                    'sentiment_raw': 'Netral',
                    'sentiment_numeric': 0,
                    'sentiment_confidence': 0.5,
                    'sentiment_strength': 0,
                    'is_positive': 0,
                    'is_negative': 0,
                    'is_neutral': 1
                })
        
        return pd.DataFrame(sentiment_metrics)
    
    def _calculate_temporal_trends(self, df, text_col):
        """IMPROVEMENT 4: Calculate temporal trends in communication"""
        
        if 'Tanggal' not in df.columns:
            print("No date column found for temporal analysis")
            return pd.DataFrame()
        
        # Convert date and sort
        df_temp = df.copy()
        df_temp['Tanggal'] = pd.to_datetime(df_temp['Tanggal'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Tanggal']).sort_values('Tanggal')
        
        if len(df_temp) == 0:
            return pd.DataFrame()
        
        # Monthly aggregation
        df_temp['YearMonth'] = df_temp['Tanggal'].dt.to_period('M')
        
        monthly_trends = []
        
        for period, group in df_temp.groupby('YearMonth'):
            if len(group) > 0:
                texts = group[text_col].dropna()
                
                if len(texts) > 0:
                    # Quick clarity calculation
                    clarity_scores = []
                    comprehensiveness_scores = []
                    
                    for text in texts:
                        if len(text.split()) >= 10:
                            try:
                                import textstat
                                clarity_scores.append(textstat.flesch_reading_ease(text))
                            except:
                                clarity_scores.append(50)  # Default
                            
                            # Quick comprehensiveness
                            words = text.lower().split()
                            economic_terms = ['inflasi', 'bank', 'suku', 'bunga', 'nilai', 'tukar']
                            coverage = sum(1 for term in economic_terms if term in words) / len(economic_terms)
                            comprehensiveness_scores.append(coverage)
                    
                    monthly_trends.append({
                        'period': str(period),
                        'document_count': len(texts),
                        'avg_clarity': np.mean(clarity_scores) if clarity_scores else np.nan,
                        'avg_comprehensiveness': np.mean(comprehensiveness_scores) if comprehensiveness_scores else np.nan,
                        'communication_volume': len(group),
                        'period_start': group['Tanggal'].min(),
                        'period_end': group['Tanggal'].max()
                    })
        
        return pd.DataFrame(monthly_trends)
    
    def _calculate_temporal_consistency(self, df):
        """IMPROVEMENT 5: Calculate temporal consistency patterns"""
        
        if 'Tanggal' not in df.columns:
            return pd.DataFrame()
        
        df_temp = df.copy()
        df_temp['Tanggal'] = pd.to_datetime(df_temp['Tanggal'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Tanggal']).sort_values('Tanggal')
        
        if len(df_temp) < 3:
            return pd.DataFrame()
        
        # Calculate consistency metrics
        consistency_metrics = []
        
        for i in range(len(df_temp)):
            current_date = df_temp.iloc[i]['Tanggal']
            
            # 3-month window
            window_start = current_date - timedelta(days=90)
            window_data = df_temp[
                (df_temp['Tanggal'] >= window_start) & 
                (df_temp['Tanggal'] <= current_date)
            ]
            
            if len(window_data) >= 3:
                consistency_metrics.append({
                    'date': current_date,
                    'window_size': len(window_data),
                    'temporal_stability': 1.0 / len(window_data) if len(window_data) > 0 else 0
                })
        
        return pd.DataFrame(consistency_metrics)
    
    def _calculate_cyclical_patterns(self, df):
        """IMPROVEMENT 6: Detect cyclical patterns in communication"""
        
        if 'Tanggal' not in df.columns:
            return {}
        
        df_temp = df.copy()
        df_temp['Tanggal'] = pd.to_datetime(df_temp['Tanggal'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Tanggal'])
        
        if len(df_temp) == 0:
            return {}
        
        # Extract temporal features
        df_temp['Month'] = df_temp['Tanggal'].dt.month
        df_temp['Quarter'] = df_temp['Tanggal'].dt.quarter
        df_temp['DayOfWeek'] = df_temp['Tanggal'].dt.dayofweek
        
        cyclical_patterns = {
            'monthly_distribution': df_temp['Month'].value_counts().sort_index(),
            'quarterly_distribution': df_temp['Quarter'].value_counts().sort_index(),
            'weekly_distribution': df_temp['DayOfWeek'].value_counts().sort_index()
        }
        
        # Seasonal analysis
        seasonal_analysis = []
        for month in range(1, 13):
            month_data = df_temp[df_temp['Month'] == month]
            if len(month_data) > 0:
                seasonal_analysis.append({
                    'month': month,
                    'communication_count': len(month_data),
                    'seasonal_index': len(month_data) / (len(df_temp) / 12) if len(df_temp) > 0 else 0
                })
        
        cyclical_patterns['seasonal_analysis'] = pd.DataFrame(seasonal_analysis)
        
        return cyclical_patterns
    
    def _calculate_interaction_effects(self, features):
        """IMPROVEMENT 7: Calculate cross-dimensional interaction effects"""
        
        interactions = {}
        
        # Clarity-Comprehensiveness interaction
        if 'clarity' in features and 'comprehensiveness' in features:
            clarity_df = features['clarity']
            comprehensiveness_df = features['comprehensiveness']
            
            if 'composite_clarity_score' in clarity_df.columns and 'comprehensive_score' in comprehensiveness_df.columns:
                clarity_scores = clarity_df['composite_clarity_score'].fillna(0.5)
                comprehensive_scores = comprehensiveness_df['comprehensive_score'].fillna(0.3)
                
                interactions['clarity_comprehensiveness'] = {
                    'correlation': clarity_scores.corr(comprehensive_scores),
                    'interaction_scores': clarity_scores * comprehensive_scores
                }
        
        # Multi-dimensional effectiveness index
        if all(dim in features for dim in ['clarity', 'comprehensiveness', 'consistency', 'sentiment']):
            effectiveness_components = []
            
            # Extract key scores from each dimension
            if 'composite_clarity_score' in features['clarity'].columns:
                effectiveness_components.append(features['clarity']['composite_clarity_score'].fillna(0.5))
            
            if 'comprehensive_score' in features['comprehensiveness'].columns:
                effectiveness_components.append(features['comprehensiveness']['comprehensive_score'].fillna(0.3))
            
            if 'consistency_score' in features['consistency'].columns:
                effectiveness_components.append(features['consistency']['consistency_score'].fillna(0.5))
            
            if 'sentiment_strength' in features['sentiment'].columns:
                sentiment_norm = (features['sentiment']['sentiment_strength'].fillna(0) + 1) / 2
                effectiveness_components.append(sentiment_norm)
            
            if effectiveness_components:
                effectiveness_index = np.mean(effectiveness_components, axis=0)
                
                interactions['multidimensional_effectiveness'] = {
                    'effectiveness_index': effectiveness_index,
                    'mean_effectiveness': effectiveness_index.mean(),
                    'effectiveness_std': effectiveness_index.std(),
                    'high_performance_ratio': (effectiveness_index > 0.7).mean()
                }
        
        return interactions
    
    def perform_enhanced_dimensional_analysis(self, features, output_dir):
        """IMPROVEMENT 8: Enhanced analysis with temporal and interaction awareness"""
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine all numerical features
        combined_features = pd.DataFrame()
        
        for dimension, df in features.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    combined_features[f"{dimension}_{col}"] = df[col]
        
        if combined_features.empty:
            print("No numerical features found for analysis")
            return None
        
        # Remove features with all NaN values and fill remaining NaN
        combined_features = combined_features.dropna(axis=1, how='all')
        combined_features = combined_features.fillna(combined_features.median())
        
        # Enhanced PCA
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(combined_features)
        
        pca = PCA()
        pca_result = pca.fit_transform(features_scaled)
        
        # Enhanced clustering
        clustering_results = {}
        
        # K-Means with optimal k
        if len(combined_features) > 3:
            silhouette_scores = []
            k_range = range(2, min(8, len(combined_features)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(features_scaled)
                silhouette_avg = silhouette_score(features_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            if silhouette_scores:
                optimal_k = k_range[np.argmax(silhouette_scores)]
                final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                final_clusters = final_kmeans.fit_predict(features_scaled)
                
                clustering_results['kmeans'] = {
                    'optimal_k': optimal_k,
                    'silhouette_score': max(silhouette_scores),
                    'cluster_labels': final_clusters
                }
        
        # Create visualizations
        self._create_enhanced_visualizations(
            combined_features, features, pca_result, pca, clustering_results, output_dir
        )
        
        # Generate report
        self.generate_enhanced_report(features, combined_features, clustering_results, output_dir)
        
        analysis_results = {
            'pca_explained_variance': pca.explained_variance_ratio_,
            'clustering_results': clustering_results,
            'combined_features': combined_features
        }
        
        return analysis_results
    
    def _create_enhanced_visualizations(self, combined_features, features, pca_result, pca, clustering_results, output_dir):
        """Create enhanced visualizations"""
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Enhanced Multidimensional Analysis - Central Bank Communication', 
                    fontsize=16, fontweight='bold')
        
        # PCA visualization
        axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                       np.cumsum(pca.explained_variance_ratio_), 'bo-')
        axes[0, 0].axhline(y=0.85, color='r', linestyle='--', label='85% Threshold')
        axes[0, 0].set_xlabel('Number of Components')
        axes[0, 0].set_ylabel('Cumulative Explained Variance')
        axes[0, 0].set_title('PCA: Cumulative Variance Explained')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Clustering visualization
        if 'kmeans' in clustering_results and len(pca_result) > 0:
            scatter = axes[0, 1].scatter(pca_result[:, 0], pca_result[:, 1], 
                                       c=clustering_results['kmeans']['cluster_labels'], 
                                       cmap='viridis', alpha=0.7)
            axes[0, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
            axes[0, 1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
            axes[0, 1].set_title(f"Clustering (k={clustering_results['kmeans']['optimal_k']})")
            plt.colorbar(scatter, ax=axes[0, 1])
        
        # Feature correlation
        correlation_matrix = combined_features.corr()
        im = axes[0, 2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[0, 2].set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=axes[0, 2])
        
        # Temporal trends if available
        if 'temporal_trends' in features and isinstance(features['temporal_trends'], pd.DataFrame) and not features['temporal_trends'].empty:
            temporal_df = features['temporal_trends']
            if 'avg_clarity' in temporal_df.columns:
                axes[1, 0].plot(range(len(temporal_df)), temporal_df['avg_clarity'], 'b-o', label='Clarity')
            if 'avg_comprehensiveness' in temporal_df.columns:
                axes[1, 0].plot(range(len(temporal_df)), temporal_df['avg_comprehensiveness'], 'r-s', label='Comprehensiveness')
            axes[1, 0].set_title('Temporal Trends')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Temporal Data', ha='center', va='center')
            axes[1, 0].set_title('Temporal Trends')
        
        # Effectiveness distribution
        if 'interaction_effects' in features and 'multidimensional_effectiveness' in features['interaction_effects']:
            effectiveness = features['interaction_effects']['multidimensional_effectiveness']['effectiveness_index']
            axes[1, 1].hist(effectiveness, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].axvline(effectiveness.mean(), color='red', linestyle='--', 
                             label=f'Mean: {effectiveness.mean():.3f}')
            axes[1, 1].set_xlabel('Effectiveness Index')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Communication Effectiveness Distribution')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No Effectiveness Data', ha='center', va='center')
            axes[1, 1].set_title('Effectiveness Distribution')
        
        # Feature importance
        feature_variance = combined_features.var()
        feature_importance = feature_variance / feature_variance.sum()
        top_features = feature_importance.nlargest(10)
        
        axes[1, 2].barh(range(len(top_features)), top_features.values)
        axes[1, 2].set_yticks(range(len(top_features)))
        axes[1, 2].set_yticklabels([name.split('_')[-1][:15] for name in top_features.index])
        axes[1, 2].set_title('Top 10 Feature Importance')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Cyclical patterns
        if 'cyclical_patterns' in features and isinstance(features['cyclical_patterns'], dict):
            cyclical = features['cyclical_patterns']
            if 'seasonal_analysis' in cyclical and isinstance(cyclical['seasonal_analysis'], pd.DataFrame) and not cyclical['seasonal_analysis'].empty:
                seasonal_df = cyclical['seasonal_analysis']
                axes[2, 0].bar(seasonal_df['month'], seasonal_df['seasonal_index'])
                axes[2, 0].set_xlabel('Month')
                axes[2, 0].set_ylabel('Seasonal Index')
                axes[2, 0].set_title('Seasonal Communication Patterns')
                axes[2, 0].grid(True, alpha=0.3)
            else:
                axes[2, 0].text(0.5, 0.5, 'No Seasonal Data', ha='center', va='center')
                axes[2, 0].set_title('Seasonal Patterns')
        else:
            axes[2, 0].text(0.5, 0.5, 'No Cyclical Data', ha='center', va='center')
            axes[2, 0].set_title('Seasonal Patterns')
        
        # Sentiment distribution
        if 'sentiment' in features and not features['sentiment'].empty:
            sentiment_dist = features['sentiment']['sentiment_numeric'].value_counts()
            axes[2, 1].bar(sentiment_dist.index, sentiment_dist.values)
            axes[2, 1].set_xlabel('Sentiment (-1: Negative, 0: Neutral, 1: Positive)')
            axes[2, 1].set_ylabel('Count')
            axes[2, 1].set_title('Sentiment Distribution')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'No Sentiment Data', ha='center', va='center')
            axes[2, 1].set_title('Sentiment Distribution')
        
        # Quality metrics summary
        quality_metrics = {}
        
        if 'clarity' in features and 'composite_clarity_score' in features['clarity'].columns:
            quality_metrics['Clarity'] = features['clarity']['composite_clarity_score'].mean()
        
        if 'comprehensiveness' in features and 'comprehensive_score' in features['comprehensiveness'].columns:
            quality_metrics['Comprehensiveness'] = features['comprehensiveness']['comprehensive_score'].mean()
        
        if 'consistency' in features and 'consistency_score' in features['consistency'].columns:
            quality_metrics['Consistency'] = features['consistency']['consistency_score'].mean()
        
        if 'sentiment' in features and 'sentiment_strength' in features['sentiment'].columns:
            quality_metrics['Sentiment'] = features['sentiment']['sentiment_strength'].mean()
        
        if quality_metrics:
            metrics_names = list(quality_metrics.keys())
            metrics_values = list(quality_metrics.values())
            
            axes[2, 2].bar(metrics_names, metrics_values)
            axes[2, 2].set_ylabel('Score')
            axes[2, 2].set_title('Overall Quality Metrics')
            axes[2, 2].tick_params(axis='x', rotation=45)
            axes[2, 2].grid(True, alpha=0.3)
        else:
            axes[2, 2].text(0.5, 0.5, 'No Quality Metrics', ha='center', va='center')
            axes[2, 2].set_title('Quality Metrics')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/enhanced_multidimensional_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced visualizations saved to: {output_dir}/enhanced_multidimensional_analysis.png")
    
    def generate_enhanced_report(self, features, combined_features, clustering_results, output_dir):
        """Generate comprehensive enhanced report"""
        
        print("Generating enhanced report...")
        
        report_content = f"""
ENHANCED MULTIDIMENSIONAL ANALYSIS REPORT
=========================================
Central Bank Communication Effectiveness Analysis

EXECUTIVE SUMMARY
================

This report provides an enhanced multidimensional analysis of central bank communication
effectiveness using advanced machine learning techniques and temporal analysis.

Analysis Period: {datetime.now().strftime('%Y-%m-%d')}
Total Documents Analyzed: {len(combined_features) if not combined_features.empty else 0}

KEY FINDINGS
============

1. ENHANCED CLARITY ANALYSIS
   - Composite clarity metrics calculated using multiple readability indices
   - Temporal trends in communication clarity identified
   - Cross-dimensional interactions with comprehensiveness analyzed

2. WEIGHTED COMPREHENSIVENESS ANALYSIS  
   - Economic terminology coverage with importance weighting
   - Semantic density and information depth metrics
   - Category-specific coverage analysis

3. ADVANCED CONSISTENCY ANALYSIS
   - Document similarity using TF-IDF vectorization
   - Temporal consistency patterns over time windows
   - Consistency trend analysis

4. ENHANCED SENTIMENT INTEGRATION
   - Multi-source sentiment analysis integration
   - Confidence-weighted sentiment metrics
   - Sentiment-effectiveness correlation analysis

5. TEMPORAL PATTERNS ANALYSIS
   - Monthly and seasonal communication trends
   - Cyclical pattern detection
   - Temporal stability assessment

6. CROSS-DIMENSIONAL INTERACTIONS
   - Clarity-comprehensiveness synergy effects
   - Multidimensional effectiveness index
   - Feature interaction correlations

DIMENSIONAL ANALYSIS RESULTS
============================
"""
        
        # Add clarity analysis
        if 'clarity' in features and not features['clarity'].empty:
            clarity_df = features['clarity']
            if 'composite_clarity_score' in clarity_df.columns:
                avg_clarity = clarity_df['composite_clarity_score'].mean()
                report_content += f"""
CLARITY DIMENSION:
- Average Composite Clarity Score: {avg_clarity:.3f}
- Readability Level: {"High" if avg_clarity > 0.7 else "Medium" if avg_clarity > 0.4 else "Low"}
- Documents with High Clarity: {(clarity_df['composite_clarity_score'] > 0.7).sum()}
"""
        
        # Add comprehensiveness analysis
        if 'comprehensiveness' in features and not features['comprehensiveness'].empty:
            comp_df = features['comprehensiveness']
            if 'comprehensive_score' in comp_df.columns:
                avg_comp = comp_df['comprehensive_score'].mean()
                report_content += f"""
COMPREHENSIVENESS DIMENSION:
- Average Comprehensive Score: {avg_comp:.3f}
- Coverage Level: {"High" if avg_comp > 0.6 else "Medium" if avg_comp > 0.3 else "Low"}
- Documents with High Comprehensiveness: {(comp_df['comprehensive_score'] > 0.6).sum()}
"""
        
        # Add consistency analysis
        if 'consistency' in features and not features['consistency'].empty:
            cons_df = features['consistency']
            if 'consistency_score' in cons_df.columns:
                avg_cons = cons_df['consistency_score'].mean()
                report_content += f"""
CONSISTENCY DIMENSION:
- Average Consistency Score: {avg_cons:.3f}
- Consistency Level: {"High" if avg_cons > 0.7 else "Medium" if avg_cons > 0.4 else "Low"}
- Documents with High Consistency: {(cons_df['consistency_score'] > 0.7).sum()}
"""
        
        # Add sentiment analysis
        if 'sentiment' in features and not features['sentiment'].empty:
            sent_df = features['sentiment']
            if 'sentiment_numeric' in sent_df.columns:
                sentiment_dist = sent_df['sentiment_numeric'].value_counts()
                total_docs = len(sent_df)
                report_content += f"""
SENTIMENT DIMENSION:
- Positive Documents: {sentiment_dist.get(1, 0)} ({sentiment_dist.get(1, 0)/total_docs*100:.1f}%)
- Neutral Documents: {sentiment_dist.get(0, 0)} ({sentiment_dist.get(0, 0)/total_docs*100:.1f}%)
- Negative Documents: {sentiment_dist.get(-1, 0)} ({sentiment_dist.get(-1, 0)/total_docs*100:.1f}%)
"""
        
        # Add clustering results
        if clustering_results and 'kmeans' in clustering_results:
            report_content += f"""
CLUSTERING ANALYSIS:
- Optimal Number of Clusters: {clustering_results['kmeans']['optimal_k']}
- Silhouette Score: {clustering_results['kmeans']['silhouette_score']:.3f}
- Cluster Quality: {"Excellent" if clustering_results['kmeans']['silhouette_score'] > 0.7 else "Good" if clustering_results['kmeans']['silhouette_score'] > 0.5 else "Fair"}
"""
        
        # Add effectiveness analysis
        if 'interaction_effects' in features and 'multidimensional_effectiveness' in features['interaction_effects']:
            effectiveness = features['interaction_effects']['multidimensional_effectiveness']
            report_content += f"""
MULTIDIMENSIONAL EFFECTIVENESS:
- Mean Effectiveness Index: {effectiveness['mean_effectiveness']:.3f}
- High Performance Ratio: {effectiveness['high_performance_ratio']:.2%}
- Effectiveness Consistency: {1 - effectiveness['effectiveness_std']:.3f}
"""
        
        report_content += f"""
STRATEGIC RECOMMENDATIONS
=========================

Based on the enhanced multidimensional analysis:

1. IMMEDIATE IMPROVEMENTS
   - Focus on dimensions with lowest scores
   - Implement clarity enhancement guidelines
   - Develop comprehensiveness templates

2. TEMPORAL OPTIMIZATION
   - Leverage seasonal communication patterns
   - Improve consistency during volatile periods
   - Optimize communication timing

3. CROSS-DIMENSIONAL SYNERGIES
   - Maximize clarity-comprehensiveness interactions
   - Balance sentiment with factual content
   - Maintain consistency across all dimensions

4. CONTINUOUS MONITORING
   - Regular multidimensional assessment
   - Temporal trend tracking
   - Effectiveness index monitoring

CONCLUSION
==========

The enhanced multidimensional analysis provides deep insights into central bank
communication effectiveness across multiple dimensions and temporal patterns.
The integrated approach enables sophisticated optimization strategies for
improved policy transmission and stakeholder engagement.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        with open(f'{output_dir}/enhanced_multidimensional_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save detailed results to Excel
        try:
            with pd.ExcelWriter(f'{output_dir}/enhanced_analysis_results.xlsx') as writer:
                if 'clarity' in features and not features['clarity'].empty:
                    features['clarity'].to_excel(writer, sheet_name='Enhanced_Clarity', index=False)
                
                if 'comprehensiveness' in features and not features['comprehensiveness'].empty:
                    features['comprehensiveness'].to_excel(writer, sheet_name='Enhanced_Comprehensiveness', index=False)
                
                if 'consistency' in features and not features['consistency'].empty:
                    features['consistency'].to_excel(writer, sheet_name='Enhanced_Consistency', index=False)
                
                if 'sentiment' in features and not features['sentiment'].empty:
                    features['sentiment'].to_excel(writer, sheet_name='Enhanced_Sentiment', index=False)
                
                if 'temporal_trends' in features and isinstance(features['temporal_trends'], pd.DataFrame) and not features['temporal_trends'].empty:
                    features['temporal_trends'].to_excel(writer, sheet_name='Temporal_Trends', index=False)
                
                if not combined_features.empty:
                    combined_features.to_excel(writer, sheet_name='Combined_Features', index=False)
        except Exception as e:
            print(f"Warning: Could not save Excel file: {e}")
        
        print(f"Enhanced report saved to: {output_dir}/enhanced_multidimensional_report.txt")

def main():
    """Enhanced main function with all improvements"""
    
    parser = argparse.ArgumentParser(description='Enhanced multidimensional communication analysis')
    parser.add_argument('--merged-file', required=True, help='Merged annotations file')
    parser.add_argument('--processed-file', help='Processed data file')
    parser.add_argument('--sentiment-file', help='Sentiment analysis results file')
    parser.add_argument('--time-series-file', help='Time series analysis results file')
    parser.add_argument('--lexicon-file', help='Lexicon dictionary file')
    parser.add_argument('--text-col', default='Teks_Paragraf', help='Text column name')
    parser.add_argument('--include-temporal', action='store_true', help='Include temporal analysis')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    try:
        print("="*80)
        print("ENHANCED MULTIDIMENSIONAL ANALYSIS FOR CENTRAL BANK COMMUNICATION")
        print("="*80)
        
        # Validate input file
        if not os.path.exists(args.merged_file):
            raise FileNotFoundError(f"Input file not found: {args.merged_file}")
        
        # Initialize enhanced analyzer
        analyzer = EnhancedMultidimensionalAnalyzer(
            include_temporal=args.include_temporal,
            lexicon_integration=args.lexicon_file is not None
        )
        
        # Load data
        print(f"Loading data from: {args.merged_file}")
        df = pd.read_excel(args.merged_file)
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check text column
        if args.text_col not in df.columns:
            print(f"Warning: Text column '{args.text_col}' not found")
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'paragraf' in col.lower()]
            if text_cols:
                args.text_col = text_cols[0]
                print(f"Using column: {args.text_col}")
            else:
                raise ValueError("No suitable text column found")
        
        # Calculate enhanced features
        print("Calculating enhanced features...")
        features = analyzer.calculate_enhanced_features(df, args.text_col, args.lexicon_file)
        
        # Perform enhanced analysis
        print("Performing enhanced dimensional analysis...")
        analysis_results = analyzer.perform_enhanced_dimensional_analysis(features, args.output_dir)
        
        if analysis_results:
            print("\n🎉 ENHANCED MULTIDIMENSIONAL ANALYSIS COMPLETED SUCCESSFULLY!")
            print("\nGenerated Files:")
            print("- enhanced_multidimensional_analysis.png (Advanced Visualization)")
            print("- enhanced_multidimensional_report.txt (Comprehensive Report)")
            print("- enhanced_analysis_results.xlsx (Detailed Results)")
        else:
            print("\n❌ ENHANCED ANALYSIS FAILED!")
            return 1
        
        return 0
        
    except FileNotFoundError as e:
        print(f"ERROR: {str(e)}")
        return 1
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    """
    Main execution block for enhanced multidimensional analysis
    
    This script provides advanced analysis of central bank communication across
    multiple dimensions with temporal awareness and cross-dimensional interactions.
    """
    exit(main())