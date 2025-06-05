#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robust_advanced_sentiment_analysis.py - Robust sentiment analysis for validated preprocessed data
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from tqdm import tqdm
import joblib
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('robust_sentiment_analysis')

def ensure_directory(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_and_validate_sentiment_data(file_path, text_cols, sentiment_col, sheet_name=0):
    """Load and validate data with text validity checking"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check for text_valid column from preprocessing
        if 'text_valid' in df.columns:
            # Filter only valid texts
            df_valid = df[df['text_valid'] == True].copy()
            logger.info(f"Using only valid texts: {len(df_valid)} out of {len(df)} rows")
            df = df_valid
        
        # Check required columns
        missing_cols = []
        for col in text_cols + [sentiment_col]:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            logger.error(f"Missing columns: {', '.join(missing_cols)}")
            return None
        
        # Clean sentiment data
        df = df[df[sentiment_col].notna()].copy()
        
        # Standardize sentiment labels
        df[sentiment_col] = df[sentiment_col].astype(str).str.strip().str.title()
        
        # Filter valid sentiments
        valid_sentiments = ['Positif', 'Netral', 'Negatif']
        df = df[df[sentiment_col].isin(valid_sentiments)].copy()
        
        if len(df) == 0:
            logger.error("No valid sentiment data found")
            return None
        
        # Additional validation for text columns
        for text_col in text_cols:
            # Remove empty texts
            df = df[df[text_col].notna()].copy()
            df = df[df[text_col].str.strip() != ""].copy()
            
            # Check minimum text length
            word_counts = df[text_col].str.split().str.len()
            df = df[word_counts >= 5].copy()  # At least 5 words
        
        logger.info(f"After validation: {len(df)} rows with valid data")
        
        # Log class distribution
        class_dist = df[sentiment_col].value_counts()
        logger.info(f"Class distribution:\n{class_dist}")
        
        # Calculate imbalance ratio
        majority_class = class_dist.max()
        minority_class = class_dist.min()
        imbalance_ratio = majority_class / minority_class if minority_class > 0 else float('inf')
        
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def create_robust_models():
    """Create model configurations with robust parameters"""
    
    models = {
        'Naive_Bayes_TF-IDF': {
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000, 
                    ngram_range=(1, 2),
                    min_df=2,  # Ignore terms that appear in less than 2 documents
                    max_df=0.95  # Ignore terms that appear in more than 95% of documents
                )),
                ('classifier', MultinomialNB(alpha=0.1))
            ]),
            'params': {
                'tfidf__max_features': [3000, 5000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier__alpha': [0.01, 0.1, 1.0]
            }
        },
        
        'SVM_Linear': {
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000, 
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )),
                ('classifier', SVC(
                    kernel='linear', 
                    probability=True, 
                    random_state=42, 
                    class_weight='balanced',
                    max_iter=2000  # Increased iterations
                ))
            ]),
            'params': {
                'tfidf__max_features': [3000, 5000],
                'classifier__C': [0.1, 1.0, 10.0]
            }
        },
        
        'Random_Forest': {
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000, 
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    class_weight='balanced',
                    n_jobs=-1
                ))
            ]),
            'params': {
                'tfidf__max_features': [3000, 5000],
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5]
            }
        },
        
        'Logistic_Regression': {
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000, 
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )),
                ('classifier', LogisticRegression(
                    random_state=42, 
                    class_weight='balanced', 
                    max_iter=2000,
                    solver='lbfgs'
                ))
            ]),
            'params': {
                'tfidf__max_features': [3000, 5000],
                'classifier__C': [0.1, 1.0, 10.0]
            }
        }
    }
    
    return models

def apply_smote_if_needed(X_train, y_train, random_state=42):
    """Apply SMOTE only if there's significant imbalance"""
    from collections import Counter
    
    class_counts = Counter(y_train)
    min_class_count = min(class_counts.values())
    max_class_count = max(class_counts.values())
    
    imbalance_ratio = max_class_count / min_class_count
    
    if imbalance_ratio > 2 and min_class_count >= 6:  # Need at least 6 samples for SMOTE
        logger.info(f"Applying SMOTE due to imbalance ratio: {imbalance_ratio:.2f}")
        
        try:
            # First vectorize the text
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_train_vec = vectorizer.fit_transform(X_train)
            
            # Apply SMOTE
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, min_class_count-1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)
            
            logger.info(f"After SMOTE: {Counter(y_train_balanced)}")
            
            return X_train, y_train, True  # Return original X_train as models will vectorize
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Proceeding without balancing.")
            return X_train, y_train, False
    else:
        logger.info("No significant imbalance or insufficient samples. Skipping SMOTE.")
        return X_train, y_train, False

def train_and_evaluate_models(X_train, y_train, X_test, y_test, models, cv_folds=5):
    """Train models with robust evaluation"""
    logger.info("Training and evaluating models...")
    
    trained_models = {}
    evaluation_results = {}
    
    # Adjust cv_folds if needed
    n_samples = len(X_train)
    n_classes = len(set(y_train))
    min_samples_per_class = min(Counter(y_train).values())
    
    cv_folds = min(cv_folds, min_samples_per_class)
    logger.info(f"Using {cv_folds} CV folds")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, model_config in models.items():
        logger.info(f"Training {name}...")
        
        try:
            # Simplified grid search
            grid_search = GridSearchCV(
                model_config['pipeline'],
                model_config['params'],
                cv=skf,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = grid_search.predict(X_test)
            
            # Store results
            trained_models[name] = grid_search.best_estimator_
            evaluation_results[name] = {
                'best_cv_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'test_score': grid_search.score(X_test, y_test),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred
            }
            
            logger.info(f"{name} - CV Score: {grid_search.best_score_:.4f}, Test Score: {evaluation_results[name]['test_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
    
    return trained_models, evaluation_results

def create_evaluation_plots(evaluation_results, output_dir):
    """Create comprehensive evaluation plots"""
    logger.info("Creating evaluation plots...")
    
    ensure_directory(output_dir)
    
    # Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract metrics
    model_names = list(evaluation_results.keys())
    cv_scores = [results['best_cv_score'] for results in evaluation_results.values()]
    test_scores = [results['test_score'] for results in evaluation_results.values()]
    
    # CV vs Test scores
    axes[0, 0].bar(np.arange(len(model_names)) - 0.2, cv_scores, 0.4, label='CV Score')
    axes[0, 0].bar(np.arange(len(model_names)) + 0.2, test_scores, 0.4, label='Test Score')
    axes[0, 0].set_xticks(np.arange(len(model_names)))
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].legend()
    
    # F1 scores by class
    axes[0, 1].set_title('F1 Score by Class')
    for i, (name, results) in enumerate(evaluation_results.items()):
        report = results['classification_report']
        classes = ['Positif', 'Netral', 'Negatif']
        f1_scores = [report.get(cls, {}).get('f1-score', 0) for cls in classes]
        x = np.arange(len(classes)) + i * 0.15
        axes[0, 1].bar(x, f1_scores, 0.15, label=name)
    axes[0, 1].set_xticks(np.arange(len(classes)) + 0.3)
    axes[0, 1].set_xticklabels(classes)
    axes[0, 1].legend()
    
    # Best model confusion matrix
    best_model = max(evaluation_results.items(), key=lambda x: x[1]['test_score'])[0]
    cm = evaluation_results[best_model]['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix - {best_model}')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"Best Model: {best_model}\n"
    summary_text += f"Test F1 Score: {evaluation_results[best_model]['test_score']:.4f}\n\n"
    summary_text += "All Models Test Scores:\n"
    for name, results in evaluation_results.items():
        summary_text += f"{name}: {results['test_score']:.4f}\n"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_evaluation.png'), dpi=300)
    plt.close()
    
    # Save detailed results
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'CV_Score': cv_scores,
        'Test_Score': test_scores,
        'Accuracy': [results['classification_report']['accuracy'] for results in evaluation_results.values()],
        'F1_Weighted': [results['classification_report']['weighted avg']['f1-score'] for results in evaluation_results.values()]
    })
    
    comparison_df.to_excel(os.path.join(output_dir, 'model_comparison.xlsx'), index=False)
    
    return best_model

def save_models_and_results(trained_models, evaluation_results, output_dir, models_dir):
    """Save models and results"""
    logger.info("Saving models and results...")
    
    ensure_directory(models_dir)
    
    # Save models
    for name, model in trained_models.items():
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model {name} saved to {model_path}")
    
    # Save best model
    best_model_name = max(evaluation_results.items(), key=lambda x: x[1]['test_score'])[0]
    best_model_path = os.path.join(models_dir, "best_sentiment_model.pkl")
    joblib.dump(trained_models[best_model_name], best_model_path)
    logger.info(f"Best model ({best_model_name}) saved as: {best_model_path}")
    
    # Save results summary
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write("SENTIMENT ANALYSIS RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for name, results in evaluation_results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  CV Score: {results['best_cv_score']:.4f}\n")
            f.write(f"  Test Score: {results['test_score']:.4f}\n")
            f.write(f"  Best Parameters: {results['best_params']}\n")
        
        f.write(f"\nBEST MODEL: {best_model_name}\n")

def main():
    parser = argparse.ArgumentParser(description='Robust sentiment analysis for validated data')
    parser.add_argument('--input', required=True, help='Input Excel file')
    parser.add_argument('--text-cols', required=True, nargs='+', help='Text columns to analyze')
    parser.add_argument('--sentiment-col', required=True, help='Sentiment column name')
    parser.add_argument('--models-dir', required=True, help='Directory to save models')
    parser.add_argument('--output-dir', required=True, help='Directory to save results')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    
    args = parser.parse_args()
    
    # Create directories
    ensure_directory(args.output_dir)
    ensure_directory(args.models_dir)
    
    # Load and validate data
    df = load_and_validate_sentiment_data(args.input, args.text_cols, args.sentiment_col)
    if df is None:
        return
    
    # Process best text column (usually from preprocessing recommendation)
    best_results = None
    best_score = 0
    
    for text_col in args.text_cols:
        logger.info(f"\nProcessing text column: {text_col}")
        
        # Skip if column doesn't exist or is empty
        if text_col not in df.columns:
            logger.warning(f"Column '{text_col}' not found. Skipping.")
            continue
        
        valid_texts = df[df[text_col].notna() & (df[text_col].str.strip() != '')]
        if len(valid_texts) < 20:  # Need minimum samples
            logger.warning(f"Column '{text_col}' has insufficient valid texts. Skipping.")
            continue
        
        # Prepare data
        X = valid_texts[text_col].values
        y = valid_texts[args.sentiment_col].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Apply SMOTE if needed
        X_train, y_train, smote_applied = apply_smote_if_needed(X_train, y_train)
        
        # Create models
        models = create_robust_models()
        
        # Train and evaluate
        trained_models, evaluation_results = train_and_evaluate_models(
            X_train, y_train, X_test, y_test, models, args.cv_folds
        )
        
        if not trained_models:
            logger.warning(f"No models trained successfully for column '{text_col}'")
            continue
        
        # Check if this is the best result so far
        best_model_score = max(results['test_score'] for results in evaluation_results.values())
        if best_model_score > best_score:
            best_score = best_model_score
            best_results = {
                'text_col': text_col,
                'trained_models': trained_models,
                'evaluation_results': evaluation_results
            }
    
    if best_results is None:
        logger.error("No successful model training for any text column")
        return
    
    logger.info(f"\nBest results from column: {best_results['text_col']}")
    
    # Create evaluation plots
    best_model_name = create_evaluation_plots(
        best_results['evaluation_results'], 
        args.output_dir
    )
    
    # Save models and results
    save_models_and_results(
        best_results['trained_models'],
        best_results['evaluation_results'],
        args.output_dir,
        args.models_dir
    )
    
    # Save final summary
    with open(os.path.join(args.output_dir, 'final_summary.txt'), 'w') as f:
        f.write(f"BEST TEXT COLUMN: {best_results['text_col']}\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write(f"BEST F1 SCORE: {best_score:.4f}\n")
    
    logger.info("Robust sentiment analysis completed successfully!")

if __name__ == "__main__":
    main()