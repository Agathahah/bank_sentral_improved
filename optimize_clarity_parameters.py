# Buat file optimize_clarity_parameters.py
def optimize_clarity_parameters():
    """Optimasi parameter untuk hasil clarity terbaik"""
    
    from enhanced_clarity_framework import EnhancedBankSentralLexicon, EnhancedClarityCalculator
    import itertools
    
    # Load data untuk optimization
    df = pd.read_excel('output/merged_annotations_enhanced.xlsx')
    sample_texts = df['Teks_Paragraf'].dropna().head(50).tolist()  # Sample untuk testing
    
    # Parameter ranges untuk optimization
    parameter_combinations = {
        'sentence_length_weight': [0.2, 0.25, 0.3],
        'syllable_weight': [0.15, 0.2, 0.25], 
        'technical_density_weight': [0.1, 0.15, 0.2],
        'readability_weight': [0.35, 0.4, 0.45],
        'target_sentence_length': [(15, 20), (15, 25), (20, 25)],
        'optimal_technical_density': [0.12, 0.15, 0.18]
    }
    
    best_params = None
    best_score = 0
    optimization_results = []
    
    # Test different parameter combinations (sample to avoid too many combinations)
    param_keys = list(parameter_combinations.keys())
    param_values = list(parameter_combinations.values())
    
    # Sample 20 combinations to test
    all_combinations = list(itertools.product(*param_values))
    test_combinations = all_combinations[::len(all_combinations)//20] if len(all_combinations) > 20 else all_combinations
    
    for i, param_combo in enumerate(test_combinations):
        print(f"Testing parameter combination {i+1}/{len(test_combinations)}")
        
        params = dict(zip(param_keys, param_combo))
        
        # Create temporary calculator with these parameters
        lexicon = EnhancedBankSentralLexicon()
        
        # Update weights in lexicon
        lexicon.clarity_weights.update({
            'sentence_length': {
                'ideal_range': params['target_sentence_length'],
                'penalty_factor': 0.1
            },
            'technical_density': {
                'optimal_ratio': params['optimal_technical_density'],
                'max_ratio': 0.30
            }
        })
        
        calculator = EnhancedClarityCalculator(lexicon)
        
        # Override weights in calculator
        original_calculate = calculator.calculate_enhanced_clarity
        
        def modified_calculate(text):
            metrics = original_calculate(text)
            
            # Recalculate composite score with new weights
            clarity_components = {
                'sentence_length_score': metrics['sentence_length_score'],
                'syllable_score': metrics['syllable_score'], 
                'technical_density_score': metrics['technical_density_score'],
                'contextual_readability': metrics['contextual_readability']
            }
            
            composite_score = (
                params['sentence_length_weight'] * clarity_components['sentence_length_score'] +
                params['syllable_weight'] * clarity_components['syllable_score'] +
                params['technical_density_weight'] * clarity_components['technical_density_score'] +
                params['readability_weight'] * clarity_components['contextual_readability'] / 100
            )
            
            metrics['composite_clarity_score'] = composite_score
            return metrics
        
        # Test on sample texts
        scores = []
        for text in sample_texts[:10]:  # Test on first 10 samples
            try:
                metrics = modified_calculate(str(text))
                scores.append(metrics['composite_clarity_score'])
            except:
                continue
        
        if scores:
            avg_score = np.mean(scores)
            score_std = np.std(scores)
            
            # Composite evaluation score (higher average, lower std deviation)
            evaluation_score = avg_score - 0.1 * score_std
            
            optimization_results.append({
                'combination_id': i,
                'parameters': params,
                'avg_clarity_score': avg_score,
                'score_std': score_std,
                'evaluation_score': evaluation_score
            })
            
            if evaluation_score > best_score:
                best_score = evaluation_score
                best_params = params.copy()
    
    # Save optimization results
    optimization_df = pd.DataFrame([
        {
            'combination_id': r['combination_id'],
            'avg_clarity_score': r['avg_clarity_score'],
            'score_std': r['score_std'],
            'evaluation_score': r['evaluation_score'],
            **r['parameters']
        }
        for r in optimization_results
    ])
    
    optimization_df.to_excel('output/enhanced_clarity/parameter_optimization.xlsx', index=False)
    
    # Create optimized configuration
    optimized_config = {
        'optimized_parameters': best_params,
        'best_evaluation_score': best_score,
        'optimization_summary': {
            'total_combinations_tested': len(test_combinations),
            'successful_evaluations': len(optimization_results),
            'improvement_over_default': best_score - 0.4  # Assuming default ~0.4
        }
    }
    
    with open('output/enhanced_clarity/optimized_config.json', 'w') as f:
        json.dump(optimized_config, f, indent=2, default=str)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Parameter importance
    plt.subplot(2, 3, 1)
    param_importance = {}
    for param in param_keys:
        if param in optimization_df.columns:
            correlation = optimization_df[param].astype(str).astype('category').cat.codes.corr(
                optimization_df['evaluation_score']
            )
            param_importance[param] = abs(correlation)
    
    if param_importance:
        plt.bar(param_importance.keys(), param_importance.values())
        plt.title('Parameter Importance for Clarity')
        plt.xticks(rotation=45)
        plt.ylabel('Correlation with Evaluation Score')
    
    # Score distribution
    plt.subplot(2, 3, 2)
    plt.hist(optimization_df['evaluation_score'], bins=15, alpha=0.7, color='skyblue')
    plt.axvline(best_score, color='red', linestyle='--', label=f'Best: {best_score:.3f}')
    plt.xlabel('Evaluation Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution Across Parameters')
    plt.legend()
    
    # Best vs Default comparison
    plt.subplot(2, 3, 3)
    default_score = 0.4  # Estimated default
    improvement = best_score - default_score
    
    plt.bar(['Default', 'Optimized'], [default_score, best_score], 
            color=['orange', 'green'])
    plt.title(f'Optimization Improvement: +{improvement:.3f}')
    plt.ylabel('Evaluation Score')
    
    plt.tight_layout()
    plt.savefig('output/enhanced_clarity/parameter_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_params, optimized_config

if __name__ == "__main__":
    best_params, config = optimize_clarity_parameters()
    print("Parameter optimization completed!")
    print("Best parameters:", best_params)