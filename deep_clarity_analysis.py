# Buat file deep_clarity_analysis.py
def analyze_clarity_factors():
    """Analisis mendalam faktor-faktor yang mempengaruhi clarity"""
    
    enhanced_df = pd.read_excel('output/enhanced_clarity_analysis.xlsx')
    
    # Analyze correlation between factors
    clarity_factors = [
        'enhanced_clean_sentence_length_score',
        'enhanced_clean_syllable_score', 
        'enhanced_clean_technical_density_score',
        'enhanced_clean_contextual_readability',
        'enhanced_clean_composite_clarity_score'
    ]
    
    available_factors = [col for col in clarity_factors if col in enhanced_df.columns]
    
    if len(available_factors) > 1:
        correlation_matrix = enhanced_df[available_factors].corr()
        
        plt.figure(figsize=(15, 12))
        
        # Correlation heatmap
        plt.subplot(2, 3, 1)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Clarity Factors Correlation')
        
        # Individual factor distributions
        for i, factor in enumerate(available_factors[:4], 2):
            plt.subplot(2, 3, i)
            data = enhanced_df[factor].dropna()
            if len(data) > 0:
                plt.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title(factor.replace('enhanced_clean_', '').replace('_', ' ').title())
                plt.xlabel('Score')
                plt.ylabel('Frequency')
                
                # Add mean line
                plt.axvline(data.mean(), color='red', linestyle='--', 
                           label=f'Mean: {data.mean():.3f}')
                plt.legend()
        
        # Factor importance analysis
        plt.subplot(2, 3, 6)
        if 'enhanced_clean_composite_clarity_score' in enhanced_df.columns:
            target = enhanced_df['enhanced_clean_composite_clarity_score']
            other_factors = [col for col in available_factors if 'composite' not in col]
            
            importances = []
            factor_names = []
            
            for factor in other_factors:
                if factor in enhanced_df.columns:
                    correlation = enhanced_df[factor].corr(target)
                    importances.append(abs(correlation))
                    factor_names.append(factor.replace('enhanced_clean_', '').replace('_score', ''))
            
            if importances:
                plt.barh(factor_names, importances)
                plt.xlabel('Absolute Correlation with Composite Score')
                plt.title('Factor Importance for Overall Clarity')
                plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/enhanced_clarity/clarity_factors_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate insights
    insights = generate_clarity_insights(enhanced_df)
    
    with open('output/enhanced_clarity/clarity_insights.txt', 'w', encoding='utf-8') as f:
        f.write(insights)
    
    return insights

def generate_clarity_insights(df):
    """Generate insights dari analisis clarity"""
    
    insights = """
DEEP CLARITY ANALYSIS INSIGHTS
===============================

"""
    
    # Analyze each factor
    factors_analysis = {
        'sentence_length': 'enhanced_clean_avg_sentence_length',
        'technical_density': 'enhanced_clean_technical_density', 
        'syllable_complexity': 'enhanced_clean_syllable_complexity',
        'composite_score': 'enhanced_clean_composite_clarity_score'
    }
    
    for factor_name, col_name in factors_analysis.items():
        if col_name in df.columns:
            data = df[col_name].dropna()
            if len(data) > 0:
                insights += f"{factor_name.replace('_', ' ').title()}:\n"
                insights += f"  - Mean: {data.mean():.3f}\n"
                insights += f"  - Std Dev: {data.std():.3f}\n"
                insights += f"  - Min: {data.min():.3f}\n"
                insights += f"  - Max: {data.max():.3f}\n"
                
                # Provide interpretation
                if factor_name == 'sentence_length':
                    if data.mean() > 25:
                        insights += "  - ISSUE: Sentences too long (target: 15-25 words)\n"
                    elif data.mean() < 15:
                        insights += "  - NOTE: Sentences quite short (could be choppy)\n"
                    else:
                        insights += "  - GOOD: Sentence length in optimal range\n"
                
                elif factor_name == 'technical_density':
                    if data.mean() > 0.3:
                        insights += "  - ISSUE: Too many technical terms (>30%)\n"
                    elif data.mean() < 0.1:
                        insights += "  - NOTE: Very low technical density (may lack substance)\n"
                    else:
                        insights += "  - GOOD: Technical density in acceptable range\n"
                
                elif factor_name == 'composite_score':
                    score_100 = data.mean() * 100
                    if score_100 > 70:
                        insights += "  - EXCELLENT: High clarity achieved\n"
                    elif score_100 > 50:
                        insights += "  - GOOD: Acceptable clarity level\n"
                    elif score_100 > 30:
                        insights += "  - NEEDS IMPROVEMENT: Below target\n"
                    else:
                        insights += "  - CRITICAL: Very low clarity\n"
                
                insights += "\n"
    
    # Recommendations
    insights += """
STRATEGIC RECOMMENDATIONS
=========================

Based on the analysis, here are specific recommendations:

1. SENTENCE STRUCTURE OPTIMIZATION:
   - Target 15-25 words per sentence
   - Use active voice where possible
   - Break complex sentences into simpler ones
   - Use transitional phrases for flow

2. TECHNICAL TERMINOLOGY BALANCE:
   - Maintain 10-20% technical term density
   - Define technical terms when first introduced
   - Use analogies for complex concepts
   - Provide context for specialized vocabulary

3. READABILITY ENHANCEMENT:
   - Reduce average syllables per word
   - Use simpler alternatives where possible
   - Maintain economic precision while improving accessibility
   - Consider audience-specific versions

4. CONTEXTUAL IMPROVEMENTS:
   - Preserve important economic terms
   - Maintain formal communication style
   - Ensure logical flow and coherence
   - Use consistent terminology throughout

5. IMPLEMENTATION PRIORITIES:
   - Focus on highest-impact factors first
   - Test changes with sample audiences
   - Monitor clarity scores regularly
   - Adjust based on feedback and results
"""
    
    return insights

if __name__ == "__main__":
    insights = analyze_clarity_factors()
    print("Deep clarity analysis completed!")