# Buat file compare_clarity_methods.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_clarity_methods():
    # Load original results
    try:
        original_df = pd.read_excel('output/preprocessing_analysis/preprocessing_comparison.xlsx')
        original_scores = original_df['Avg_Reading_Ease'].mean()
    except:
        original_scores = 20  # Baseline

def compare_clarity_methods():
    # Load original results
    try:
        original_df = pd.read_excel('output/preprocessing_analysis/preprocessing_comparison.xlsx')
        original_scores = original_df['Avg_Reading_Ease'].mean()
    except:
        original_scores = 20  # Baseline dari hasil sebelumnya

    # Load enhanced results
    enhanced_df = pd.read_excel('output/enhanced_clarity_analysis.xlsx')
    
    # Compare scores
    comparison_data = []
    
    # Original method scores
    comparison_data.append({
        'Method': 'Original Flesch-Kincaid',
        'Average_Score': original_scores,
        'Method_Type': 'Traditional',
        'Context_Aware': 'No'
    })
    
    # Enhanced method scores
    enhanced_variants = ['enhanced_clean', 'enhanced_context_filtered']
    
    for variant in enhanced_variants:
        score_col = f'{variant}_composite_clarity_score'
        if score_col in enhanced_df.columns:
            avg_score = enhanced_df[score_col].mean() * 100  # Convert to 0-100 scale
            comparison_data.append({
                'Method': variant.replace('_', ' ').title(),
                'Average_Score': avg_score,
                'Method_Type': 'Enhanced',
                'Context_Aware': 'Yes'
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Bar plot comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(comparison_df['Method'], comparison_df['Average_Score'], 
                   color=['red' if x == 'Traditional' else 'green' for x in comparison_df['Method_Type']])
    plt.title('Clarity Score Comparison: Traditional vs Enhanced')
    plt.ylabel('Average Clarity Score')
    plt.xticks(rotation=45)
    plt.axhline(y=50, color='orange', linestyle='--', label='Acceptable Threshold')
    plt.axhline(y=70, color='green', linestyle='--', label='Good Threshold')
    plt.legend()
    
    # Improvement visualization
    plt.subplot(2, 2, 2)
    if len(comparison_df[comparison_df['Method_Type'] == 'Enhanced']) > 0:
        enhanced_scores = comparison_df[comparison_df['Method_Type'] == 'Enhanced']['Average_Score']
        improvement = enhanced_scores.max() - original_scores
        
        plt.bar(['Original', 'Enhanced (Best)'], 
                [original_scores, enhanced_scores.max()],
                color=['red', 'green'])
        plt.title(f'Improvement: +{improvement:.1f} points')
        plt.ylabel('Clarity Score')
        
        # Add improvement text
        plt.text(0.5, max(original_scores, enhanced_scores.max()) * 0.5, 
                f'+{improvement:.1f}\npoints', 
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Distribution comparison (if we have individual scores)
    plt.subplot(2, 2, 3)
    if 'enhanced_clean_composite_clarity_score' in enhanced_df.columns:
        enhanced_scores_dist = enhanced_df['enhanced_clean_composite_clarity_score'] * 100
        enhanced_scores_dist = enhanced_scores_dist.dropna()
        
        if len(enhanced_scores_dist) > 0:
            plt.hist(enhanced_scores_dist, bins=20, alpha=0.7, color='green', 
                    label='Enhanced Method', density=True)
            plt.axvline(enhanced_scores_dist.mean(), color='green', linestyle='-', 
                       linewidth=2, label=f'Enhanced Mean: {enhanced_scores_dist.mean():.1f}')
            plt.axvline(original_scores, color='red', linestyle='--', 
                       linewidth=2, label=f'Original Mean: {original_scores:.1f}')
            plt.xlabel('Clarity Score')
            plt.ylabel('Density')
            plt.title('Score Distribution Comparison')
            plt.legend()
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    summary_text = "CLARITY IMPROVEMENT SUMMARY\n"
    summary_text += "=" * 30 + "\n\n"
    summary_text += f"Original Average Score: {original_scores:.1f}\n"
    
    if len(comparison_df[comparison_df['Method_Type'] == 'Enhanced']) > 0:
        best_enhanced = comparison_df[comparison_df['Method_Type'] == 'Enhanced']['Average_Score'].max()
        improvement_pct = ((best_enhanced - original_scores) / original_scores) * 100
        
        summary_text += f"Best Enhanced Score: {best_enhanced:.1f}\n"
        summary_text += f"Absolute Improvement: +{best_enhanced - original_scores:.1f}\n"
        summary_text += f"Relative Improvement: +{improvement_pct:.1f}%\n\n"
        
        if best_enhanced > 50:
            summary_text += "✅ TARGET ACHIEVED:\n"
            summary_text += "  - Clarity above acceptable threshold\n"
        else:
            summary_text += "⚠️  STILL NEEDS IMPROVEMENT:\n"
            summary_text += "  - Below acceptable threshold (50)\n"
        
        if improvement_pct > 50:
            summary_text += "  - Significant improvement achieved\n"
        elif improvement_pct > 20:
            summary_text += "  - Moderate improvement achieved\n"
        else:
            summary_text += "  - Minor improvement achieved\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('output/enhanced_clarity/clarity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    comparison_df.to_excel('output/enhanced_clarity/clarity_method_comparison.xlsx', index=False)
    
    return comparison_df

if __name__ == "__main__":
    comparison_results = compare_clarity_methods()
    print("Clarity comparison completed!")