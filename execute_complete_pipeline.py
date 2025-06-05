# Buat file execute_complete_pipeline.py
def execute_complete_enhanced_clarity_pipeline():
    """Eksekusi lengkap pipeline enhanced clarity"""
    
    print("="*80)
    print("EXECUTING COMPLETE ENHANCED CLARITY PIPELINE")
    print("="*80)
    
    execution_log = []
    
    try:
        # Step 1: Preparation
        log_step("1. Environment Preparation", preparation_phase, execution_log)
        
        # Step 2: Enhanced Merge
        log_step("2. Enhanced Annotation Merge", enhanced_merge_phase, execution_log)
        
        # Step 3: Enhanced Preprocessing  
        log_step("3. Enhanced Preprocessing", enhanced_preprocessing_phase, execution_log)
        
        # Step 4: Clarity Analysis
        log_step("4. Enhanced Clarity Analysis", clarity_analysis_phase, execution_log)
        
        # Step 5: Optimization
        log_step("5. Parameter Optimization", optimization_phase, execution_log)
        
        # Step 6: Validation
        log_step("6. Comprehensive Validation", validation_phase, execution_log)
        
        # Step 7: Final Integration
        log_step("7. Final Integration", integration_phase, execution_log)
        
        # Step 8: Reporting
        log_step("8. Comprehensive Reporting", reporting_phase, execution_log)
        
        # Step 9: Testing
        log_step("9. System Testing", testing_phase, execution_log)
        
        print("\n🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n💥 PIPELINE EXECUTION FAILED: {e}")
        execution_log.append({
            'step': 'PIPELINE_FAILURE',
            'status': 'FAILED',
            'error': str(e),
            'timestamp': datetime.now()
        })
    
    finally:
        # Always generate execution summary
        generate_execution_summary(execution_log)
    
    return execution_log

def log_step(step_name, step_function, execution_log):
    """Log dan eksekusi step dengan error handling"""
    
    print(f"\n{step_name}")
    print("-" * 60)
    
    start_time = datetime.now()
    
    try:
        result = step_function()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        execution_log.append({
            'step': step_name,
            'status': 'SUCCESS',
            'duration': duration,
            'result': result,
            'timestamp': end_time
        })
        
        print(f"✅ {step_name} completed successfully in {duration:.1f}s")
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        execution_log.append({
            'step': step_name,
            'status': 'FAILED',
            'duration': duration,
            'error': str(e),
            'timestamp': end_time
        })
        
        print(f"❌ {step_name} failed after {duration:.1f}s: {e}")
        raise

def preparation_phase():
    """Fase persiapan environment"""
    
    # Ensure directories exist
    directories = [
        'output/enhanced_clarity',
        'output/final_reports',
        'logs',
        'data/processed'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Validate input files
    required_files = [
        'Annotator_1.xlsx',
        'Annotator_2.xlsx', 
        'Annotator_3.xlsx'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    return f"Environment prepared, {len(directories)} directories created"

def enhanced_merge_phase():
    """Fase enhanced merge annotations"""
    
    import subprocess
    
    cmd = [
        'python', 'enhanced_merge_annotations.py',
        '--input-files', 'Annotator_1.xlsx', 'Annotator_2.xlsx', 'Annotator_3.xlsx',
        '--output-file', 'output/merged_annotations_enhanced.xlsx',
        '--output-dir', 'output/annotation_quality'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Enhanced merge failed: {result.stderr}")
    
    # Validate output
    if not os.path.exists('output/merged_annotations_enhanced.xlsx'):
        raise FileNotFoundError("Enhanced merge output file not created")
    
    return "Enhanced annotation merge completed successfully"

def enhanced_preprocessing_phase():
    """Fase enhanced preprocessing"""
    
    exec(open('enhanced_preprocessing_main.py').read())
    
    # Validate output
    if not os.path.exists('output/enhanced_clarity_analysis.xlsx'):
        raise FileNotFoundError("Enhanced preprocessing output not created")
    
    return "Enhanced preprocessing completed"

def clarity_analysis_phase():
    """Fase analisis clarity yang enhanced"""
    
    exec(open('deep_clarity_analysis.py').read())
    
    return "Enhanced clarity analysis completed"

def optimization_phase():
    """Fase optimasi parameter"""
    
    exec(open('optimize_clarity_parameters.py').read())
    
    return "Parameter optimization completed"

def validation_phase():
    """Fase validasi comprehensive"""
    
    exec(open('validate_clarity_improvements.py').read())
    
    return "Comprehensive validation completed"

def integration_phase():
    """Fase integrasi final"""
    
    exec(open('final_enhanced_clarity.py').read())
    
    return "Final integration completed"

def reporting_phase():
    """Fase comprehensive reporting"""
    
    # Generate all reports
    exec(open('compare_clarity_methods.py').read())
    
    return "Comprehensive reporting completed"

def testing_phase():
    """Fase testing comprehensive"""
    
    exec(open('comprehensive_testing.py').read())
    
    return "System testing completed"

def generate_execution_summary(execution_log):
    """Generate summary eksekusi pipeline"""
    
    total_steps = len(execution_log)
    successful_steps = sum(1 for step in execution_log if step['status'] == 'SUCCESS')
    total_duration = sum(step.get('duration', 0) for step in execution_log)
    
    summary = f"""
ENHANCED CLARITY PIPELINE EXECUTION SUMMARY
==========================================

Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Steps: {total_steps}
Successful Steps: {successful_steps}
Success Rate: {(successful_steps/total_steps)*100:.1f}%
Total Duration: {total_duration:.1f} seconds

STEP-BY-STEP RESULTS:
====================

"""
    
    for step in execution_log:
        status_icon = "✅" if step['status'] == 'SUCCESS' else "❌"
        duration = step.get('duration', 0)
        
        summary += f"{status_icon} {step['step']}: {step['status']} ({duration:.1f}s)\n"
        
        if step['status'] == 'FAILED':
            summary += f"    Error: {step.get('error', 'Unknown error')}\n"
        
        summary += f"    Timestamp: {step['timestamp'].strftime('%H:%M:%S')}\n\n"
    
    # Generate file inventory
    summary += """
GENERATED FILES INVENTORY:
=========================

"""
    
    output_files = []
    for root, dirs, files in os.walk('output'):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            output_files.append((file_path, file_size))
    
    for file_path, file_size in sorted(output_files):
        size_mb = file_size / (1024 * 1024)
        summary += f"• {file_path} ({size_mb:.2f} MB)\n"
    
    summary += f"""

PERFORMANCE METRICS:
===================

- Total Processing Time: {total_duration:.1f} seconds
- Average Step Duration: {total_duration/total_steps:.1f} seconds
- Files Generated: {len(output_files)}
- Total Output Size: {sum(size for _, size in output_files)/(1024*1024):.2f} MB

NEXT STEPS:
==========

1. REVIEW RESULTS:
   • Check comprehensive clarity reports in output/enhanced_clarity/
   • Review validation results and test outcomes
   • Examine parameter optimization recommendations

2. IMPLEMENT IMPROVEMENTS:
   • Apply recommended clarity guidelines
   • Train communication staff on best practices
   • Implement automated clarity checking

3. MONITOR PERFORMANCE:
   • Regular clarity assessment of new documents
   • Track improvement metrics over time
   • Adjust parameters based on feedback

4. SHARE LEARNINGS:
   • Distribute best practice guidelines
   • Conduct training sessions
   • Document lessons learned

CONCLUSION:
==========

"""
    
    if successful_steps == total_steps:
        summary += """🏆 EXCELLENT: Pipeline executed flawlessly
All enhancement objectives achieved successfully.
Ready for production deployment and implementation."""
    elif successful_steps >= total_steps * 0.8:
        summary += """✅ GOOD: Pipeline mostly successful
Minor issues encountered but core objectives met.
Address remaining issues before full deployment."""
    else:
        summary += """⚠️ NEEDS ATTENTION: Significant issues encountered
Review failed steps and address critical problems
before proceeding with implementation."""
    
    # Save execution summary
    with open('output/pipeline_execution_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(summary)
    
    return summary

if __name__ == "__main__":
    execution_log = execute_complete_enhanced_clarity_pipeline()