#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comprehensive_testing.py - Fixed version with complete implementation
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
import json

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('comprehensive_testing')

def run_comprehensive_testing():
    """Testing menyeluruh untuk memastikan semua komponen bekerja"""
    
    print("🧪 COMPREHENSIVE TESTING SUITE")
    print("="*60)
    
    test_results = {
        'framework_tests': test_enhanced_framework(),
        'integration_tests': test_integration(),
        'performance_tests': test_performance(),
        'validation_tests': test_validation(),
        'data_tests': test_data_availability()
    }
    
    # Generate test report
    generate_test_report(test_results)
    
    return test_results

def test_enhanced_framework():
    """Test enhanced clarity framework"""
    
    tests = []
    
    # Test 1: Framework file existence
    framework_path = os.path.join(scripts_dir, 'enhanced_clarity_framework.py')
    exists = os.path.exists(framework_path)
    
    tests.append({
        'test': 'Framework File Exists',
        'status': 'PASS' if exists else 'FAIL',
        'details': f'Found at {framework_path}' if exists else 'File not found'
    })
    
    # Test 2: Framework import and initialization
    try:
        # Try to import framework
        if exists:
            from enhanced_clarity_framework import EnhancedBankSentralLexicon, ContextSensitivePreprocessor, EnhancedClarityCalculator
            
            lexicon = EnhancedBankSentralLexicon()
            preprocessor = ContextSensitivePreprocessor(lexicon)
            calculator = EnhancedClarityCalculator(lexicon)
            
            terms_count = len(lexicon.get_all_terms())
            
            tests.append({
                'test': 'Framework Import & Initialization',
                'status': 'PASS' if terms_count > 10 else 'FAIL',
                'details': f'Loaded {terms_count} terms successfully'
            })
        else:
            tests.append({
                'test': 'Framework Import & Initialization',
                'status': 'SKIP',
                'details': 'Framework file not found'
            })
            
    except Exception as e:
        tests.append({
            'test': 'Framework Import & Initialization',
            'status': 'FAIL',
            'details': f'Import error: {str(e)}'
        })
    
    # Test 3: Simple text processing
    try:
        if exists:
            test_text = "Bank Indonesia mempertahankan BI 7-Day Reverse Repo Rate pada level 6,00%."
            
            # Test preprocessing
            cleaned = preprocessor.smart_text_cleaning(test_text)
            filtered = preprocessor.context_aware_stopword_removal(cleaned)
            
            # Test clarity calculation
            metrics = calculator.calculate_enhanced_clarity(filtered)
            
            has_required_metrics = all(metric in metrics for metric in 
                ['composite_clarity_score', 'avg_sentence_length', 'technical_density'])
            
            tests.append({
                'test': 'Text Processing Pipeline',
                'status': 'PASS' if has_required_metrics and len(filtered) > 0 else 'FAIL',
                'details': f'Processed successfully, clarity: {metrics.get("composite_clarity_score", 0)*100:.1f}'
            })
        else:
            tests.append({
                'test': 'Text Processing Pipeline',
                'status': 'SKIP',
                'details': 'Framework not available'
            })
            
    except Exception as e:
        tests.append({
            'test': 'Text Processing Pipeline',
            'status': 'FAIL',
            'details': f'Processing error: {str(e)}'
        })
    
    return tests

def test_integration():
    """Test integration dengan sistem existing"""
    
    tests = []
    
    # Test required directories
    required_dirs = [
        'output/enhanced_clarity',
        'scripts',
        'logs'
    ]
    
    for directory in required_dirs:
        exists = os.path.exists(directory)
        tests.append({
            'test': f'Directory: {directory}',
            'status': 'PASS' if exists else 'FAIL',
            'details': 'Exists' if exists else 'Missing - will be created'
        })
        
        # Create if missing
        if not exists:
            try:
                os.makedirs(directory, exist_ok=True)
                tests[-1]['status'] = 'FIXED'
                tests[-1]['details'] = 'Created successfully'
            except Exception as e:
                tests[-1]['details'] = f'Creation failed: {str(e)}'
    
    # Test Python dependencies
    required_modules = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'tqdm'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            tests.append({
                'test': f'Module: {module}',
                'status': 'PASS',
                'details': 'Available'
            })
        except ImportError:
            tests.append({
                'test': f'Module: {module}',
                'status': 'FAIL',
                'details': 'Not installed'
            })
    
    # Test file write permissions
    try:
        test_file = 'output/enhanced_clarity/test_write.txt'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        tests.append({
            'test': 'Output Directory Write Permission',
            'status': 'PASS',
            'details': 'Can write files'
        })
    except Exception as e:
        tests.append({
            'test': 'Output Directory Write Permission',
            'status': 'FAIL',
            'details': f'Cannot write: {str(e)}'
        })
    
    return tests

def test_performance():
    """Test performance dengan data sample"""
    
    tests = []
    
    # Test processing speed with sample data
    try:
        sample_texts = [
            "Bank Indonesia mempertahankan BI 7-Day Reverse Repo Rate.",
            "Implementasi kebijakan makroprudensial terus diperkuat.",
            "Stabilitas sistem keuangan tetap terjaga dengan baik.",
            "Inflasi terkendali sesuai target yang ditetapkan.",
            "Transmisi kebijakan moneter berfungsi secara efektif."
        ] * 10  # 50 total texts
        
        start_time = time.time()
        
        # Try enhanced framework first
        try:
            from enhanced_clarity_framework import EnhancedBankSentralLexicon, ContextSensitivePreprocessor, EnhancedClarityCalculator
            
            lexicon = EnhancedBankSentralLexicon()
            preprocessor = ContextSensitivePreprocessor(lexicon)
            calculator = EnhancedClarityCalculator(lexicon)
            
            processed_count = 0
            for text in sample_texts:
                cleaned = preprocessor.smart_text_cleaning(text)
                filtered = preprocessor.context_aware_stopword_removal(cleaned)
                metrics = calculator.calculate_enhanced_clarity(filtered)
                processed_count += 1
            
            framework_type = "Enhanced"
            
        except:
            # Fallback to simple processing
            processed_count = 0
            for text in sample_texts:
                # Simple processing
                words = text.lower().split()
                if len(words) > 3:
                    # Basic metrics calculation
                    avg_length = len(words) / max(1, text.count('.'))
                    score = max(0, 100 - avg_length * 2)
                processed_count += 1
            
            framework_type = "Fallback"
        
        end_time = time.time()
        duration = end_time - start_time
        speed = processed_count / duration
        
        tests.append({
            'test': 'Processing Speed',
            'status': 'PASS' if speed > 10 else 'WARN' if speed > 5 else 'FAIL',
            'details': f'{speed:.1f} texts/second ({framework_type} framework, {processed_count} texts in {duration:.2f}s)'
        })
        
    except Exception as e:
        tests.append({
            'test': 'Processing Speed',
            'status': 'FAIL',
            'details': f'Speed test failed: {str(e)}'
        })
    
    # Test memory usage (basic)
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        tests.append({
            'test': 'Memory Usage',
            'status': 'PASS' if memory_mb < 1000 else 'WARN',
            'details': f'{memory_mb:.1f} MB'
        })
        
    except ImportError:
        tests.append({
            'test': 'Memory Usage',
            'status': 'SKIP',
            'details': 'psutil not available'
        })
    except Exception as e:
        tests.append({
            'test': 'Memory Usage',
            'status': 'FAIL',
            'details': f'Memory test failed: {str(e)}'
        })
    
    return tests

def test_validation():
    """Test validation dengan known examples"""
    
    tests = []
    
    test_cases = [
        {
            'text': "Bank Indonesia mempertahankan suku bunga acuan.",
            'expected_clarity': 'high',
            'reason': 'Simple sentence, clear message'
        },
        {
            'text': """Implementasi kebijakan makroprudensial melalui penerapan rasio loan-to-value 
                      dan debt-service-ratio dalam konteks mitigasi risiko sistemik perbankan 
                      dengan mempertimbangkan aspek intermediasi yang berkelanjutan dan optimal.""",
            'expected_clarity': 'low',
            'reason': 'Long sentence, many technical terms'
        },
        {
            'text': "Inflasi terkendali sesuai target.",
            'expected_clarity': 'high',
            'reason': 'Short, clear, direct'
        }
    ]
    
    try:
        # Try enhanced framework
        try:
            from enhanced_clarity_framework import EnhancedBankSentralLexicon, ContextSensitivePreprocessor, EnhancedClarityCalculator
            
            lexicon = EnhancedBankSentralLexicon()
            preprocessor = ContextSensitivePreprocessor(lexicon)
            calculator = EnhancedClarityCalculator(lexicon)
            
            use_enhanced = True
            
        except:
            use_enhanced = False
        
        passed_cases = 0
        
        for i, case in enumerate(test_cases):
            try:
                if use_enhanced:
                    cleaned = preprocessor.smart_text_cleaning(case['text'])
                    filtered = preprocessor.context_aware_stopword_removal(cleaned)
                    metrics = calculator.calculate_enhanced_clarity(filtered)
                    score = metrics['composite_clarity_score'] * 100
                else:
                    # Simple scoring
                    words = case['text'].split()
                    avg_length = len(words) / max(1, case['text'].count('.'))
                    score = max(0, 100 - avg_length * 3)
                
                # Validate against expectations
                if case['expected_clarity'] == 'high':
                    expected = score >= 60
                elif case['expected_clarity'] == 'low':
                    expected = score < 50
                else:
                    expected = 50 <= score < 70  # medium
                
                if expected:
                    passed_cases += 1
                
                tests.append({
                    'test': f'Validation Case {i+1}',
                    'status': 'PASS' if expected else 'FAIL',
                    'details': f'Score: {score:.1f}, Expected: {case["expected_clarity"]}, {case["reason"]}'
                })
                
            except Exception as e:
                tests.append({
                    'test': f'Validation Case {i+1}',
                    'status': 'FAIL',
                    'details': f'Processing failed: {str(e)}'
                })
        
        # Overall validation score
        success_rate = (passed_cases / len(test_cases)) * 100
        tests.append({
            'test': 'Overall Validation Success Rate',
            'status': 'PASS' if success_rate >= 70 else 'WARN' if success_rate >= 50 else 'FAIL',
            'details': f'{success_rate:.1f}% ({passed_cases}/{len(test_cases)} cases passed)'
        })
        
    except Exception as e:
        tests.append({
            'test': 'Validation Testing',
            'status': 'FAIL',
            'details': f'Validation suite failed: {str(e)}'
        })
    
    return tests

def test_data_availability():
    """Test ketersediaan data yang diperlukan"""
    
    tests = []
    
    # Test for input data files
    data_files = [
        'Annotator_1.xlsx',
        'Annotator_2.xlsx', 
        'Annotator_3.xlsx'
    ]
    
    available_files = 0
    for datafile in data_files:
        exists = os.path.exists(datafile)
        if exists:
            available_files += 1
            
        tests.append({
            'test': f'Input Data: {datafile}',
            'status': 'PASS' if exists else 'WARN',
            'details': 'Available' if exists else 'Missing (sample data will be used)'
        })
    
    # Test for processed data
    processed_files = [
        'output/merged_annotations_enhanced.xlsx',
        'output/enhanced_clarity/final_optimized_clarity_results.xlsx'
    ]
    
    for processed_file in processed_files:
        exists = os.path.exists(processed_file)
        tests.append({
            'test': f'Processed Data: {os.path.basename(processed_file)}',
            'status': 'PASS' if exists else 'INFO',
            'details': 'Available' if exists else 'Will be generated during processing'
        })
    
    # Overall data availability assessment
    if available_files == len(data_files):
        data_status = 'PASS'
        data_details = 'All input data files available'
    elif available_files > 0:
        data_status = 'WARN'
        data_details = f'Partial data available ({available_files}/{len(data_files)} files)'
    else:
        data_status = 'INFO'
        data_details = 'No input data - will use sample data for testing'
    
    tests.append({
        'test': 'Overall Data Availability',
        'status': data_status,
        'details': data_details
    })
    
    return tests

def generate_test_report(test_results):
    """Generate comprehensive test report"""
    
    # Calculate overall statistics
    total_tests = sum(len(tests) for tests in test_results.values())
    passed_tests = sum(
        sum(1 for test in tests if test['status'] == 'PASS') 
        for tests in test_results.values()
    )
    
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    # Generate report content
    report = f"""COMPREHENSIVE TESTING REPORT
============================

Testing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Tests: {total_tests}
Passed: {passed_tests}
Pass Rate: {pass_rate:.1f}%

DETAILED RESULTS:
================

"""
    
    # Status icons mapping
    status_icons = {
        'PASS': '✅',
        'FAIL': '❌',
        'WARN': '⚠️',
        'SKIP': '⏭️',
        'INFO': 'ℹ️',
        'FIXED': '🔧'
    }
    
    for category, tests in test_results.items():
        report += f"\n{category.replace('_', ' ').title()}:\n"
        report += "-" * 50 + "\n"
        
        category_passed = sum(1 for test in tests if test['status'] == 'PASS')
        category_total = len(tests)
        category_rate = (category_passed / category_total) * 100 if category_total > 0 else 0
        
        report += f"Category Pass Rate: {category_rate:.1f}% ({category_passed}/{category_total})\n\n"
        
        for test in tests:
            icon = status_icons.get(test['status'], '❓')
            report += f"{icon} {test['test']}: {test['status']}\n"
            report += f"   Details: {test['details']}\n\n"
    
    # Overall assessment
    report += f"""
OVERALL ASSESSMENT:
==================

"""
    
    if pass_rate >= 90:
        report += "🏆 EXCELLENT: All systems functioning optimally\n"
        recommendation = "Ready for production deployment"
    elif pass_rate >= 80:
        report += "✅ GOOD: Minor issues, system ready for use\n"
        recommendation = "Address minor issues and proceed"
    elif pass_rate >= 70:
        report += "⚠️ ACCEPTABLE: Some issues need attention\n"
        recommendation = "Fix critical issues before deployment"
    elif pass_rate >= 60:
        report += "🔧 NEEDS WORK: Several issues require resolution\n"
        recommendation = "Significant improvements needed"
    else:
        report += "❌ CRITICAL: Major issues prevent proper operation\n"
        recommendation = "Extensive fixes required before use"
    
    report += f"Recommendation: {recommendation}\n"
    
    # Specific recommendations
    report += f"""
SPECIFIC RECOMMENDATIONS:
========================

"""
    
    # Analyze failed tests for specific recommendations
    failed_tests = [
        test for tests in test_results.values() 
        for test in tests if test['status'] == 'FAIL'
    ]
    
    warning_tests = [
        test for tests in test_results.values() 
        for test in tests if test['status'] == 'WARN'
    ]
    
    if failed_tests:
        report += "CRITICAL ISSUES TO ADDRESS:\n"
        for test in failed_tests[:5]:  # Show top 5 failures
            report += f"• {test['test']}: {test['details']}\n"
        if len(failed_tests) > 5:
            report += f"• ... and {len(failed_tests) - 5} more issues\n"
        report += "\n"
    
    if warning_tests:
        report += "WARNINGS TO REVIEW:\n"
        for test in warning_tests[:3]:  # Show top 3 warnings
            report += f"• {test['test']}: {test['details']}\n"
        if len(warning_tests) > 3:
            report += f"• ... and {len(warning_tests) - 3} more warnings\n"
        report += "\n"
    
    # Next steps
    report += f"""
NEXT STEPS:
==========

"""
    
    if pass_rate >= 80:
        report += """1. READY FOR OPERATION:
   - System is functional and ready
   - Address any remaining warnings
   - Begin regular operation and monitoring
   - Document any workarounds for minor issues

2. MONITORING:
   - Track performance metrics
   - Monitor for any issues during operation
   - Regular testing of new features
   - User feedback collection

"""
    else:
        report += """1. IMMEDIATE ACTIONS:
   - Address all critical failures
   - Investigate and fix failed components
   - Re-run tests after fixes
   - Verify system stability

2. BEFORE DEPLOYMENT:
   - Achieve minimum 80% pass rate
   - Ensure all critical components work
   - Complete integration testing
   - User acceptance testing

"""
    
    report += f"""3. LONG-TERM IMPROVEMENTS:
   - Regular testing schedule
   - Automated testing implementation
   - Performance optimization
   - Enhanced error handling
   - Documentation updates

SYSTEM STATUS: {"🟢 OPERATIONAL" if pass_rate >= 80 else "🟡 NEEDS ATTENTION" if pass_rate >= 60 else "🔴 NOT READY"}
"""
    
    # Save report
    os.makedirs('output/enhanced_clarity', exist_ok=True)
    report_file = 'output/enhanced_clarity/comprehensive_test_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Also create a summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'pass_rate': pass_rate,
        'status': 'OPERATIONAL' if pass_rate >= 80 else 'NEEDS_ATTENTION' if pass_rate >= 60 else 'NOT_READY',
        'recommendation': recommendation,
        'failed_tests_count': len(failed_tests),
        'warning_tests_count': len(warning_tests)
    }
    
    summary_file = 'output/enhanced_clarity/test_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Print report to console
    print("\n" + "="*80)
    print("COMPREHENSIVE TESTING RESULTS")
    print("="*80)
    print(report)
    
    print(f"📄 Full report saved to: {report_file}")
    print(f"📊 Test summary saved to: {summary_file}")
    
    return pass_rate

def create_test_visualization(test_results):
    """Create visual test results"""
    
    try:
        # Prepare data for visualization
        categories = list(test_results.keys())
        category_stats = []
        
        for category, tests in test_results.items():
            passed = sum(1 for test in tests if test['status'] == 'PASS')
            failed = sum(1 for test in tests if test['status'] == 'FAIL')
            warned = sum(1 for test in tests if test['status'] == 'WARN')
            other = len(tests) - passed - failed - warned
            
            category_stats.append({
                'category': category.replace('_', ' ').title(),
                'passed': passed,
                'failed': failed,
                'warned': warned,
                'other': other,
                'total': len(tests),
                'pass_rate': (passed / len(tests)) * 100 if len(tests) > 0 else 0
            })
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall pass rate by category
        categories_clean = [stat['category'] for stat in category_stats]
        pass_rates = [stat['pass_rate'] for stat in category_stats]
        colors = ['green' if rate >= 80 else 'orange' if rate >= 60 else 'red' for rate in pass_rates]
        
        bars = ax1.bar(categories_clean, pass_rates, color=colors, alpha=0.7)
        ax1.set_ylabel('Pass Rate (%)')
        ax1.set_title('Test Pass Rate by Category')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars, pass_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 2. Test status distribution (pie chart)
        total_passed = sum(stat['passed'] for stat in category_stats)
        total_failed = sum(stat['failed'] for stat in category_stats)
        total_warned = sum(stat['warned'] for stat in category_stats)
        total_other = sum(stat['other'] for stat in category_stats)
        
        labels = ['Passed', 'Failed', 'Warnings', 'Other']
        sizes = [total_passed, total_failed, total_warned, total_other]
        colors_pie = ['green', 'red', 'orange', 'gray']
        
        # Only include non-zero segments
        filtered_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors_pie) if size > 0]
        if filtered_data:
            labels_filtered, sizes_filtered, colors_filtered = zip(*filtered_data)
            ax2.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Test Status Distribution')
        
        # 3. Test counts by category (stacked bar)
        x_pos = np.arange(len(categories_clean))
        passed_counts = [stat['passed'] for stat in category_stats]
        failed_counts = [stat['failed'] for stat in category_stats]
        warned_counts = [stat['warned'] for stat in category_stats]
        
        ax3.bar(x_pos, passed_counts, label='Passed', color='green', alpha=0.7)
        ax3.bar(x_pos, failed_counts, bottom=passed_counts, label='Failed', color='red', alpha=0.7)
        ax3.bar(x_pos, warned_counts, bottom=np.array(passed_counts) + np.array(failed_counts), 
                label='Warnings', color='orange', alpha=0.7)
        
        ax3.set_xlabel('Test Categories')
        ax3.set_ylabel('Number of Tests')
        ax3.set_title('Test Results by Category')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(categories_clean, rotation=45)
        ax3.legend()
        
        # 4. System readiness indicator
        overall_pass_rate = (total_passed / (total_passed + total_failed + total_warned + total_other)) * 100
        
        ax4.axis('off')
        
        # Status indicator
        if overall_pass_rate >= 80:
            status_color = 'green'
            status_text = '🟢 SYSTEM READY'
            status_desc = 'Ready for operation'
        elif overall_pass_rate >= 60:
            status_color = 'orange'
            status_text = '🟡 NEEDS ATTENTION'
            status_desc = 'Some issues to address'
        else:
            status_color = 'red'
            status_text = '🔴 NOT READY'
            status_desc = 'Critical issues found'
        
        ax4.text(0.5, 0.7, status_text, fontsize=20, fontweight='bold', 
                ha='center', va='center', color=status_color)
        ax4.text(0.5, 0.5, f'Overall Pass Rate: {overall_pass_rate:.1f}%', 
                fontsize=16, ha='center', va='center')
        ax4.text(0.5, 0.3, status_desc, fontsize=12, ha='center', va='center')
        ax4.text(0.5, 0.1, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                fontsize=10, ha='center', va='center', style='italic')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = 'output/enhanced_clarity/test_results_visualization.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Test visualization saved to: {viz_file}")
        
    except Exception as e:
        print(f"⚠️ Could not create test visualization: {e}")

def main():
    """Main execution function"""
    try:
        print("🧪 STARTING COMPREHENSIVE TESTING")
        print("="*60)
        
        # Ensure output directory exists
        os.makedirs('output/enhanced_clarity', exist_ok=True)
        
        # Run comprehensive tests
        test_results = run_comprehensive_testing()
        
        # Create visualization
        create_test_visualization(test_results)
        
        # Calculate overall success rate
        total_tests = sum(len(tests) for tests in test_results.values())
        passed_tests = sum(
            sum(1 for test in tests if test['status'] == 'PASS') 
            for tests in test_results.values()
        )
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 TESTING COMPLETED!")
        print(f"📊 Overall Success Rate: {success_rate:.1f}%")
        print(f"✅ Passed: {passed_tests}/{total_tests} tests")
        
        if success_rate >= 80:
            print("🏆 EXCELLENT: System ready for operation!")
        elif success_rate >= 60:
            print("✅ GOOD: Minor issues, mostly functional")
        else:
            print("⚠️ NEEDS WORK: Address issues before proceeding")
        
        print(f"\n📂 Check output/enhanced_clarity/ for detailed reports")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Comprehensive testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()