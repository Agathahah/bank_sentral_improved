#!/usr/bin/env python3
"""
quick_keyerror_fix.py - Quick fix untuk KeyError di line 570
"""

import os

def quick_fix_keyerror():
    """Quick fix untuk KeyError consistency_score"""
    
    file_path = "scripts/create_final_report.py"
    
    print(f"Applying quick fix to: {file_path}")
    
    # Backup
    backup_path = file_path + '.quick_backup'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup created: {backup_path}")
    
    # Apply targeted fixes
    fixes = [
        # Fix 1: The main KeyError line
        (
            "if other_col != col and not pd.isna(corr_matrix.loc[col, other_col]):",
            """if other_col != col and col in corr_matrix.index and other_col in corr_matrix.columns:
                        try:
                            corr_val = corr_matrix.loc[col, other_col]
                            if not pd.isna(corr_val):"""
        ),
        
        # Fix 2: The line that follows (around line 571)
        (
            "consistency_corrs.append((other_col, corr_matrix.loc[col, other_col]))",
            """                                consistency_corrs.append((other_col, corr_val))
                        except (KeyError, IndexError):
                            continue"""
        ),
        
        # Fix 3: Remove all f-string prefixes
        ("f\"\"\"", "\"\"\""),
        ("f\"", "\""),
        ("f'", "'"),
        
        # Fix 4: Replace string formatting in problematic areas
        (
            "- {agreement_level} (Cohen's κ = {kappa:.3f})",
            "- \" + agreement_level + \" (Cohen's κ = \" + str(round(kappa, 3)) + \")\""
        ),
        (
            "- {best_variant}",
            "- \" + best_variant + \"\""
        ),
        (
            "- {best_score:.3f}",
            "- \" + str(round(best_score, 3)) + \"\""
        )
    ]
    
    # Apply fixes
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            print(f"✓ Applied fix: {old[:50]}...")
    
    # Write fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Test syntax
    try:
        compile(content, file_path, 'exec')
        print("✓ Syntax validation successful!")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error at line {e.lineno}: {e.msg}")
        
        # Restore backup
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_content = f.read()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(backup_content)
        print("✗ Restored from backup")
        return False

if __name__ == "__main__":
    success = quick_fix_keyerror()
    if success:
        print("\n✓ Ready to run: python scripts/create_final_report.py ...")
    else:
        print("\n✗ Manual intervention required")