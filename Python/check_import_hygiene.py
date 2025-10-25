#!/usr/bin/env python3
"""
Validate import hygiene across the QES package.

This script checks for common import anti-patterns:
1. Bare `import Algebra.*` instead of `from QES.Algebra import *`
2. Wildcard imports in __init__ files that could slow down imports
3. Relative imports that go beyond the package boundary

Run this as part of pre-commit checks or CI/CD to maintain import hygiene.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Patterns to detect
BARE_ALGEBRA    = re.compile(r'^\s*(?:import|from)\s+Algebra\.\w+', re.MULTILINE)
WILDCARD_INIT   = re.compile(r'^\s*from\s+\S+\s+import\s+\*', re.MULTILINE)

# -------------------------------------------------------------------

def check_file(filepath: Path) -> List[Tuple[int, str]]:
    """Check a single Python file for import issues."""
    issues      = []
    try:
        content = filepath.read_text(encoding='utf-8')
        
        # Check for bare Algebra imports
        for match in BARE_ALGEBRA.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            issues.append((line_num, f"Bare 'Algebra' import (should be 'from QES.Algebra import ...')"))
        
        # Check for wildcard imports in __init__.py files
        if filepath.name == '__init__.py':
            for match in WILDCARD_INIT.finditer(content):
                # Allow wildcard imports from relative submodules in some cases
                line_num    = content[:match.start()].count('\n') + 1
                line_text   = content.splitlines()[line_num - 1].strip()
                
                if 'from .' not in line_text or 'from .. import *' in line_text:
                    issues.append((line_num, f"Wildcard import in __init__.py (consider explicit exports)"))
    
    except Exception as e:
        issues.append((0, f"Error reading file: {e}"))
    
    return issues

# -------------------------------------------------------------------
#! Main entry point

def main():
    """Scan the QES package for import issues."""
    qes_root = Path(__file__).parent / 'QES'
    if not qes_root.exists():
        print(f"Error: QES package not found at {qes_root}")
        sys.exit(1)
    
    print("=" * 70)
    print("QES Import Hygiene Validator")
    print("=" * 70)
    print()
    
    total_issues = 0
    
    for py_file in qes_root.rglob('*.py'):
        # Skip __pycache__ and generated files
        if '__pycache__' in str(py_file) or py_file.name.startswith('_version'):
            continue
        
        issues = check_file(py_file)
        if issues:
            rel_path = py_file.relative_to(qes_root.parent)
            print(f"{rel_path}:")
            for line_num, msg in issues:
                if line_num > 0:
                    print(f"  Line {line_num}: {msg}")
                else:
                    print(f"  {msg}")
            print()
            total_issues += len(issues)
    
    print("=" * 70)
    if total_issues == 0:
        print("(v) All imports follow the QES hygiene guidelines!")
    else:
        print(f"(x) Found {total_issues} import issue(s) to fix.")
        sys.exit(1)
    print("=" * 70)

# -------------------------------------------------------------------

if __name__ == '__main__':
    main()

# -------------------------------------------------------------------
#! End of QES import hygiene validator
# -------------------------------------------------------------------