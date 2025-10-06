#!/usr/bin/env python3
"""
Script to fix inconsistent import patterns that cause double module loading.
Converts 'from general_python...' imports to 'from QES.general_python...' imports.

>>> fix_imports_in_file("path/to/your/file.py")
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: from general_python.xxx import yyy
    content = re.sub(
        r'^(\s*)from general_python\.([a-zA-Z0-9_.]+) import (.+)$',
        r'\1from QES.general_python.\2 import \3',
        content,
        flags=re.MULTILINE
    )
    
    # Pattern 2: import general_python.xxx as yyy
    content = re.sub(
        r'^(\s*)import general_python\.([a-zA-Z0-9_.]+)( as .+)?$',
        r'\1import QES.general_python.\2\3',
        content,
        flags=re.MULTILINE
    )
    
    if content != original_content:
        print(f"Fixed imports in: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Main function to fix imports in all Python files."""
    base_path   = Path(__file__).parent / "QES"
    fixed_count = 0
    
    # Only fix files within the QES directory structure
    for py_file in base_path.rglob("*.py"):
        if fix_imports_in_file(py_file):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files.")

if __name__ == "__main__":
    main()