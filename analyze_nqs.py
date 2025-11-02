#!/usr/bin/env python3
"""
Analysis script to examine current NQS implementation.

This script analyzes:
1. Learning phases and training structure
2. Evaluation function consolidation
3. Unused code and imports
4. Model support and integration

Output: Detailed report in ANALYSIS_REPORT.txt
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

# Add parent directory to path
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

@dataclass
class FunctionInfo:
    name: str
    lineno: int
    length: int
    docstring: bool
    returns_energy: bool = False

class NQSAnalyzer(ast.NodeVisitor):
    """Analyze NQS code structure."""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = set()
        self.current_file = None
        self.lines = []
        
    def visit_FunctionDef(self, node):
        has_docstring = bool(ast.get_docstring(node))
        # Estimate function length
        length = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
        
        # Check if returns energy-related
        source = '\n'.join(self.lines[node.lineno-1:node.end_lineno or node.lineno])
        returns_energy = any(word in source.lower() for word in 
                           ['energy', 'loss', 'objective'])
        
        self.functions.append(FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            length=length,
            docstring=has_docstring,
            returns_energy=returns_energy
        ))
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.add(f"{node.module}.{alias.name}" if node.module else alias.name)
        self.generic_visit(node)

def analyze_file(filepath: Path) -> Dict:
    """Analyze a Python file."""
    print(f"Analyzing {filepath.name}...", end=" ")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        tree = ast.parse(content)
        analyzer = NQSAnalyzer()
        analyzer.lines = lines
        analyzer.visit(tree)
        
        # Count lines
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        
        # Find evaluation functions
        eval_functions = [f for f in analyzer.functions if f.returns_energy]
        
        # Check for problematic patterns
        dead_code = 0
        for line in lines:
            if re.match(r'^\s*#.*def ', line):  # Commented-out function
                dead_code += 1
        
        print("✓")
        return {
            'filepath': str(filepath),
            'total_lines': total_lines,
            'blank_lines': blank_lines,
            'comment_lines': comment_lines,
            'classes': analyzer.classes,
            'num_functions': len(analyzer.functions),
            'num_long_functions': sum(1 for f in analyzer.functions if f.length > 100),
            'num_undocumented': sum(1 for f in analyzer.functions if not f.docstring),
            'eval_functions': eval_functions,
            'imports': analyzer.imports,
            'dead_code_lines': dead_code,
        }
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    """Run analysis."""
    nqs_dir = Path(__file__).parent / 'Python' / 'QES' / 'NQS'
    
    if not nqs_dir.exists():
        print(f"Error: NQS directory not found at {nqs_dir}")
        return
    
    print("=" * 80)
    print("NQS IMPLEMENTATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze key files
    key_files = [
        nqs_dir / 'nqs.py',
        nqs_dir / 'src' / 'nqs_backend.py',
        nqs_dir / 'src' / 'nqs_networks.py',
        nqs_dir / 'src' / 'nqs_physics.py',
    ]
    
    results = {}
    for filepath in key_files:
        if filepath.exists():
            result = analyze_file(filepath)
            if result:
                results[filepath.stem] = result
    
    # Generate report
    report = []
    report.append("ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # File Statistics
    report.append("1. FILE STATISTICS")
    report.append("-" * 80)
    total_code_lines = 0
    total_functions = 0
    
    for name, data in results.items():
        if data:
            code_lines = data['total_lines'] - data['blank_lines'] - data['comment_lines']
            total_code_lines += code_lines
            total_functions += data['num_functions']
            
            report.append(f"\n{name}.py:")
            report.append(f"  Total lines: {data['total_lines']}")
            report.append(f"  Code lines: {code_lines}")
            report.append(f"  Classes: {data['num_classes'] if 'num_classes' in data else len(data['classes'])}")
            report.append(f"  Functions: {data['num_functions']}")
            report.append(f"  Functions >100 LOC: {data['num_long_functions']}")
            report.append(f"  Undocumented functions: {data['num_undocumented']}")
            report.append(f"  Dead code lines: {data['dead_code_lines']}")
    
    report.append(f"\nTotal code lines: {total_code_lines}")
    report.append(f"Total functions: {total_functions}")
    
    # Evaluation Functions Analysis
    report.append("")
    report.append("2. EVALUATION FUNCTIONS (Energy/Loss Computation)")
    report.append("-" * 80)
    
    all_eval_funcs = []
    for name, data in results.items():
        if data and data['eval_functions']:
            report.append(f"\nIn {name}.py:")
            for func in data['eval_functions']:
                report.append(f"  - {func.name} (line {func.lineno}, {func.length} LOC)")
                all_eval_funcs.append((name, func))
    
    if all_eval_funcs:
        report.append(f"\nTotal evaluation functions found: {len(all_eval_funcs)}")
        report.append("ACTION: Review for consolidation and duplication")
    else:
        report.append("WARNING: No energy/loss functions detected!")
    
    # Code Quality
    report.append("")
    report.append("3. CODE QUALITY METRICS")
    report.append("-" * 80)
    
    avg_func_length = total_code_lines / max(total_functions, 1)
    avg_undoc = sum(data.get('num_undocumented', 0) for data in results.values() if data) / len([d for d in results.values() if d])
    
    report.append(f"Average function length: {avg_func_length:.0f} LOC")
    report.append(f"Average undocumented ratio: {avg_undoc:.1%}")
    report.append("")
    report.append("Recommendations:")
    if avg_func_length > 50:
        report.append("  ⚠ Functions are long (>50 LOC) - consider refactoring")
    if avg_undoc > 0.2:
        report.append("  ⚠ Many undocumented functions - add docstrings")
    
    # Imports Analysis
    report.append("")
    report.append("4. IMPORT ANALYSIS")
    report.append("-" * 80)
    
    all_imports = set()
    for data in results.values():
        if data:
            all_imports.update(data['imports'])
    
    report.append(f"Total unique imports: {len(all_imports)}")
    report.append("Top imports:")
    for imp in sorted(list(all_imports))[:20]:
        report.append(f"  - {imp}")
    
    # Output report
    report_text = '\n'.join(report)
    print("\n" + report_text)
    
    # Save to file
    report_path = Path(__file__).parent / 'ANALYSIS_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n✓ Report saved to {report_path}")

if __name__ == '__main__':
    main()
