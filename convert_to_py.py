#!/usr/bin/env python3

import json
import sys
def notebook_to_script(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            code = ''.join(cell.get('source', []))
            code_cells.append(code)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(code_cells))

    print(f"Extracted {len(code_cells)} code cells into '{output_path}'.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_code_cells.py <input_notebook.ipynb> <output_script.py>")
    else:
        notebook_to_script(sys.argv[1], sys.argv[2])