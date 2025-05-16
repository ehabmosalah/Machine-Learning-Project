#!/usr/bin/env python3
# extract_classes.py
import ast

def extract_classes(source_file, output_file):
    with open(source_file, 'r') as f:
        tree = ast.parse(f.read())

    classes = []
    imports = set()

    # Extract imports and class definitions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            imports.add(ast.unparse(node))
        elif isinstance(node, ast.ClassDef):
            class_code = ast.unparse(node)
            classes.append(class_code)

    # Generate output content
    output_content = []
    output_content.append("# ======================== IMPORTS ========================")
    output_content.extend(sorted(imports))
    output_content.append("\n# ======================== CLASSES ========================")
    output_content.extend(classes)

    # Write to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_content))

if __name__ == "__main__":
    input_file = "ML_Train2.py"  # Replace with your source file
    output_file = "extracted_classes.py"  # Output module name
    extract_classes(input_file, output_file)
    print(f"Classes extracted to {output_file}")