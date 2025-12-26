import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

def convert_percent_to_notebook(input_path, output_path):
    """Convert percent format to proper Jupyter notebook."""

    # Read the file
    with open(input_path, 'r') as f:
        lines = f.readlines()

    cells = []
    current_cell_lines = []
    current_type = None

    for line in lines:
        stripped = line.strip()

        if stripped == '#%% md':
            # Save previous cell if exists
            if current_cell_lines:
                content = ''.join(current_cell_lines).strip()
                if content:
                    if current_type == 'md':
                        cells.append(new_markdown_cell(content))
                    elif current_type == 'code':
                        cells.append(new_code_cell(content))

            # Start new markdown cell
            current_cell_lines = []
            current_type = 'md'

        elif stripped == '#%%':
            # Save previous cell if exists
            if current_cell_lines:
                content = ''.join(current_cell_lines).strip()
                if content:
                    if current_type == 'md':
                        cells.append(new_markdown_cell(content))
                    elif current_type == 'code':
                        cells.append(new_code_cell(content))

            # Start new code cell
            current_cell_lines = []
            current_type = 'code'

        else:
            # Add line to current cell
            current_cell_lines.append(line)

    # Don't forget the last cell
    if current_cell_lines:
        content = ''.join(current_cell_lines).strip()
        if content:
            if current_type == 'md':
                cells.append(new_markdown_cell(content))
            elif current_type == 'code':
                cells.append(new_code_cell(content))

    # Create notebook
    nb = new_notebook(cells=cells)

    # Write to file
    with open(output_path, 'w') as f:
        nbformat.write(nb, f)

    print(f"✅ Converted {input_path.name} → {output_path.name}")

if __name__ == '__main__':
    notebooks_dir = Path('notebooks')

    # Convert all notebooks
    for nb_name in ['01_eda_classification', '02_classification', '03_eda_regression', '04_regression']:
        input_file = notebooks_dir / f'{nb_name}.ipynb'
        output_file = notebooks_dir / f'{nb_name}_fixed.ipynb'

        if input_file.exists():
            convert_percent_to_notebook(input_file, output_file)

    print("\n✅ All notebooks converted!")

