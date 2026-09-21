import nbformat
import sys

notebooks = [
    'Task1_AirQuality_Analysis.ipynb',
    'Task2_AirQuality_Prediction.ipynb',
    'Task3_Sentiment_Analysis.ipynb',
    'Task4_Fraud_Detection.ipynb'
]

for nb_file in notebooks:
    print(f"Testing {nb_file}...")
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        code_cells = [cell.source for cell in nb.cells if cell.cell_type == 'code']
        code = '\n'.join(code_cells)
        # Remove magics and block plt.show
        code = '\n'.join(line for line in code.split('\n') if not line.strip().startswith('%'))
        code = code.replace('plt.show()', 'pass')
        
        exec(code, {})
        print(f"SUCCESS: {nb_file}\n")
    except Exception as e:
        print(f"ERROR in {nb_file}: {e}\n")

