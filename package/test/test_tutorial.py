import os
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def test_tutorial_runs_without_errors():
    """
    Executes the tutorial_walk.ipynb notebook to ensure it runs without errors.
    This catches issues like the heaving_count assertion error in CI.
    """
    # current_dir is .../semi-analytical-hydro/package/test
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up two levels to get to project root, then into docs
    # From package/test -> package -> root -> docs
    notebook_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'docs', 'tutorial_walk.ipynb'))
    
    if not os.path.exists(notebook_path):
        pytest.fail(f"Notebook not found at {notebook_path}")
        
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
        
    # Set timeout to 600s to allow for simulation time
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        # Run the notebook, effectively executing all cells
        # We set the path to the notebook's directory so any relative file operations inside it work
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    except Exception as e:
        pytest.fail(f"Tutorial notebook execution failed: {e}")

if __name__ == "__main__":
    test_tutorial_runs_without_errors()
    print("Tutorial test passed!")