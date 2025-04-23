 # Tests Directory

 This directory contains unit tests to validate the functionality of data processing modules, dataset implementations, and model components.

 ## Test Files

 - `test-graph_builder.py`: Tests for graph construction logic (node and edge extraction, sampling, snapshot creation).
 - `test_dataset.py`: Tests for dataset loading, tensor preparation, and mini-batch/subgraph sampling.
 - `test_models.py`: Tests for model layer computations and STGformer forward pass.

 ## Running Tests

 To execute the full test suite, run:
 ```bash
 pytest
 ```

 For more verbose output:
 ```bash
 pytest -v
 ```