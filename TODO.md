# TODO

This file outlines future improvements and additions to the project. Below are tasks categorized by the part of the project they relate to.

---

## General Improvements

1. **Add Docstrings**: Ensure all functions and classes have comprehensive docstrings.
2. **Add Type Hinting**: Add missing type hints for function arguments and return values.
3. **Ensure Consistent Default Values**: Standardize default values across the codebase.
4. **Improve Logging**: Add more granular and consistent logging throughout the scripts.
5. **Add Unit Tests**: Implement unit tests for critical components (e.g., datasets, training pipeline, callbacks).
   - Create a Parent Unit test class that the other inherit from
6. **Clean Code**: Implement unit tests for critical components (e.g., datasets, training pipeline, callbacks).
7. **Add Comments**: Add detailed comments in the code.
8. **Improve naming convention**: Improve the naming of some functions and variables.
9. **Update README Files**: Keep the ReadMe files up to date.

---

## Dataset Enhancements

1. **Dynamic Dataset Support**: Refactor datasets to support more flexible dataset paths.
2. **Implement Lazy Loading**: Optimize memory usage for large datasets by implementing lazy loading.
3. **Extend Dataset Registry**: Add support for additional biomedical datasets.
4. **Dataset Statistics**: Improve the dataset logging to include more detailed class statistics (e.g., mean and variance of pixel intensities).
5. **Dataset class leverage**: Refactor other datasetss to inherit from CustomClass.

---

## Training Pipeline

1. **Implement Cross-Validation**: Add K-Fold cross-validation to improve model robustness.
2. **Enhance Callback Functionality**:
   - Implement additional callbacks (e.g., Gradient Clipping, Custom Logger).
   - Improve `process_callbacks` to dynamically accept user-defined parameters.
3. **Add Support for Custom Loss Functions**: Allow users to define and register custom loss functions.
4. **Model Ensemble**: Implement model ensembling techniques to improve predictions.

---

## Metrics and Evaluation

1. **Implement More Metric Plots**:
   - Precision-Recall Curves
   - ROC Curves
   - Per-class metric visualizations
2. **Add Support for Regression Metrics**: Extend metric functionality to include support for regression tasks (e.g., Mean Squared Error, R-squared).
3. **Generate Reports**: Automatically generate evaluation reports in PDF or HTML format.

---

## Code Quality

1. **Refactor Callbacks**: Simplify the callback design to reduce redundancy.
2. **Optimize Parsing**: Simplify the argument parsing and encourage configuration-driven workflows.
3. **Integrate Pre-commit Hooks**: Use tools like `black` and `flake8` for code formatting and linting.
4. **Optimize Performance**: Identify and optimize performance bottlenecks (e.g., data loading, augmentation).

---

## Documentation

1. **Add Comprehensive Documentation**:
   - Create a user-friendly `docs/` folder with detailed guides for installation, usage, and customization.
   - Include examples for commonly used configurations.
2. **Add README Files for Empty Folders**: Add placeholder `README.md` files for directories like `models/` to explain their intended purpose.

---

## Suggested Implementations

1. **Hyperparameter Tuning**:
   - Integrate hyperparameter tuning libraries like Optuna or Ray Tune for automated optimization.
2. **Distributed Training**:
   - Add support for distributed training using PyTorch's `torch.distributed` or frameworks like `Horovod`.
3. **Integrate Visualization Dashboards**:
   - Use tools like TensorBoard or Weights & Biases for real-time metric visualization.
4. **Model Compression**:
   - Implement model pruning, quantization, or knowledge distillation to reduce model size.
5. **Multi-GPU Training**:
   - Add support for training on multiple GPUs.
6. **Deployable Models**:
   - Create deployment scripts for trained models (e.g., Flask or FastAPI for REST APIs).

---

## Completed Tasks

- **Initial Project Setup**
- **Dataset Registry Implementation**
- **Basic Training Pipeline**

