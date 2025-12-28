# mlka

**mlka** — *Machine Learning helpers by KA*

A lightweight collection of reusable utilities for day-to-day machine learning work. The
package now follows a module-per-domain layout to keep functionality discoverable while
retaining backward-compatible APIs.

## Package layout

```
mlka/
├── diagnostics/
│   └── cross_validation.py  # cross_validate_classification, cross_validate_regression
├── transformers/
│   └── data_cleaning.py     # FixNulls, DropHighlyMissingColumns
└── visualizers/
    ├── tabular.py           # peek_df
    ├── regression.py        # visualize_linear_regression
    └── classification.py    # visualize_logistic_regression
```

Each subpackage exposes its public helpers through ``__init__`` so imports remain concise:

```python
from mlka.visualizers import peek_df, visualize_linear_regression
from mlka.transformers import FixNulls
from mlka.diagnostics import cross_validate_classification, cross_validate_regression
```

Top-level re-exports are also kept for quick interactive work:

```python
from mlka import (
    peek_df,
    FixNulls,
    cross_validate_classification,
    cross_validate_regression,
)
```

Future work: expand ``mlka.diagnostics`` with additional metrics and model health reports.
