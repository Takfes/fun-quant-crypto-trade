# Subpackaging

In a subpackage you can structure your package to allow both types of imports:
To achieve this, you need to:
- Create a subpackage for `structure` with multiple modules (e.g., `extrema.py`, `directional_change.py`, etc.).
- Use an `__init__.py` file in the `structure` subpackage to expose the public API.

### Example Directory Structure
```bash
src/pytrade/features/structure/
    __init__.py
    extrema.py
    directional_change.py
    pips.py
```

### To allow imports from the subpackage
To allow imports from the subpackage (from pytrade.features.structure import ...), you need to expose the public API in __init__.py:

```python
from .extrema import local_maxima_transformer_func, local_minima_transformer_func
from .directional_change import directional_change_transformer_func
from .pips import pips_transformer_func

__all__ = [
    "local_maxima_transformer_func",
    "local_minima_transformer_func",
    "directional_change_transformer_func",
    "pips_transformer_func",
]
```
### How to import
#### Direct Import from a Specific Module:
```python
from pytrade.features.structure.extrema import local_maxima_transformer_func
```

#### Import from the Subpackage:
```python
from pytrade.features.structure import local_maxima_transformer_func
```
