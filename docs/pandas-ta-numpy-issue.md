
### install
```bash
uv add pandas_ta
```

### in this file
```bash
.venv/lib/python3.13/site-packages/pandas_ta/momentum/squeeze_pro.py
```

### replace this (should be line 2)
```python
from numpy import NaN as npNaN
```

### with this
```python
from numpy import nan as npNaN
```
