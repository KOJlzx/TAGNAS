# Notice

This directory adapts message-passing internals and convolution operators from
[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (PyG) to give
the neural architecture search space (`GassoSpace`, see `../gassomodels.py`) a
uniform, NAS-friendly interface across candidate operators.

Files adapted from PyG:

- `message_passing.py`, `message_passing.jinja` — PyG's jinja2-templated
  `MessagePassing` base-class code generation.
- `gcn_conv.py`, `gat_conv.py`, `sage_conv.py`, `cheb_conv.py`, `gin_conv.py`,
  `arma_conv.py`, `edge_conv.py` — PyG's graph convolution layers.
- `utils/jit.py`, `utils/inspector.py`, `utils/typing.py`, `utils/helpers.py`
  — supporting utilities for the above.

Modifications from upstream PyG mainly:
- normalize each operator's constructor signature to a common
  `(in_dim, out_dim, ...)` shape so `gnn_map()` in `__init__.py` can
  instantiate any candidate uniformly during architecture search, and
- add the search-only operators (`ZeroConv`, `LinearConv`, `SkipConv` in
  `ops.py`) that aren't part of upstream PyG.

PyTorch Geometric is distributed under the MIT License:

```
Copyright (c) 2023 PyG Team <team@pyg.org>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
