# pycircuit-sim
A lightweight, Python-based circuit simulator for DC analysis. Designed for education, experimentation, and rapid prototyping of basic electrical circuits.
Designed for education, experimentation, and rapid prototyping of simple electrical circuits.

---

## Features

- Nodal analysis for resistive DC circuits  
- Resistors + independent voltage sources  
- Netlist parser  
- Easy-to-read educational code

---

## Installation

```bash
pip install .
```

or from GitHub:

```bash
pip install git+https://github.com/yourusername/pycircuit-sim.git
```

---

## Example

```python
from pycircuit_sim.core import Circuit

c = Circuit()
c.add_resistor("R1", "n1", "0", 1000)
c.add_resistor("R2", "n1", "n2", 2000)
c.add_voltage_source("V1", "n2", "0", 5)

res = c.solve()
print(res)
```

Output:

```
{'nodes': {'0': 0.0, 'n1': 3.3333, 'n2': 5.0}, 'vsrc_currents': {'V1': 0.0016667}}
```

---

## Netlist Example

```text
* Simple voltage divider
V1 n2 0 5
R1 n1 n2 1000
R2 n1 0 2000
```

Run it:

```bash
python examples/run_example.py examples/voltage_divider.net
```

---

## License

MIT License © 2025
