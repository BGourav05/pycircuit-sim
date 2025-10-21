"""Minimal DC circuit solver (nodal analysis) for:
 - Resistors (R)
 - Independent voltage sources (V)

Returns node voltages (w.r.t. ground node named "0") and currents through voltage sources.
"""

from typing import Dict, List
import numpy as np
import re

class Element:
    pass

class Resistor(Element):
    def __init__(self, name: str, n1: str, n2: str, value_ohm: float):
        self.name = name
        self.n1 = n1
        self.n2 = n2
        self.value = float(value_ohm)

class VoltageSource(Element):
    def __init__(self, name: str, positive: str, negative: str, value_volt: float):
        self.name = name
        self.pos = positive
        self.neg = negative
        self.value = float(value_volt)

class Circuit:
    def __init__(self):
        self.resistors: List[Resistor] = []
        self.voltage_sources: List[VoltageSource] = []
        self.nodes = set(["0"])  # ground node always present

    def add_resistor(self, name: str, n1: str, n2: str, value_ohm: float):
        self.resistors.append(Resistor(name, n1, n2, value_ohm))
        self.nodes.update([n1, n2])

    def add_voltage_source(self, name: str, positive: str, negative: str, value_volt: float):
        self.voltage_sources.append(VoltageSource(name, positive, negative, value_volt))
        self.nodes.update([positive, negative])

    def _node_index_map(self) -> Dict[str, int]:
        non_ground = sorted([n for n in self.nodes if n != "0"])
        return {n: i for i, n in enumerate(non_ground)}

    def solve(self) -> Dict[str, float]:
        node_idx = self._node_index_map()
        N = len(node_idx)
        M = len(self.voltage_sources)

        G = np.zeros((N, N), dtype=float)
        B = np.zeros((N, M), dtype=float)
        z = np.zeros((N + M), dtype=float)

        # Stamp resistors
        for r in self.resistors:
            n1, n2 = r.n1, r.n2
            g = 1.0 / r.value
            if n1 != "0" and n2 != "0":
                i = node_idx[n1]; j = node_idx[n2]
                G[i, i] += g
                G[j, j] += g
                G[i, j] -= g
                G[j, i] -= g
            elif n1 == "0" and n2 != "0":
                j = node_idx[n2]
                G[j, j] += g
            elif n2 == "0" and n1 != "0":
                i = node_idx[n1]
                G[i, i] += g

        # Stamp voltage sources
        for k, vs in enumerate(self.voltage_sources):
            p, n = vs.pos, vs.neg
            if p != "0":
                B[node_idx[p], k] = 1.0
            if n != "0":
                B[node_idx[n], k] = -1.0
            z[N + k] = vs.value

        # Construct MNA matrix
        if M > 0:
            top = np.hstack([G, B])
            bottom = np.hstack([B.T, np.zeros((M, M), dtype=float)])
            A = np.vstack([top, bottom])
        else:
            A = G

        try:
            x = np.linalg.solve(A, z)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Circuit matrix singular — likely floating node or disconnected subcircuit.") from e

        node_volts = {}
        for n in self.nodes:
            if n == "0":
                node_volts[n] = 0.0
            else:
                node_volts[n] = float(x[node_idx[n]])

        vsrc_currents = {}
        if M > 0:
            I_v = x[N:N+M]
            for k, vs in enumerate(self.voltage_sources):
                vsrc_currents[vs.name] = float(I_v[k])

        return {"nodes": node_volts, "vsrc_currents": vsrc_currents}

# --- Netlist parsing utilities ---

_SUFFIXES = {
    'G': 1e9,
    'MEG': 1e6,
    'M': 1e6,
    'K': 1e3,
    'KILO': 1e3,
    '': 1.0,
    'm': 1e-3,
    'u': 1e-6,
    'µ': 1e-6,
    'n': 1e-9,
    'p': 1e-12,
}

def parse_value(tok: str) -> float:
    """
    Parse numeric token with optional SI suffixes like 1k, 2.2M, 5m, 10u, etc.
    Falls back to float(tok) and raises ValueError if parsing fails.
    """
    tok = tok.strip()
    # Try plain float first
    try:
        return float(tok)
    except ValueError:
        pass

    # Normalize token: allow forms like "1k", "2.2k", "3.3M"
    m = re.fullmatch(r'([-+]?[0-9]*\.?[0-9]+)([A-Za-zµ]*)', tok)
    if not m:
        raise ValueError(f"Unable to parse numeric value: '{tok}'")
    number_str, suffix = m.group(1), m.group(2)
    suffix = suffix.upper()
    # Some suffix aliases
    if suffix == 'U' or suffix == 'MICRO':
        suffix = 'U'
    # Handle common letter suffixes: K, M, G, m, u, n, p
    mul = None
    # Exact match first for uppercase dictionary
    if suffix in _SUFFIXES:
        mul = _SUFFIXES[suffix]
    else:
        # try single-letter mapping (case-sensitive)
        if suffix.lower() in _SUFFIXES:
            mul = _SUFFIXES[suffix.lower()]
    if mul is None:
        # Unknown suffix; raise
        raise ValueError(f"Unknown suffix '{suffix}' in numeric token '{tok}'")
    return float(number_str) * mul

def parse_netlist_lines(lines: List[str]) -> Circuit:
    """
    Parse a list of netlist lines and return a Circuit object.

    Accepts simple SPICE-like lines:
      Rname node1 node2 value
      Vname pos neg value

    Parsing is tolerant of extra trailing tokens (they are ignored) and understands
    common SI suffixes (e.g., 1k, 2.2M, 10u).
    """
    c = Circuit()
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith('*') or s.startswith(';'):
            continue
        # Split by whitespace and ignore inline comments after a semicolon or '*'
        parts = s.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed element line (expected at least 4 tokens): '{raw}'")
        name = parts[0]
        n1 = parts[1]
        n2 = parts[2]
        val_tok = parts[3]

        tag = name.upper()
        if tag.startswith('R'):
            try:
                val = parse_value(val_tok)
            except ValueError as e:
                raise ValueError(f"Invalid resistor value on line: '{raw}': {e}") from e
            c.add_resistor(name, n1, n2, val)
        elif tag.startswith('V'):
            try:
                val = parse_value(val_tok)
            except ValueError as e:
                raise ValueError(f"Invalid voltage source value on line: '{raw}': {e}") from e
            c.add_voltage_source(name, n1, n2, val)
        else:
            raise ValueError(f"Unknown element type on line: '{raw}'")
    return c

if __name__ == "__main__":
    lines = [
        "V1 n2 0 5",
        "R1 n1 n2 1000",
        "R2 n1 0 2000"
    ]
    circuit = parse_netlist_lines(lines)
    sol = circuit.solve()
    print("Node voltages (V):", sol["nodes"])
    print("Voltage source currents (A):", sol["vsrc_currents"])