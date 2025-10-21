from .core import parse_netlist_lines, Circuit

def parse_netlist(file_path: str) -> Circuit:
    """Read a SPICE-like netlist file and return a Circuit object."""
    with open(file_path) as f:
        lines = f.readlines()
    return parse_netlist_lines(lines)
