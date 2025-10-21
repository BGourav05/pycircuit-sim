import sys
from pycircuit_sim.netlist import parse_netlist

if len(sys.argv) != 2:
    print("Usage: python run_example.py <netlist_file>")
    sys.exit(1)

ckt = parse_netlist(sys.argv[1])
res = ckt.solve()
print("Node Voltages (V):", res["nodes"])
print("Voltage Source Currents (A):", res["vsrc_currents"])
