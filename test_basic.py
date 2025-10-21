from pycircuit_sim.core import Circuit

def test_voltage_divider():
    c = Circuit()
    c.add_resistor("R1", "n1", "0", 1000)
    c.add_resistor("R2", "n1", "n2", 2000)
    c.add_voltage_source("V1", "n2", "0", 5)
    res = c.solve()
    v = res["nodes"]
    assert abs(v["n1"] - 3.3333333333) < 1e-6
    assert abs(v["n2"] - 5.0) < 1e-9
