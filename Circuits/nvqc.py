import cirq
import sympy

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout  
        
    def add_layer(self, circuit, qubits, gate1, gate2, prefix1, prefix2):        
        for i in range(1, len(qubits), 2):
            symbol = sympy.Symbol(prefix1 + '-' + str(i))
            ry = gate1(rads=symbol)
            circuit.append(ry(qubits[i]))
        
        n=0
        for i in range(0, len(qubits), 2):
            symbol = sympy.Symbol(prefix2 + '-' + str(n))
            n+=1
            circuit.append(gate2(qubits[i], qubits[i+1])**symbol)

    
def create_nvqc(observable):
    data_qubits = cirq.GridQubit.rect(4, 4)  # 4x4 grid
    circuit = cirq.Circuit()

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits, readout=data_qubits[0])

    qubits=data_qubits[:]
    # Add rectangle layers
    k = 1
    remains = len(qubits)
    while remains >= 2:
        builder.add_layer(circuit, qubits, cirq.Ry, cirq.XX, "ry" + str(k), "xx" + str(k))
        del qubits[1:len(data_qubits):2]
        remains=remains/2
        k += 1

    # Prepare measurement
    if observable == 'x':
        circuit.append(cirq.H(data_qubits[0]))

    if observable == 'y':
        s_dg = cirq.ops.ZPowGate(exponent=-(1 / 2))
        circuit.append(s_dg(data_qubits[0]))
        circuit.append(cirq.H(data_qubits[0]))

    return circuit, cirq.Z(data_qubits[0])