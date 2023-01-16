'''
MPS BASED ON (BASARIM 2020? edit?) Tüsüz et al. Performance of Particle Tracking Using a Quantum Graph Network
'''
import cirq
import sympy

'''
LOSS IS 1.0! TO DO: RE-IMPLEMENT!
'''
class CircuitLayerBuilder_mps():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout    
        
    def add_layers(self, circuit, qubits, gate1, prefix, gate2):     
        i = 0
        for k, qubit in enumerate(qubits[:-1]):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            ry = gate1(rads=symbol)
            circuit.append(ry(qubit))
            i+=1
            symbol = sympy.Symbol(prefix + '-' + str(i))
            ry = gate1(rads=symbol)
            circuit.append(ry(qubit))
            i+=1
            circuit.append(gate2(qubit, qubits[k+1]))

            
            
def create_mps(observable, grid=[4, 4]):
    data_qubits = cirq.GridQubit.rect(int(grid[0]), int(grid[0]))
    readout = data_qubits[-1]
    circuit = cirq.Circuit()

    builder = CircuitLayerBuilder_mps(
        data_qubits=data_qubits, readout=readout)

    # Add layers
    builder.add_layers(circuit, data_qubits, cirq.Ry, "ry", cirq.CNOT)    

    # Prepare measurement
    if observable == 'x':
        circuit.append(cirq.H(readout))

    if observable == 'y':
        s_dg = cirq.ops.ZPowGate(exponent=-(1 / 2))
        circuit.append(s_dg(readout))
        circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)