'''
BASED ON CIRCUIT 6 FROM HUBREGTSEN ET AL 2021: EVALUATION OF PARAMETERIZED QUANTUM CIRCUITS
'''
import cirq
import sympy


class CircuitLayerBuilder_hvqc():
    def __init__(self, data_qubits):
        self.data_qubits = data_qubits 
        
    def add_oqr_layer(self, circuit, gate1, gate2, prefix1, prefix2):        #oqr=one qubit rotation
        for i, qubit in enumerate(self.data_qubits):
            symbol1 = sympy.Symbol(prefix1 + '-' + str(i))
            symbol2 = sympy.Symbol(prefix1 + '-' + str(i))
            rx = gate1(rads=symbol1)
            rz = gate2(rads=symbol2)
            circuit.append(rx(qubit))
            circuit.append(rz(qubit))

    def add_tqr_layer(self, circuit, qubits, prefix, count):          #tqr=two qubit rotation
        symbol = sympy.Symbol(prefix + '-' + str(count))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]) ** symbol)
    
'''
THIS DOES DEFINITLY NOT WORK ! TO DO: RE-IMPLEMENT !
'''
def create_hvqc(observable, grid=[4, 4]):                                                #svqc=simple variational circuit
    data_qubits = cirq.GridQubit.rect(int(grid[0]), int(grid[1]))
    circuit = cirq.Circuit()
    
    count=0
    
    builder = CircuitLayerBuilder_hvqc(data_qubits=data_qubits)
    
    builder.add_oqr_layer(circuit, cirq.Rx, cirq.Rz, 'rx0', 'rz0')
    for i in range(len(data_qubits)-2, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[15], data_qubits[i]], 'crx', count)
        count+=1
        
    builder.add_tqr_layer(circuit, [data_qubits[14], data_qubits[15]], 'crx', count)
    count+=1

    for i in range(len(data_qubits)-3, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[14], data_qubits[i]], 'crx', count)
        count+=1
        
    builder.add_tqr_layer(circuit, [data_qubits[13], data_qubits[15]], 'crx', count)
    count+=1

    builder.add_tqr_layer(circuit, [data_qubits[13], data_qubits[14]], 'crx', count)
    count+=1

    for i in range(len(data_qubits)-4, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[13], data_qubits[i]], 'crx', count)
        count+=1
    
    builder.add_tqr_layer(circuit, [data_qubits[12], data_qubits[15]], 'crx', count)
    count+=1

    builder.add_tqr_layer(circuit, [data_qubits[12], data_qubits[14]], 'crx', count)
    count+=1
    
    builder.add_tqr_layer(circuit, [data_qubits[12], data_qubits[13]], 'crx', count)
    count+=1
    
    for i in range(len(data_qubits)-5, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[12], data_qubits[i]], 'crx', count)
        count+=1   
        
    for i in range(len(data_qubits)-1, 11, -1):
        builder.add_tqr_layer(circuit, [data_qubits[11], data_qubits[i]], 'crx', count)
        count+=1
    for i in range(len(data_qubits)-6, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[11], data_qubits[i]], 'crx', count)
        count+=1
        
    for i in range(len(data_qubits)-1, 10, -1):
        builder.add_tqr_layer(circuit, [data_qubits[10], data_qubits[i]], 'crx', count)
        count+=1
    for i in range(len(data_qubits)-7, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[10], data_qubits[i]], 'crx', count)
        count+=1
        
    for i in range(len(data_qubits)-1, 9, -1):
        builder.add_tqr_layer(circuit, [data_qubits[9], data_qubits[i]], 'crx', count)
        count+=1
    for i in range(len(data_qubits)-8, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[9], data_qubits[i]], 'crx', count)
        count+=1
        
    for i in range(len(data_qubits)-1, 8, -1):
        builder.add_tqr_layer(circuit, [data_qubits[8], data_qubits[i]], 'crx', count)
        count+=1
    for i in range(len(data_qubits)-9, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[8], data_qubits[i]], 'crx', count)
        count+=1 
        
    for i in range(len(data_qubits)-1, 7, -1):
        builder.add_tqr_layer(circuit, [data_qubits[7], data_qubits[i]], 'crx', count)
        count+=1
    for i in range(len(data_qubits)-10, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[7], data_qubits[i]], 'crx', count)
        count+=1
        
    for i in range(len(data_qubits)-1, 6, -1):
        builder.add_tqr_layer(circuit, [data_qubits[6], data_qubits[i]], 'crx', count)
        count+=1
        
    for i in range(len(data_qubits)-11, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[6], data_qubits[i]], 'crx', count)
        count+=1   
        
    for i in range(len(data_qubits)-1, 5, -1):
        builder.add_tqr_layer(circuit, [data_qubits[5], data_qubits[i]], 'crx', count)
        count+=1
    for i in range(len(data_qubits)-12, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[5], data_qubits[i]], 'crx', count)
        count+=1
        
    for i in range(len(data_qubits)-1, 4, -1):
        builder.add_tqr_layer(circuit, [data_qubits[4], data_qubits[i]], 'crx', count)
        count+=1
    for i in range(len(data_qubits)-13, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[4], data_qubits[i]], 'crx', count)
        count+=1
        
    for i in range(len(data_qubits)-1, 3, -1):
        builder.add_tqr_layer(circuit, [data_qubits[3], data_qubits[i]], 'crx', count)
        count+=1
    for i in range(len(data_qubits)-14, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[3], data_qubits[i]], 'crx', count)
        count+=1
        
    for i in range(len(data_qubits)-1, 2, -1):
        builder.add_tqr_layer(circuit, [data_qubits[2], data_qubits[i]], 'crx', count)
        count+=1
    for i in range(len(data_qubits)-15, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[2], data_qubits[i]], 'crx', count)
        count+=1

    for i in range(len(data_qubits)-1, 1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[1], data_qubits[i]], 'crx', count)
        count+=1
    for i in range(len(data_qubits)-16, -1, -1):
        builder.add_tqr_layer(circuit, [data_qubits[1], data_qubits[i]], 'crx', count)
        count+=1
        
    for i in range(len(data_qubits)-1, 0, -1):
        builder.add_tqr_layer(circuit, [data_qubits[0], data_qubits[i]], 'crx', count)
        count+=1
   
    builder.add_oqr_layer(circuit, cirq.Rx, cirq.Rz, 'rx1', 'rz1')
        
    # Prepare measurement
    if observable == 'x':
        for i, qubit in enumerate(data_qubits):
            circuit.append(cirq.H(qubit))

    if observable == 'y':
        for i, qubit in enumerate(data_qubits):
            s_dg = cirq.ops.ZPowGate(exponent=-(1 / 2))
            circuit.append(s_dg(qubit))
            circuit.append(cirq.H(qubit))
            
    measures = []
    for i, qubit in enumerate(data_qubits):
        measures.append(cirq.Z(qubit))
        
        
    return circuit, measures