### Final Version
import pennylane as qml
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram, circuit_drawer

###################################################################################################
################################   1 Sample    ###########################################################
dev1 = qml.device("default.qubit", wires =2)
dev1.shots=1
dev1.analytic= False

@qml.qnode(dev1)
def circuit(params, AMatrix=None):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.Hermitian(AMatrix, wires=[0,1])) 


##  A Matrix estimation
qubit_target = np.array([1,0,0,1])
state_target =  (1/np.sqrt(2)) *qubit_target

state_target = (1/np.sqrt(2))*np.array([1,0,0,1])
A = np.outer(state_target, state_target)


## Optimization of parameters

weights =[0,0]

def cost(weights):
    return np.abs(circuit(weights,AMatrix=A)-1)**2

opt = qml.AdamOptimizer(stepsize=0.1)

for i in range(2000):
    weights = opt.step(cost, weights)


accuracy = circuit(weights, AMatrix=A)


#########################################   Qiskit validation
circuit = QuantumCircuit(2,2)

circuit.rx(weights[0],0)
circuit.ry(weights[1],0)
circuit.cx(0,1)


print(circuit_drawer(circuit)) 
circuit.measure(0,0)
circuit.draw(output="mpl")

#### Medir la certidumbre total
shots= 2**14
counts= execute(circuit, Aer.get_backend("qasm_simulator"), shots=shots).result().get_counts()
plot_histogram(counts)



###################################################################################################
################################   10  Sample  ##########################################################
dev2 = qml.device("default.qubit", wires =2)
dev2.shots= 10
dev2.analytic = False

@qml.qnode(dev2)
def circuit(params, AMatrix=None):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.Hermitian(AMatrix, wires=[0,1])) 


##  A Matrix estimation
qubit_target = np.array([1,0,0,1])
state_target =  (1/np.sqrt(2)) *qubit_target

state_target = (1/np.sqrt(2))*np.array([1,0,0,1])
A = np.outer(state_target, state_target)


## Optimization of parameters

weights =[0,0]

def cost(weights):
    return np.abs(circuit(weights,AMatrix=A)-1)**2

opt = qml.AdamOptimizer(stepsize=0.1)

for i in range(2000):
    weights = opt.step(cost, weights)


accuracy = circuit(weights, AMatrix=A)  ####  **** change nombre of variable "fidelity"


#########################################   Qiskit validation
circuit = QuantumCircuit(2,2)

circuit.rx(weights[0],0)
circuit.ry(weights[1],0)
circuit.cx(0,1)


print(circuit_drawer(circuit)) 
circuit.measure(0,0)
circuit.draw(output="mpl")

#### Medir la certidumbre total
shots= 2**14
counts= execute(circuit, Aer.get_backend("qasm_simulator"), shots=shots).result().get_counts()
plot_histogram(counts)

###################################################################################################
################################   100  Sample  #########################################################
dev3 = qml.device("default.qubit", wires =2)
dev3.shots= 100
dev3.analytic = False


@qml.qnode(dev3)
def circuit(params, AMatrix=None):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.Hermitian(AMatrix, wires=[0,1])) 


##  A Matrix estimation
qubit_target = np.array([1,0,0,1])
state_target =  (1/np.sqrt(2)) *qubit_target

state_target = (1/np.sqrt(2))*np.array([1,0,0,1])
A = np.outer(state_target, state_target)


## Optimization of parameters

weights =[.15,.25]

def cost(weights):
    return np.abs(circuit(weights,AMatrix=A)-1)**2

opt = qml.AdamOptimizer(stepsize=0.1)

for i in range(2000):
    weights = opt.step(cost, weights)


accuracy = circuit(weights, AMatrix=A)  ####  **** change nombre of variable "fidelity"


#########################################   Qiskit validation
circuit = QuantumCircuit(2,2)

circuit.rx(weights[0],0)
circuit.ry(weights[1],0)
circuit.cx(0,1)


print(circuit_drawer(circuit)) 
circuit.measure(0,0)
circuit.draw(output="mpl")

#### Medir la certidumbre total
shots= 2**14
counts= execute(circuit, Aer.get_backend("qasm_simulator"), shots=shots).result().get_counts()
plot_histogram(counts)

###################################################################################################
################################   1000  Sample   ########################################################
dev4 = qml.device("default.qubit", wires =2)
dev4.shots = 1000
dev4.analytic=False

@qml.qnode(dev4)
def circuit(params, AMatrix=None):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.Hermitian(AMatrix, wires=[0,1])) 


##  A Matrix estimation
qubit_target = np.array([1,0,0,1])
state_target =  (1/np.sqrt(2)) *qubit_target

state_target = (1/np.sqrt(2))*np.array([1,0,0,1])
A = np.outer(state_target, state_target)


## Optimization of parameters

weights =[0,0]

def cost(weights):
    return np.abs(circuit(weights,AMatrix=A)-1)**2

opt = qml.AdamOptimizer(stepsize=0.1)

for i in range(2000):
    weights = opt.step(cost, weights)


accuracy = circuit(weights, AMatrix=A)  ####  **** change nombre of variable "fidelity"


#########################################   Qiskit validation
circuit = QuantumCircuit(2,2)

circuit.rx(weights[0],0)
circuit.ry(weights[1],0)
circuit.cx(0,1)


print(circuit_drawer(circuit)) 
circuit.measure(0,0)
circuit.draw(output="mpl")

#### Medir la certidumbre total
shots= 2**14
counts= execute(circuit, Aer.get_backend("qasm_simulator"), shots=shots).result().get_counts()
plot_histogram(counts)












