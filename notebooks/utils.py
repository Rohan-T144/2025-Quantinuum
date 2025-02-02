"""Utils for Hamiltonian and Lindbladian simulation."""

import networkx as nx
import numpy as np
from pytket.circuit import PauliExpBox, Qubit
from pytket.pauli import QubitPauliString


def get_ancilla_and_system_qubits(n_qubits):
    sites = nx.path_graph(n_qubits)
    nodes = list(sites.nodes)
    ancilla_qubits = [Qubit(0)]
    system_qubits = [Qubit(i + 1) for i in nodes]

    return ancilla_qubits, system_qubits


def get_pauli_list_from_QubitPauliString(qubit_pauli_string: QubitPauliString, qubits):
    pauli_list = [qubit_pauli_string.__getitem__(q) for q in qubits]
    return pauli_list


def get_pauli_exp_box_from_QubitPauliString(
    coefficient, Delta_t, qubit_pauli_string, qubits
):
    pauli_list = get_pauli_list_from_QubitPauliString(qubit_pauli_string, qubits)
    pauli_exp_box = PauliExpBox(
        paulis=pauli_list, t=coefficient * (2 / np.pi) * Delta_t
    )
    return pauli_exp_box
