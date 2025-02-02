"""Utils for Hamiltonian simulation."""

import networkx as nx
import numpy as np
from pytket import Qubit
from pytket.circuit import CircBox, Circuit, Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator, gen_term_sequence_circuit
from scipy.linalg import expm

from hlsim.utils import get_pauli_exp_box_from_QubitPauliString


def get_xxz_chain_hamiltonian(n_qubits: int, Delta_ZZ: float) -> QubitPauliOperator:
    sites = nx.path_graph(n_qubits)
    qpo_dict = {}
    for e in sites.edges:
        zz_term = QubitPauliString([Qubit(e[0]), Qubit(e[1])], [Pauli.Z, Pauli.Z])
        xx_term = QubitPauliString([Qubit(e[0]), Qubit(e[1])], [Pauli.X, Pauli.X])
        yy_term = QubitPauliString([Qubit(e[0]), Qubit(e[1])], [Pauli.Y, Pauli.Y])
        qpo_dict[zz_term] = Delta_ZZ
        qpo_dict[xx_term] = 1.0
        qpo_dict[yy_term] = 1.0

    return QubitPauliOperator(qpo_dict)


def get_transverse_field_ising_hamiltonian(
    n_qubits: int, J_ZZ: float
) -> QubitPauliOperator:
    sites = nx.path_graph(n_qubits)
    qpo_dict = {}
    for e in sites.edges:
        zz_term = QubitPauliString([Qubit(e[0]), Qubit(e[1])], [Pauli.Z, Pauli.Z])
        qpo_dict[zz_term] = -J_ZZ
    for node in sites.nodes:
        x_term = QubitPauliString([Qubit(node)], [Pauli.X])
        qpo_dict[x_term] = 1.0

    return QubitPauliOperator(qpo_dict)


def hamiltonian_time_evolution_numpy(
    hamiltonian: QubitPauliOperator, t_trotterization: float, n_qubits
):
    mat = hamiltonian.to_sparse_matrix(
        [Qubit(idx) for idx in range(n_qubits)]
    ).todense()
    U = expm(-1j * t_trotterization * mat)
    return U


def get_hamiltonian_simulation_circbox(
    hamiltonian: QubitPauliOperator,
    n_qubits: int,
    t_trotterization: float,
    n_trotter_steps: int,
):
    trotter_step_size = t_trotterization / n_trotter_steps

    # Bug in docstring of gen_term_sequence_circuit -> no minus required!
    scaled_hamiltonian = trotter_step_size * (2 / np.pi) * hamiltonian

    time_evo_circ = Circuit(n_qubits=n_qubits, name="Time evolution")
    base_circ = Circuit(n_qubits=n_qubits)
    trotter_circ = gen_term_sequence_circuit(
        scaled_hamiltonian, reference_state=base_circ
    )
    trotter_circ.name = "Trotter step"
    trotter_box = CircBox(trotter_circ)
    for _ in range(n_trotter_steps):
        time_evo_circ.add_circbox(trotter_box, range(n_qubits))

    return CircBox(time_evo_circ)


def get_first_order_trotter_step(
    qubit_pauli_operator: QubitPauliOperator, Delta_t: float, qubits: list[Qubit]
):
    # Bug in docstring of gen_term_sequence_circuit -> no minus required!
    scaled_hamiltonian = Delta_t * (2 / np.pi) * qubit_pauli_operator
    n_qubits = len(qubits)

    base_circ = Circuit(n_qubits=n_qubits)
    trotter_circ = gen_term_sequence_circuit(
        scaled_hamiltonian, reference_state=base_circ
    )
    trotter_circ.name = "1st-order Trotter step"

    return CircBox(trotter_circ)


def get_second_order_trotter_step(
    qubit_pauli_operator: QubitPauliOperator, Delta_t: float, qubits: list[Qubit]
):
    circ = Circuit(name="2nd-order Trotter step")
    reg_trotterbox = circ.add_q_register("reg_trotterbox", len(qubits))

    qubit_pauli_strings = list(qubit_pauli_operator._dict.keys())
    coefficients = list(qubit_pauli_operator._dict.values())

    pauli_exp_boxes = []
    for qps, coef in zip(qubit_pauli_strings[:-1], coefficients[:-1]):
        pauli_exp_boxes.append(
            get_pauli_exp_box_from_QubitPauliString(coef / 2, Delta_t, qps, qubits)
        )

    pauli_exp_boxes.append(
        get_pauli_exp_box_from_QubitPauliString(
            coefficients[-1], Delta_t, qubit_pauli_strings[-1], qubits
        )
    )

    for pauli_exp_box in pauli_exp_boxes[:-1]:
        circ.add_pauliexpbox(pauliexpbox=pauli_exp_box, qubits=reg_trotterbox.to_list())
    circ.add_pauliexpbox(
        pauliexpbox=pauli_exp_boxes[-1], qubits=reg_trotterbox.to_list()
    )
    for pauli_exp_box in pauli_exp_boxes[-2::-1]:
        circ.add_pauliexpbox(pauliexpbox=pauli_exp_box, qubits=reg_trotterbox.to_list())

    box = CircBox(circ)

    return box


def get_first_order_trotterization(
    hamiltonian: QubitPauliOperator,
    n_qubits: int,
    t_trotterization: float,
    n_trotter_steps: int,
):
    trotter_step_size = t_trotterization / n_trotter_steps

    qubits = [Qubit(i) for i in range(n_qubits)]

    time_evo_circ = Circuit(n_qubits=n_qubits, name="Time evolution")
    trotter_box = get_first_order_trotter_step(hamiltonian, trotter_step_size, qubits)
    for _ in range(n_trotter_steps):
        time_evo_circ.add_circbox(trotter_box, range(n_qubits))

    return CircBox(time_evo_circ)


def get_second_order_trotterization(
    hamiltonian: QubitPauliOperator,
    n_qubits: int,
    t_trotterization: float,
    n_trotter_steps: int,
):
    trotter_step_size = t_trotterization / n_trotter_steps

    qubits = [Qubit(i) for i in range(n_qubits)]

    time_evo_circ = Circuit(n_qubits=n_qubits, name="Time evolution")
    trotter_box = get_second_order_trotter_step(hamiltonian, trotter_step_size, qubits)
    for _ in range(n_trotter_steps):
        time_evo_circ.add_circbox(trotter_box, range(n_qubits))

    return CircBox(time_evo_circ)
