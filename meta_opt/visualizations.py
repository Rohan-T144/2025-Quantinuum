"""Visualization utilities for quantum simulation concepts."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Tuple, List, Optional
from hamiltonian_sim import (
    get_xxz_chain_hamiltonian,
    hamiltonian_time_evolution_numpy
)

def plot_trotter_error_scaling(
    n_qubits: int = 2,
    time_points: int = 100,
    max_steps: int = 50,
    step_spacing: int = 5
) -> None:
    """Visualize how Trotter error scales with number of steps.

    Creates a plot showing:
    1. Error vs number of Trotter steps
    2. Circuit depth vs number of steps
    3. The theoretical bounds
    """
    # Get system Hamiltonian
    H = get_xxz_chain_hamiltonian(n_qubits, Delta_ZZ=1.0)
    H_mat = H.to_sparse_matrix().todense()

    # Time points and step counts to evaluate
    t = np.linspace(0, 10, time_points)
    steps = np.arange(1, max_steps + 1, step_spacing)

    # Calculate exact evolution
    U_exact = hamiltonian_time_evolution_numpy(H, t[-1], n_qubits)
    psi0 = np.zeros(2**n_qubits)
    psi0[0] = 1
    psi_exact = U_exact @ psi0

    # Calculate errors for different numbers of steps
    errors = []
    depths = []
    for n_steps in steps:
        dt = t[-1] / n_steps
        U_trotter = np.eye(2**n_qubits)
        for _ in range(n_steps):
            U_step = hamiltonian_time_evolution_numpy(H, dt, n_qubits)
            U_trotter = U_step @ U_trotter

        psi_trotter = U_trotter @ psi0
        error = np.linalg.norm(psi_exact - psi_trotter)
        errors.append(error)
        depths.append(3 * n_steps * (n_qubits - 1))  # Approximate circuit depth

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Error scaling plot
    ax1.plot(steps, errors, 'b.-', label='Actual Error')
    ax1.plot(steps, 1/steps**2, 'r--', label='O(1/n²) Bound', alpha=0.7)
    ax1.set_xlabel('Number of Trotter Steps')
    ax1.set_ylabel('Error')
    ax1.set_yscale('log')
    ax1.set_title('Trotter Error Scaling')
    ax1.legend()
    ax1.grid(True)

    # Circuit complexity plot
    ax2.plot(steps, depths, 'g.-')
    ax2.set_xlabel('Number of Trotter Steps')
    ax2.set_ylabel('Approximate Circuit Depth')
    ax2.set_title('Circuit Complexity')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def animate_trotter_evolution(
    n_qubits: int = 2,
    total_time: float = 5.0,
    n_frames: int = 50,
    n_steps: int = 10,
) -> FuncAnimation:
    """Create an animation showing state evolution under Trotterization.

    Shows the real and imaginary parts of the quantum state as it evolves
    under both exact and Trotterized time evolution.
    """
    H = get_xxz_chain_hamiltonian(n_qubits, Delta_ZZ=1.0)
    H_mat = H.to_sparse_matrix().todense()
    dim = 2**n_qubits

    # Initial state
    psi0 = np.zeros(dim)
    psi0[0] = 1

    # Time points
    times = np.linspace(0, total_time, n_frames)
    dt = total_time / n_steps

    # Pre-calculate exact evolution
    exact_states = []
    trotter_states = []

    # Exact evolution
    for t in times:
        U = hamiltonian_time_evolution_numpy(H, t, n_qubits)
        exact_states.append(U @ psi0)

    # Trotter evolution
    U_step = hamiltonian_time_evolution_numpy(H, dt, n_qubits)
    state = psi0.copy()
    step_idx = 0
    for t in times:
        trotter_states.append(state.copy())
        if t >= (step_idx + 1) * dt and step_idx < n_steps:
            state = U_step @ state
            step_idx += 1

    # Set up the animation
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Quantum State Evolution: Exact vs Trotter', fontsize=14)

    bar_width = 0.35
    x = np.arange(dim)

    def update(frame):
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()

        # Exact evolution
        exact = exact_states[frame]
        ax1.bar(x, np.real(exact), bar_width, label='Real')
        ax1.set_title('Exact Evolution (Real)')
        ax2.bar(x, np.imag(exact), bar_width, color='orange', label='Imaginary')
        ax2.set_title('Exact Evolution (Imaginary)')

        # Trotter evolution
        trotter = trotter_states[frame]
        ax3.bar(x, np.real(trotter), bar_width, label='Real')
        ax3.set_title('Trotter Evolution (Real)')
        ax4.bar(x, np.imag(trotter), bar_width, color='orange', label='Imaginary')
        ax4.set_title('Trotter Evolution (Imaginary)')

        for ax in (ax1, ax2, ax3, ax4):
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

    anim = FuncAnimation(
        fig, update, frames=n_frames,
        interval=50, blit=False
    )

    return anim

def plot_lindblad_dynamics(epsilon_range: List[float] = [0.1, 0.5, 1.0]) -> None:
    """Visualize the effect of dissipation strength on system dynamics.

    Creates plots showing:
    1. System energy over time for different dissipation strengths
    2. Decoherence rates and timescales
    """
    times = np.linspace(0, 10, 100)

    plt.figure(figsize=(10, 6))

    for epsilon in epsilon_range:
        # Simplified model of energy decay
        energy = np.exp(-epsilon * times) * np.cos(2*times)
        plt.plot(times, energy, label=f'ε = {epsilon}')

    plt.xlabel('Time')
    plt.ylabel('System Energy')
    plt.title('Energy Decay Under Lindbladian Evolution')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_trotter_decomposition_diagram(n_terms: int = 3, time_steps: int = 4) -> None:
    """Creates a visual diagram showing how Trotter decomposition works.

    Generates a colored grid showing how different terms in the Hamiltonian
    are applied sequentially in the Trotter decomposition.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Exact evolution
    ax1.set_title("Exact Evolution", pad=20)
    ax1.add_patch(plt.Rectangle((0, 0), time_steps, n_terms,
                               color='purple', alpha=0.3))
    ax1.text(time_steps/2, n_terms/2, "e^{-iHt}",
             horizontalalignment='center', verticalalignment='center', fontsize=14)

    # Trotter evolution
    ax2.set_title("Trotterized Evolution", pad=20)
    colors = ['red', 'green', 'blue']
    for t in range(time_steps):
        for i in range(n_terms):
            ax2.add_patch(plt.Rectangle((t, i), 0.8, 0.8,
                                      color=colors[i % len(colors)],
                                      alpha=0.3))
            ax2.text(t+0.4, i+0.4, f"e^{{-iH_{i+1}t/n}}",
                    horizontalalignment='center',
                    verticalalignment='center')

    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, time_steps + 0.5)
        ax.set_ylim(-0.5, n_terms + 0.5)
        ax.set_xticks(range(time_steps + 1))
        ax.set_yticks(range(n_terms + 1))
        ax.grid(True)
        ax.set_xlabel("Time steps")

    plt.tight_layout()
    plt.show()

def plot_bloch_sphere_trajectory(
    times: np.ndarray,
    exact_states: List[np.ndarray],
    trotter_states: List[np.ndarray]
) -> None:
    """Plot quantum state evolution on the Bloch sphere.

    Shows both exact and Trotterized evolution trajectories.
    """
    from mpl_toolkits.mplot3d import Axes3D

    def get_bloch_coords(state):
        """Convert quantum state to Bloch sphere coordinates."""
        x = 2 * np.real(state[0] * np.conj(state[1]))
        y = 2 * np.imag(state[0] * np.conj(state[1]))
        z = abs(state[0])**2 - abs(state[1])**2
        return x, y, z

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # Plot trajectories
    exact_points = np.array([get_bloch_coords(state) for state in exact_states])
    trotter_points = np.array([get_bloch_coords(state) for state in trotter_states])

    ax.plot(exact_points[:, 0], exact_points[:, 1], exact_points[:, 2],
            'r-', label='Exact', linewidth=2)
    ax.plot(trotter_points[:, 0], trotter_points[:, 1], trotter_points[:, 2],
            'g--', label='Trotter', linewidth=2)

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('State Evolution on Bloch Sphere')
    ax.legend()

    plt.show()

def plot_gate_complexity_heatmap(
    max_qubits: int = 8,
    max_trotter_steps: int = 20
) -> None:
    """Create a heatmap showing circuit complexity vs system size and Trotter steps."""
    qubits = np.arange(2, max_qubits + 1)
    steps = np.arange(1, max_trotter_steps + 1)

    # Approximate gate counts (simplified model)
    gate_counts = np.zeros((len(qubits), len(steps)))
    for i, n_qubits in enumerate(qubits):
        for j, n_steps in enumerate(steps):
            # Approximate formula for gate count
            gate_counts[i, j] = n_steps * (3 * (n_qubits - 1))

    plt.figure(figsize=(10, 8))
    sns.heatmap(gate_counts,
                xticklabels=steps,
                yticklabels=qubits,
                cmap='viridis')

    plt.xlabel('Number of Trotter Steps')
    plt.ylabel('Number of Qubits')
    plt.title('Circuit Complexity (Gate Count)')
    plt.show()
