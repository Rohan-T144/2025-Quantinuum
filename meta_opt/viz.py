from visualizations import (
    plot_trotter_error_scaling,
    animate_trotter_evolution,
    plot_lindblad_dynamics,
    plot_trotter_decomposition_diagram,
    plot_bloch_sphere_trajectory,
    plot_gate_complexity_heatmap
)

# # View Trotter error scaling
# plot_trotter_error_scaling(n_qubits=2)

# # Create animation of state evolution
# anim = animate_trotter_evolution()
# anim.save('evolution.gif')

# # View dissipation effects
# plot_lindblad_dynamics([0.1, 0.3, 0.5])


# # Show Trotter decomposition concept
# plot_trotter_decomposition_diagram(n_terms=3, time_steps=4)

# Visualize circuit complexity scaling
plot_gate_complexity_heatmap(max_qubits=8, max_trotter_steps=20)

# If you have state evolution data:
# plot_bloch_sphere_trajectory(times, exact_states, trotter_states)
