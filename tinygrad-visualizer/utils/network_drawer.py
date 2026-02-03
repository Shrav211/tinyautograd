# utils/network_drawer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class NetworkDrawer:
    def __init__(self, layer_sizes):
        """
        layer_sizes: [2, 4, 1] for XOR network
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
    def draw_network(self, weights=None, activations=None, gradients=None):
        """
        Draw the neural network
        
        weights: list of weight matrices [W1, W2]
        activations: list of activation vectors [a0, a1, a2]
        gradients: list of gradient matrices (for backward pass)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Calculate positions
        layer_positions = self._calculate_positions()
        
        # Draw connections (weights) first (so they're behind neurons)
        if weights is not None:
            self._draw_connections(ax, layer_positions, weights, gradients)
        
        # Draw neurons
        self._draw_neurons(ax, layer_positions, activations)
        
        # Add labels
        self._add_labels(ax, layer_positions)
        
        return fig
    
    def _calculate_positions(self):
        """Calculate (x, y) positions for each neuron"""
        positions = []
        
        # Horizontal spacing between layers
        x_spacing = 1.0 / (self.num_layers - 1)
        
        for layer_idx, layer_size in enumerate(self.layer_sizes):
            x = layer_idx * x_spacing
            
            # Vertical spacing within layer
            y_spacing = 1.0 / (layer_size + 1)
            
            layer_pos = []
            for neuron_idx in range(layer_size):
                y = (neuron_idx + 1) * y_spacing
                layer_pos.append((x, y))
            
            positions.append(layer_pos)
        
        return positions
    
    def _draw_neurons(self, ax, positions, activations=None):
        """Draw circles for neurons"""
        for layer_idx, layer_pos in enumerate(positions):
            for neuron_idx, (x, y) in enumerate(layer_pos):
                # Color by activation value
                if activations is not None and layer_idx < len(activations):
                    value = activations[layer_idx][neuron_idx]
                    color = plt.cm.RdYlGn(0.5 + value * 0.5)  # Map [-1, 1] to colormap
                    alpha = 0.3 + 0.7 * abs(value)  # Transparency by magnitude
                else:
                    color = 'lightblue'
                    alpha = 0.5
                
                # Draw circle
                circle = patches.Circle(
                    (x, y), 
                    radius=0.03, 
                    facecolor=color, 
                    edgecolor='black', 
                    linewidth=2,
                    alpha=alpha
                )
                ax.add_patch(circle)
                
                # Add value text inside neuron
                if activations is not None and layer_idx < len(activations):
                    ax.text(x, y, f'{value:.2f}', 
                           ha='center', va='center', 
                           fontsize=8, fontweight='bold')
    
    def _draw_connections(self, ax, positions, weights, gradients=None):
        """Draw lines for weights between layers"""
        for layer_idx in range(len(weights)):
            W = weights[layer_idx]
            
            in_layer = positions[layer_idx]
            out_layer = positions[layer_idx + 1]
            in_size = len(in_layer)
            out_size = len(out_layer)
            
            # Handle both weight matrix conventions
            if W.shape == (in_size, out_size):
                # Weights are (in_features, out_features) - transpose
                W = W.T
            elif W.shape == (out_size, in_size):
                # Weights are already (out_features, in_features) - keep as is
                pass
            else:
                raise ValueError(
                    f"Weight matrix shape {W.shape} doesn't match layer sizes "
                    f"in={in_size}, out={out_size}. Expected ({in_size}, {out_size}) "
                    f"or ({out_size}, {in_size})"
                )
            
            # Now W is guaranteed to be (out_size, in_size)
            for out_idx, (x2, y2) in enumerate(out_layer):
                for in_idx, (x1, y1) in enumerate(in_layer):
                    weight = W[out_idx, in_idx]
                    
                    # Skip very small weights to reduce clutter
                    if abs(weight) < 0.01:
                        continue
                    
                    # Line properties based on weight value
                    max_weight = max(abs(W).max(), 0.1)  # Avoid division by zero
                    linewidth = 0.5 + 3 * abs(weight) / max_weight
                    color = 'green' if weight > 0 else 'red'
                    alpha = 0.3 + 0.7 * abs(weight) / max_weight
                    
                    # Draw line
                    ax.plot([x1, x2], [y1, y2], 
                        color=color, 
                        linewidth=linewidth, 
                        alpha=alpha,
                        zorder=1)

    def _add_labels(self, ax, positions):
        """Add layer labels"""
        labels = ['Input', 'Hidden', 'Output']
        for layer_idx, layer_pos in enumerate(positions):
            x = layer_pos[0][0]
            label = labels[min(layer_idx, len(labels)-1)]
            ax.text(x, -0.1, label, 
                   ha='center', fontsize=14, fontweight='bold')