# utils/network_drawer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class NetworkDrawer:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
    def draw_network(self, weights=None, activations=None, gradients=None, 
                     highlight_layer=None, highlight_connections=None, 
                     flow_direction='forward'):
        """
        Draw the neural network with optional highlighting
        
        weights: list of weight matrices [W1, W2]
        activations: list of activation vectors [a0, a1, a2]
        gradients: list of gradient matrices (for backward pass)
        highlight_layer: int, which layer to highlight (0-indexed)
        highlight_connections: tuple (from_layer, to_layer) to highlight connections
        flow_direction: 'forward' or 'backward'
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Calculate positions
        layer_positions = self._calculate_positions()
        
        # Draw connections (weights) first
        if weights is not None:
            self._draw_connections(
                ax, layer_positions, weights, gradients,
                highlight_connections, flow_direction
            )
        
        # Draw neurons
        self._draw_neurons(
            ax, layer_positions, activations, 
            highlight_layer, flow_direction
        )
        
        # Add labels
        self._add_labels(ax, layer_positions)
        
        # Add flow direction indicator
        if flow_direction:
            self._add_flow_indicator(ax, flow_direction)
        
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
    
    def _draw_neurons(self, ax, positions, activations=None, 
                 highlight_layer=None, flow_direction='forward'):
        """Draw circles for neurons with optional highlighting"""
        for layer_idx, layer_pos in enumerate(positions):
            for neuron_idx, (x, y) in enumerate(layer_pos):
                # Determine if this neuron should be highlighted
                is_highlighted = (highlight_layer is not None and 
                                layer_idx <= highlight_layer)
                
                # Color by activation value
                if activations is not None and layer_idx < len(activations):
                    # Extract scalar value from potentially multi-dimensional array
                    act = activations[layer_idx]
                    
                    # Handle different shapes
                    if act.ndim == 1:
                        # Shape is (n,) - simple 1D array
                        if neuron_idx < len(act):
                            value = float(act[neuron_idx])
                        else:
                            value = 0.0
                    elif act.ndim == 2:
                        # Shape is (batch, n) - take first batch
                        if neuron_idx < act.shape[1]:
                            value = float(act[0, neuron_idx])
                        else:
                            value = 0.0
                    else:
                        value = 0.0
                    
                    # Now value is guaranteed to be a Python float
                    
                    if is_highlighted:
                        # Highlighted neurons pulse
                        if flow_direction == 'forward':
                            color_val = 0.5 + 0.5 * min(abs(value), 1.0)
                            color = plt.cm.Greens(color_val)
                            glow_color = 'lime'
                        else:  # backward
                            color_val = 0.5 + 0.5 * min(abs(value), 1.0)
                            color = plt.cm.Reds(color_val)
                            glow_color = 'red'
                        alpha = 1.0
                        edgecolor = glow_color
                        linewidth = 4
                    else:
                        # Normal neurons - convert colormap to RGBA tuple
                        color_val = 0.5 + value * 0.5
                        color_val = max(0, min(1, color_val))  # Clamp to [0, 1]
                        color = plt.cm.RdYlGn(color_val)
                        alpha = 0.3 + 0.7 * min(abs(value), 1.0)
                        edgecolor = 'black'
                        linewidth = 2
                else:
                    value = 0.0
                    color = 'lightblue'
                    alpha = 0.5
                    edgecolor = 'black'
                    linewidth = 2
                
                # Draw circle
                circle = patches.Circle(
                    (x, y), 
                    radius=0.03, 
                    facecolor=color,
                    edgecolor=edgecolor, 
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=3
                )
                ax.add_patch(circle)
                
                # Add glow effect for highlighted neurons
                if is_highlighted:
                    glow = patches.Circle(
                        (x, y),
                        radius=0.04,
                        facecolor='none',
                        edgecolor=glow_color,
                        linewidth=2,
                        alpha=0.3,
                        zorder=2
                    )
                    ax.add_patch(glow)
                
                # Add value text inside neuron
                if activations is not None and layer_idx < len(activations):
                    ax.text(x, y, f'{value:.2f}', 
                        ha='center', va='center', 
                        fontsize=8, fontweight='bold')
    
    def _draw_connections(self, ax, positions, weights, gradients=None,
                         highlight_connections=None, flow_direction='forward'):
        """Draw lines for weights with optional highlighting"""
        for layer_idx in range(len(weights)):
            W = weights[layer_idx]
            
            in_layer = positions[layer_idx]
            out_layer = positions[layer_idx + 1]
            in_size = len(in_layer)
            out_size = len(out_layer)
            
            # Handle both weight matrix conventions
            if W.shape == (in_size, out_size):
                W = W.T
            
            # Check if these connections should be highlighted
            is_highlighted = (highlight_connections is not None and
                            highlight_connections == (layer_idx, layer_idx + 1))
            
            for out_idx, (x2, y2) in enumerate(out_layer):
                for in_idx, (x1, y1) in enumerate(in_layer):
                    weight = W[out_idx, in_idx]
                    
                    # Skip very small weights
                    if abs(weight) < 0.01:
                        continue
                    
                    # Line properties
                    max_weight = max(abs(W).max(), 0.1)
                    linewidth = 0.5 + 3 * abs(weight) / max_weight
                    
                    if is_highlighted:
                        # Highlighted connections pulse with flow color
                        if flow_direction == 'forward':
                            color = 'lime'
                        else:
                            color = 'red'
                        alpha = 0.8
                        linewidth *= 1.5
                    else:
                        # Normal connections
                        color = 'green' if weight > 0 else 'red'
                        alpha = 0.3 + 0.7 * abs(weight) / max_weight
                    
                    # Draw line
                    ax.plot([x1, x2], [y1, y2], 
                           color=color, 
                           linewidth=linewidth, 
                           alpha=alpha,
                           zorder=1)
    
    def _add_flow_indicator(self, ax, flow_direction):
        """Add a visual indicator of flow direction"""
        if flow_direction == 'forward':
            ax.text(0.5, 1.05, '→ Forward Pass', 
                   transform=ax.transAxes,
                   ha='center', fontsize=16, fontweight='bold',
                   color='green')
        elif flow_direction == 'backward':
            ax.text(0.5, 1.05, '← Backward Pass (Gradients)', 
                   transform=ax.transAxes,
                   ha='center', fontsize=16, fontweight='bold',
                   color='red')

    def _add_labels(self, ax, positions):
        """Add layer labels"""
        labels = ['Input', 'Hidden', 'Output']
        for layer_idx, layer_pos in enumerate(positions):
            x = layer_pos[0][0]
            label = labels[min(layer_idx, len(labels)-1)]
            ax.text(x, -0.1, label, 
                   ha='center', fontsize=14, fontweight='bold')