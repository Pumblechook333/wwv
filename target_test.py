import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def create_centered_black_white_cmap():
    """
    Creates a colormap with black in the center and white at the ends.

    Returns:
        A matplotlib.colors.LinearSegmentedColormap object.
    """

    cdict = {
        'red':   [(0.0, 1.0, 1.0),  # White at 0.0
                  (0.5, 0.0, 0.0),  # Black at 0.5
                  (1.0, 1.0, 1.0)], # White at 1.0
        'green': [(0.0, 1.0, 1.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 1.0, 1.0)],
        'blue':  [(0.0, 1.0, 1.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 1.0, 1.0)]
    }
    return colors.LinearSegmentedColormap('black_white', cdict)

def create_centered_colorbar(data, tgt, tick_labels, fSize=12, **kwargs):
    """
    Creates a colorbar centered around a target index 'tgt', using shades of grey.

    Args:
        data: The data being plotted (used to determine the colormap range).
        tgt: The target index around which the colorbar is centered.
        tick_labels: List of labels for the colorbar ticks.
        fSize: Font size for the colorbar labels.
        **kwargs: Additional keyword arguments.
    """

    # Calculate the range of the data, assuming it represents indices
    data_min = 0
    data_max = len(tick_labels) - 1

    # Normalize the data around the target index
    # norm = colors.Normalize(vmin=data_min, vmax=data_max)
    norm = colors.CenteredNorm(halfrange=data_max, vcenter=tgt)

    # Create a colormap that goes from light grey to black to light grey, centered at tgt
    # cmap = plt.cm.get_cmap('seismic') # reversed greys to have black at target
    cmap = create_centered_black_white_cmap()

    # Create a ScalarMappable
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(data) # Required for the colorbar to work

    # Create the colorbar
    cbar = plt.colorbar(sm, pad=0.08)

    # Set the tick positions and labels
    step = 1  # Ticks at each integer index
    ticks = np.arange(data_min, data_max + 1, step)
    cbar.ax.set_yticks(ticks)
    cbar.ax.set_yticklabels(tick_labels)

    # Customize the colorbar appearance
    cbar.ax.tick_params(labelsize=fSize - 2)
    cbar.ax.set_ylabel('Date of Trace', fontsize=fSize)

    #Highlight the target by making its tick label bold.
    ticklabs = cbar.ax.get_yticklabels()
    ticklabs[tgt].set_weight("bold")
    cbar.ax.set_yticklabels(ticklabs)

# Example usage:
tick_labels = ['Oct 7', 'Oct 8', 'Oct 9', 'Oct 10', 'Oct 11', 'Oct 12', 'Oct 13', 'Oct 14']
tgt = 1 # Target index (e.g., 'Oct 10')
data = np.arange(len(tick_labels)) # Example data (indices)

# Create a dummy plot for demonstration (replace with your actual plot)
plt.figure()
plt.imshow(np.array([data]), aspect='auto', cmap='Greys_r', norm=colors.Normalize(vmin=0, vmax=len(tick_labels)-1)) #Just to show the colorbar.
create_centered_colorbar(data, tgt, tick_labels)
plt.show()