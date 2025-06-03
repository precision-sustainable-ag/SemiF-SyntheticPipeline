import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from from_root import from_root
from collections import Counter
import numpy as np

def scatter_plot(batch_image_dict, species, filename="batch_sizes_plot.png"):
    # Create lists for height, width, and batch IDs
    heights = []
    widths = []
    batch_ids = []
    
    # Create a unique color for each batch using a colormap
    unique_batch_ids = list(batch_image_dict.keys())
    num_batches = len(unique_batch_ids)
    cmap = plt.get_cmap("tab20")  # Use the "tab20" colormap for distinct colors
    norm = mcolors.Normalize(vmin=0, vmax=num_batches-1)
    
    # Create a dictionary to map each batch_id to a color
    batch_color_map = {batch_id: cmap(norm(i)) for i, batch_id in enumerate(unique_batch_ids)}
    
    # Extract the data from the batch_image_dict
    for batch_id, sizes in batch_image_dict.items():
        for size in sizes:
            heights.append(size[0])  # height
            widths.append(size[1])   # width
            batch_ids.append(batch_id)  # batch_id for coloring
    
    # Convert lists to numpy arrays for easier plotting
    heights = np.array(heights)
    widths = np.array(widths)
    
    # Plot each batch with its corresponding color
    plt.figure(figsize=(10, 6))
    
    for batch_id in unique_batch_ids:
        # Get the color corresponding to the current batch
        batch_color = batch_color_map[batch_id]
        
        # Get the indices of the images that belong to this batch
        batch_indices = [i for i, b_id in enumerate(batch_ids) if b_id == batch_id]
        
        # Plot all the (height, width) points for this batch
        plt.scatter(heights[batch_indices], widths[batch_indices], color=batch_color, label=batch_id, alpha=0.7)
    
    # Add labels, title, and legend
    plt.xlabel('Height')
    plt.ylabel('Width')
    plt.title(f'{species}: Height vs Width of Images by Batch')
    plt.legend(title="Batch IDs", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot to a file
    plt.savefig(from_root(f'{file_path}/{species}_{filename}'), bbox_inches='tight')
    print(f"Plot saved as {filename}")
    plt.close()  # Close the figure to free up memory


def bar_chart_plot(shape_count_dict, species, filename="shape_count_plot.png"):
    # Get unique batch IDs
    unique_batch_ids = list(shape_count_dict.keys())
    num_batches = len(unique_batch_ids)

    # Collect all unique shape counts across batches
    all_shape_counts = set()
    shape_freq_dicts = {}

    for batch_id, shape_counts in shape_count_dict.items():
        freq = Counter(shape_counts)
        shape_freq_dicts[batch_id] = freq
        all_shape_counts.update(freq.keys())

    sorted_shape_counts = sorted(all_shape_counts)
    x = np.arange(len(sorted_shape_counts))  # x locations for shape counts

    # Create a colormap
    cmap = plt.get_cmap("tab20")
    norm = mcolors.Normalize(vmin=0, vmax=num_batches - 1)
    batch_color_map = {batch_id: cmap(norm(i)) for i, batch_id in enumerate(unique_batch_ids)}

    bar_width = 0.8 / num_batches  # total width of bars per group

    plt.figure(figsize=(12, 6))

    for i, batch_id in enumerate(unique_batch_ids):
        freq = shape_freq_dicts[batch_id]
        y = [freq.get(shape_count, 0) for shape_count in sorted_shape_counts]
        offset = (i - num_batches / 2) * bar_width + bar_width / 2
        plt.bar(x + offset, y, width=bar_width, label=batch_id, color=batch_color_map[batch_id])

    plt.xlabel("Number of Shapes")
    plt.ylabel("Frequency")
    plt.title(f"{species}: Frequency of Shape Counts by Batch")
    plt.xticks(x, sorted_shape_counts)
    plt.legend(title="Batch IDs", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y')

    # Save the plot
    plt.savefig(from_root(f'{file_path}/{species}_{filename}'), bbox_inches='tight')
    print(f"Plot saved as {filename}")
    plt.close()

'''
    The jitter plot is used to visualize the distribution of individual data points 
    along an axis by adding small random noise (jitter) to reduce overlap, making it 
    easier to see the spread and density of the data.
'''
def jitter_plot(bbox_dict, species, filename, file_path):
    # Group data by the states that they are from
    # this is done by taking the first two letters 
    # of batch ID keys.
    grouped_data = {}
    for key, values in bbox_dict.items():
        prefix = key[:2]
        if prefix not in grouped_data:
            grouped_data[prefix] = []
        grouped_data[prefix].extend(values)
    unique_states = sorted(grouped_data.keys())
    num_states = len(unique_states)
    
    # ADD COLORING
    cmap = plt.get_cmap("tab20")
    norm = mcolors.Normalize(vmin=0, vmax=num_states - 1)
    color_map = {state: cmap(norm(i)) for i, state in enumerate(unique_states)}
    
    plt.figure(figsize=(10, 2))

    for state in unique_states:
        shape_counts = grouped_data[state]
        x_vals = shape_counts
        y_vals = np.random.uniform(-0.2, 0.2, size=len(x_vals))
        color = color_map[state]
        plt.scatter(x_vals, y_vals, label=state, color=color, alpha=0.7, edgecolor='k', linewidth=0.3)

    plt.yticks([])
    plt.xlabel(filename)
    plt.title(f"{species}: {filename}")
    plt.legend(title="STATES", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x')

    # Save the plot
    plt.savefig(from_root(f'{file_path}/{species}_{filename}.png'), bbox_inches='tight')
    print(f"Plot saved as {filename}.png")
    plt.close()