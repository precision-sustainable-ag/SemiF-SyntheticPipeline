import numpy as np
import pandas as pd
import seaborn as sns
from from_root import from_root
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def scatter_plot(batch_image_dict, species, file_name, file_path, title_info):
    # Create lists for height, width, and state codes
    heights = []
    widths = []
    state_codes = []

    # Group images by state (first 2 characters of batch ID)
    grouped_data = {}
    for batch_id, sizes in batch_image_dict.items():
        state = batch_id[:2]
        if state not in grouped_data:
            grouped_data[state] = []
        grouped_data[state].extend(sizes)

    # Sorted list of unique state codes
    unique_states = sorted(grouped_data.keys())
    num_states = len(unique_states)

    # Assign a unique color to each state
    cmap = plt.get_cmap("tab20")
    norm = mcolors.Normalize(vmin=0, vmax=num_states - 1)
    state_color_map = {state: cmap(norm(i)) for i, state in enumerate(unique_states)}

    # Collect data for plotting
    all_heights = []
    all_widths = []
    all_states = []

    for state, sizes in grouped_data.items():
        for size in sizes:
            all_heights.append(size[0])
            all_widths.append(size[1])
            all_states.append(state)

    # Convert to numpy arrays for indexing
    all_heights = np.array(all_heights)
    all_widths = np.array(all_widths)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    for state in unique_states:
        indices = [i for i, s in enumerate(all_states) if s == state]
        plt.scatter(
            all_heights[indices],
            all_widths[indices],
            color=state_color_map[state],
            label=state,
            alpha=0.7
        )

    # Add labels and title
    plt.xlabel('Height')
    plt.ylabel('Width')
    plt.title(f'{species}: Height vs Width of Images by State')
    plt.legend(title="States", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save plot
    file_name = f'{species.lower()}_{title_info.lower()}_{file_name.lower()}.png'
    file_name = file_name.replace(" ", "_").lower()
    plt.savefig(from_root(f'{file_path}/{file_name}'), bbox_inches='tight')
    print(f"Plot saved as {file_name}")
    plt.close()

def bar_chart_plot(shape_count_dict, species, file_name, file_path, title_info):
    # Group shape counts by state (first 2 characters of batch ID)
    grouped_data = {}
    for batch_id, shape_counts in shape_count_dict.items():
        state = batch_id[:2]
        if state not in grouped_data:
            grouped_data[state] = []
        grouped_data[state].extend(shape_counts)

    # Get unique state codes
    unique_states = sorted(grouped_data.keys())
    num_states = len(unique_states)

    # Collect all unique shape counts across states
    all_shape_counts = set()
    shape_freq_dicts = {}

    for state, shape_counts in grouped_data.items():
        filtered_counts = [s for s in shape_counts if s is not None]
        freq = Counter(filtered_counts)
        shape_freq_dicts[state] = freq
        all_shape_counts.update(freq.keys())

    sorted_shape_counts = sorted(all_shape_counts)
    x = np.arange(len(sorted_shape_counts))  # x locations for shape counts

    # Create a colormap
    cmap = plt.get_cmap("tab20")
    norm = mcolors.Normalize(vmin=0, vmax=num_states - 1)
    state_color_map = {state: cmap(norm(i)) for i, state in enumerate(unique_states)}

    bar_width = 0.8 / num_states  # total width of bars per group

    plt.figure(figsize=(12, 6))

    for i, state in enumerate(unique_states):
        freq = shape_freq_dicts[state]
        y = [freq.get(shape_count, 0) for shape_count in sorted_shape_counts]
        offset = (i - num_states / 2) * bar_width + bar_width / 2
        plt.bar(x + offset, y, width=bar_width, label=state, color=state_color_map[state])

    plt.xlabel("Number of Shapes")
    plt.ylabel("Frequency")
    plt.title(f"{species}: Frequency of Shape Counts by State")
    plt.xticks(x, sorted_shape_counts)
    plt.legend(title="States", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y')

    # Save the plot
    file_name = f'{species.lower()}_{title_info.lower()}_{file_name.lower()}.png'
    file_name = file_name.replace(" ", "_").lower()
    plt.savefig(from_root(f'{file_path}/{file_name}'), bbox_inches='tight')
    print(f"Plot saved as {file_name}")
    plt.close()

'''
    The jitter plot is used to visualize the distribution of individual data points 
    along an axis by adding small random noise (jitter) to reduce overlap, making it 
    easier to see the spread and density of the data.
'''
def jitter_plot(bbox_dict, species, file_name, file_path, title_info):

    # Flatten bbox_dict into a DataFrame with state info
    data = []
    for batch_id, values in bbox_dict.items():
        state = batch_id[:2]
        for count in values:
            data.append({
                "State": state,
                file_name: count
            })

    df = pd.DataFrame(data)
    unique_states = sorted(df["State"].unique())

    plt.figure(figsize=(12, 4))
    sns.stripplot(
        data=df,
        x=file_name,
        y="State",
        hue="State",
        jitter=True,
        palette="tab20",
        size=5,
        legend=False  
    )

    plt.title(f"{species}: {title_info} {file_name}")
    plt.xlabel(file_name)
    plt.ylabel("State")
    plt.grid(True, axis='x')

    plt.tight_layout()
    out_name = f'{species.lower()}_{title_info.lower()}_{file_name.lower()}.png'.replace(" ", "_")
    plt.savefig(from_root(f'{file_path}/{out_name}'), bbox_inches='tight')
    print(f"Strip plot saved as {out_name}")
    plt.close()
