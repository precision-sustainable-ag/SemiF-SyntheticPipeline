import os
import math
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

log = logging.getLogger(__name__)

# use this function to keep axis ticks in check
def thousands_formatter(x: float, pos: int) -> str:
    return f'{int(x / 1000)}k'

def horizontal_bar_chart_plot(
    shape_count_dict: dict[str, list[int | None]],
    species: str,
    file_name: str,
    file_path: str,
    title_info: str,
    palette: dict[str, str],
    logrithmic: bool
) -> None:
    max_bars=10

    # Group shape counts by state (first 2 characters of batch ID)
    grouped_data = {}
    for batch_id, shape_counts in shape_count_dict.items():
        state = batch_id[:2]
        if state not in grouped_data:
            grouped_data[state] = []
        grouped_data[state].extend(shape_counts)

    unique_states = sorted(grouped_data.keys())
    num_states = len(unique_states)

    # Collect all unique shape counts
    all_shape_counts = set()
    shape_freq_dicts = {}

    for state, shape_counts in grouped_data.items():
        filtered_counts = [s for s in shape_counts if s is not None]
        freq = Counter(filtered_counts)
        shape_freq_dicts[state] = freq
        all_shape_counts.update(freq.keys())

    sorted_shape_counts = sorted(all_shape_counts)

    # Binning logic if too many unique shape counts
    if len(sorted_shape_counts) > max_bars:
        min_val, max_val = min(sorted_shape_counts), max(sorted_shape_counts)
        bin_width = math.ceil((max_val - min_val + 1) / max_bars)
        bins = [(i, i + bin_width - 1) for i in range(min_val, max_val + 1, bin_width)]

        # Generate bin labels
        bin_labels = [f"{start}-{end}" if start != end else f"{start}" for start, end in bins]
        x = np.arange(len(bin_labels))

        # Compute binned frequencies
        binned_freqs = {}
        for state in unique_states:
            freq = shape_freq_dicts[state]
            binned_freq = [0] * len(bins)
            for i, (start, end) in enumerate(bins):
                binned_freq[i] = sum(count for val, count in freq.items() if start <= val <= end)
            binned_freqs[state] = binned_freq
    else:
        # Use raw shape counts
        bin_labels = list(map(str, sorted_shape_counts))
        x = np.arange(len(sorted_shape_counts))
        binned_freqs = {}
        for state in unique_states:
            freq = shape_freq_dicts[state]
            binned_freqs[state] = [freq.get(sc, 0) for sc in sorted_shape_counts]

    # Bar chart plotting
    if num_states:
        bar_width = 0.8 / num_states
    plt.figure(figsize=(6, 6))

    for i, state in enumerate(unique_states):
        y = binned_freqs[state]
        offset = (i - num_states / 2) * bar_width + bar_width / 2
        plt.bar(x + offset, y, width=bar_width, label=state, color=palette[state])

    if logrithmic: 
        plt.yscale('log')

    plt.title(f"{title_info} {species}".title(), fontsize=25)
    plt.xlabel(file_name, fontsize=20)
    plt.xticks(x, bin_labels, rotation=45, fontsize=15)  
    plt.ylabel("Frequency", fontsize=20)
    plt.yticks(fontsize=15)
    plt.legend(title="States", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, title_fontsize=15)
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the plot
    save_plot(species, file_name, file_path, title_info)

def boolean_horizontal_bar_chart_plot(
    boolean_count_dict: dict[str, list[bool | None]],
    species: str,
    file_name: str,
    file_path: str,
    title_info: str,
    palette: dict[str, str],
    logrithmic: bool = False
) -> None:
    # Group True/False values by state (first 2 letters of batch ID)
    grouped_data = {}
    for batch_id, bool_list in boolean_count_dict.items():
        state = batch_id[:2]
        if state not in grouped_data:
            grouped_data[state] = []
        grouped_data[state].extend([b for b in bool_list if b in (True, False)])  # filter valid bools only

    unique_states = sorted(grouped_data.keys())
    num_states = len(unique_states)

    # Define x-axis categories and bar locations
    x_labels = ['False', 'True']
    x = np.arange(len(x_labels))  # [0, 1]

    # Count True/False per state
    bool_freqs = {}
    for state in unique_states:
        freq = Counter(grouped_data[state])
        bool_freqs[state] = [freq.get(False, 0), freq.get(True, 0)]

    # Plotting
    if num_states:
        bar_width = 0.8 / num_states
    plt.figure(figsize=(6, 6))

    for i, state in enumerate(unique_states):
        y = bool_freqs[state]
        offset = (i - num_states / 2) * bar_width + bar_width / 2
        plt.bar(x + offset, y, width=bar_width, label=state, color=palette[state])

    if logrithmic:
        plt.yscale('log')

    plt.title(f"{title_info} {species}".title(), fontsize=25)
    plt.xlabel(file_name, fontsize=20)
    plt.xticks(x, x_labels, fontsize=15)
    plt.ylabel("Frequency", fontsize=20)
    plt.yticks(fontsize=15)
    plt.legend(title="States", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, title_fontsize=15)
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the plot
    save_plot(species, file_name, file_path, title_info)

'''
    The jitter plot is used to visualize the distribution of individual data points 
    along an axis by adding small random noise (jitter) to reduce overlap, making it 
    easier to see the spread and density of the data.
'''
def jitter_plot(
    meta_data_dict: dict[str, list[int | float | None]],
    species: str,
    file_name: str,
    file_path: str,
    title_info: str,
    palette: dict[str, str],
) -> None:
    # Flatten meta_data_dict into a DataFrame with state info
    data = []
    for batch_id, values in meta_data_dict.items():
        state = batch_id[:2]
        for count in values:
            data.append({
                "State": state,
                file_name: count
            })

    df = pd.DataFrame(data)

    # Columns expected by stripplot
    expected_cols = [file_name, "State"]

    if df.empty or any(col not in df.columns for col in expected_cols):
        # Create empty DataFrame with needed columns
        df = pd.DataFrame(columns=expected_cols)
        ordered_states = []
    else:
        if "State" in df.columns and not df["State"].empty:
            ordered_states = sorted(df["State"].unique())
        else:
            ordered_states = []


    plt.figure(figsize=(6, 6))

    sns.stripplot(
        data=df,
        x=file_name,
        y="State",
        hue="State",
        order=ordered_states,
        jitter=True,
        palette=palette,
        size=5,
        legend=False  
    )

    plt.title(f"{title_info} {species}".title(), fontsize=25)
    plt.xlabel(file_name, fontsize=20)
    plt.xticks(fontsize=15)  
    plt.ylabel("State", fontsize=20)
    plt.yticks(fontsize=15)  
    plt.grid(True, axis='x')

    # Reformat ticks if x axis is in 1000's
    # do this to avoid overflow in the x axis
    xticks = plt.gca().get_xticks()
    if len(xticks) > 1 and int(xticks[len(xticks)-1]) >= 10000:
        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    save_plot(species, file_name, file_path, title_info)

def vertical_bar_chart_plot(
    data_dict: dict[str, int],
    file_name: str,
    file_path: str,
    storage_location_path: str,
    species: str
) -> None:
    labels = []
    for key in data_dict.keys():
        label_type, path = key.split(":", 1)
        path = path.strip()
        last_folder = os.path.basename(path)
        labels.append(f"{label_type.strip()}: {last_folder}")

    sizes = list(data_dict.values())
    df = pd.DataFrame({
        "Path": labels,
        "Image Count": sizes
    })

    # Use different pallette than states to avoid confusion between states and storages
    # Only three storages so not worried about repeats
    palette = [
        "#8da0cb",  # soft blue
        "#fc8d62",  # warm coral
        "#66c2a5",  # soft green-teal
    ]

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Image Count", y="Path", hue="Path", dodge=False, palette=palette)
    plt.title(f"Specified Cutout Paths For {species.title()}", fontsize=25)
    plt.xlabel("Image Count", fontsize=20)
    plt.ylabel("Storage Path", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()

    save_plot(species, file_name, file_path, storage_location_path)

def save_plot(
    species: str,
    file_name: str,
    file_path: str,
    title_info: str
) -> None:
    # Save plot
    plt.tight_layout()
    if species:
        file_name = f'{species}_{file_name}.png'
    file_name = file_name.replace(" ", "_").lower()
    if title_info:
        plt.savefig(f'{file_path}/{title_info.lower()}/{file_name}', bbox_inches='tight')
    log.info(f"Plot saved as {file_path}/{title_info.lower()}/{file_name}")
    plt.close()