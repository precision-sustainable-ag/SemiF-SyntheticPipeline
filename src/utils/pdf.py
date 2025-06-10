import os
import hydra
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from utils.utils import count_all_files
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from omegaconf import DictConfig, OmegaConf
from reportlab.pdfbase.pdfmetrics import stringWidth

def generate_pdf(cfg, species_list):

    output_pdf = str(f"{cfg.paths.analysisdir}/pre_synth_analysis.pdf")
    c = canvas.Canvas(output_pdf, pagesize=letter)

    # PAGE INFO
    page_width, page_height = letter
    margin = 0.5 * inch
    page_info = {
        "height": page_height,
        "width": page_width,
        "margin": margin,
        "title_y": page_height - margin / 2
    }

    # FONT INFO
    fonts = {
        "title_size": 16,
        "name_font_size": 12,
        "report_info_size": 8
    }

    # POSITION, KEEP TRACK OF WHERE WE ARE INSERTING ON THE PAGE
    position_state = {
        "distance_from_top_of_page": 0,
        "x_offset": 0
    }


    '''
        Below we set the title of the report.
    '''
    title = "Pre-Synthesis Analysis"
    title_font = "Helvetica-Bold"
    c.setFont(title_font, fonts["title_size"])
    c.drawCentredString(page_width / 2, page_info["title_y"], title)
    position_state["distance_from_top_of_page"] += fonts["title_size"]


    '''
        Below we author the report.
    '''
    name = "PSA CV Team"
    c.setFont("Helvetica", fonts["name_font_size"])
    c.drawCentredString(page_width / 2, page_info["title_y"] - position_state["distance_from_top_of_page"] - 4, f"Maintainer: {name}")
    position_state["distance_from_top_of_page"] += (fonts["name_font_size"]+4)


    '''
        Below is a brief description intended to give the reader context and insight into the graphs presented in this report.
    '''
    sorted_species = sorted(species_list, key=lambda s: s.lower())
    if len(species_list) > 1:
        titled = [s.title() for s in sorted_species]
        species_str = ', '.join(titled[:-1]) + f", and {titled[-1]}"
    else:
        species_str = species_list[0].title()
    subj = f"The following is a report of the {species_str} in the database. The aim is to display the metadata of all cutouts vs cutouts you specified in your configuration"
    wrap_body_text(c, subj, position_state, page_info, fonts["report_info_size"])
    # Add spacing between description and images
    position_state["distance_from_top_of_page"] += 0.15 * inch 


    '''
        Below inserts metadata graphs for each species into the final report.
    '''

    # Load in the image (graph) paths
    all_cutout_dir = str(f"{cfg.paths.analysisdir}/all")
    downloaded_cutout_dir = str(f"{cfg.paths.analysisdir}/specified")
    all_cutout_graphs = sorted([f for f in os.listdir(all_cutout_dir) if os.path.splitext(f)[1].lower() == '.png'])
    downloaded_cutout_graphs = sorted([f for f in os.listdir(downloaded_cutout_dir) if os.path.splitext(f)[1].lower() == '.png'])
    num_images = min(len(all_cutout_graphs), len(downloaded_cutout_graphs)) # should be the same length

    # Calculate how many graphs each species has
    num_of_images_on_line = 0
    number_of_species = len(species_list)
    number_of_graphs = count_all_files(all_cutout_dir)
    graphs_per_species = int(number_of_graphs/number_of_species)

    # Add the first species heading
    current_species_index = 0
    subj = sorted_species[current_species_index]
    current_species_index += 1
    wrap_body_text(c, subj.title(), position_state, page_info, fonts["name_font_size"])


    for i in range(num_images):
        
        # Load next couptle of graphs to place on pdf
        all_cutout_path = os.path.join(all_cutout_dir, all_cutout_graphs[i])
        specified_cutout_path = os.path.join(downloaded_cutout_dir, downloaded_cutout_graphs[i])

        # Keep track of image per line, to know when to add line break
        num_of_images_on_line += 1
        new_line = False

        # X, Y offset logic for new lines
        if num_of_images_on_line == 1:
            position_state["x_offset"] = 0
        if num_of_images_on_line == 2:
            new_line = True      
            num_of_images_on_line = 0
        if i != 0 and i % graphs_per_species == (graphs_per_species-1):
            position_state["x_offset"] = 0
            new_line = True
            num_of_images_on_line = 0

        # Update flags for first image
        flags = {
            "new_line": False,
            "center": False
        }
        place_image(all_cutout_path, c, page_info, position_state, flags)

        # Update flags for second image
        flags = {
            "new_line": new_line,
            "center": False
        }
        place_image(specified_cutout_path, c, page_info, position_state, flags)    

        # Add heading for the next batch of graphs for the next species
        if i % graphs_per_species == (graphs_per_species-1):
            if current_species_index < len(sorted_species):
                subj = sorted_species[current_species_index]
                current_species_index += 1
                wrap_body_text(c, subj.title(), position_state, page_info, fonts["name_font_size"])

    # Place final image (storage distriubtion). Do this seperately as its a unique graph
    final_image = str(f"{cfg.paths.analysisdir}/Cutout Distribution Across Storages.png")
    if os.path.exists(final_image):
        # Center the last image
        flags = {
            "new_line": False,
            "center": True
        }
        place_image(final_image, c, page_info, position_state, flags, 3)

    c.save()
    log.info(f"PDF saved to {output_pdf}")

def place_image(final_image, c, page_info, position_state, flags, scaler=1):
    final_width = 2 * inch * scaler
    img = ImageReader(final_image)
    img_width, img_height = img.getSize()
    aspect_ratio = img_height / img_width

    image_margin = 0.25 * inch

    # Calculate height to preserve aspect ratio
    final_height = final_width * aspect_ratio

    # check to see if we need to move to a new page
    if (page_info["height"]-(position_state["distance_from_top_of_page"] + final_height)) <= 0:
        c.showPage()
        position_state["distance_from_top_of_page"] = 0

    y_pos = page_info["title_y"] - position_state["distance_from_top_of_page"] - final_height

    # X calculations
    x_pos = image_margin + position_state["x_offset"]
    position_state["x_offset"] += final_width
    if flags.get("center", False):
        x_pos = (page_info["width"] - final_width)/2

    c.drawImage(final_image, x_pos, y_pos, width=final_width, height=final_height, preserveAspectRatio=True)

    # Y calculations
    # check to see if we need to add a new line
    if (flags.get("new_line", False)):
        position_state["distance_from_top_of_page"] += final_height + 0.25 * inch

def wrap_body_text(c, subj, position_state, page_info, font_size):
    words = subj.split()
    lines = []
    line = ""
    max_width = page_info["width"] - 2 * page_info["margin"]
    font = "Helvetica"
    c.setFont(font, font_size)

    # Wrap text
    for word in words:
        test_line = f"{line} {word}".strip()
        if stringWidth(test_line, font, font_size) <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    # Draw each line centered
    line_spacing = 4
    for i, line in enumerate(lines):
        y_pos = page_info["title_y"] - position_state["distance_from_top_of_page"] - 4 - i * (font_size + line_spacing)
        c.drawCentredString(page_info["width"] / 2, y_pos, line)

    position_state["distance_from_top_of_page"] += len(lines) * (font_size + line_spacing)


@hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    generate_pdf(cfg, cfg.cutout_filters.category.common_name)

if __name__ == "__main__":
    main()