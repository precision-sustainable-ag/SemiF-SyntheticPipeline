import os
import hydra
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
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
    available_height = page_height - 2 * margin

    # TEXT PLACEMENT INFO
    title_x = page_width / 2
    title_y = page_height - margin / 2
    distance_from_title_y = 0 # keep track of how far we are away from the title

    # FONT SIZES
    title_size = 16
    name_font_size = 12
    report_info_size = 8

    '''
        Below we set the title of the report.
    '''
    title = "Pre-Synthesis Analysis"
    title_font = "Helvetica-Bold"
    c.setFont(title_font, title_size)
    c.drawCentredString(title_x, title_y, title)
    distance_from_title_y += title_size

    '''
        Below we author the report.
    '''
    name = "PSA CV Team"
    c.setFont("Helvetica", name_font_size)
    c.drawCentredString(title_x, title_y - distance_from_title_y - 4, f"Maintainer: {name}")
    distance_from_title_y += (name_font_size+4)

    '''
        Below is a brief description intended to give the reader context and insight into the graphs presented in this report.
    '''
    if len(species_list) > 1:
        species_str = ', '.join(species_list[:-1]) + f", and {species_list[-1]}"
    else:
        species_str = species_list[0]
    subj = f"The following is a report of the {species_str} in the database. The aim is to display the metadata of all cutouts vs cutouts you specified in your configuration"

    words = subj.split()
    lines = []
    line = ""
    max_width = page_width - 2 * margin
    font = "Helvetica"
    size = report_info_size
    c.setFont(font, size)

    # Wrap text
    for word in words:
        test_line = f"{line} {word}".strip()
        if stringWidth(test_line, font, size) <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    # Draw each line centered
    line_spacing = 4
    for i, line in enumerate(lines):
        y_pos = title_y - distance_from_title_y - 4 - i * (size + line_spacing)
        c.drawCentredString(title_x, y_pos, line)

    distance_from_title_y += len(lines) * (size + line_spacing)

    # Add spacing between description and images
    distance_from_title_y += 20 

    '''
        Below inserts metadata graphs for each species into the final report.
    '''
    all_cutout_dir = str(f"{cfg.paths.analysisdir}/all")
    downloaded_cutout_dir = str(f"{cfg.paths.analysisdir}/specified")
    extensions = {'.png', '.jpg', '.jpeg'}
    all_cutout_graphs = sorted([f for f in os.listdir(all_cutout_dir) if os.path.splitext(f)[1].lower() in extensions])
    downloaded_cutout_graphs = sorted([f for f in os.listdir(downloaded_cutout_dir) if os.path.splitext(f)[1].lower() in extensions])
    num_images = min(len(all_cutout_graphs), len(downloaded_cutout_graphs)) # should be the same length

    x_offset = 0
    for i in range(num_images):

        all_cutout_path = os.path.join(all_cutout_dir, all_cutout_graphs[i])
        specified_cutout_path = os.path.join(downloaded_cutout_dir, downloaded_cutout_graphs[i])

        page_down = False
        if i != 0 and i % 2 == 0:
            x_offset = 0
            page_down = True

        last_image = False
        # check if we are on the last image, if so add y offset for storage picture
        if i == (num_images-1):
            last_image = True

        distance_from_title_y, x_offset = place_image(all_cutout_path, c, page_height, margin, title_y, distance_from_title_y, page_down, False, x_offset)
        distance_from_title_y, x_offset = place_image(specified_cutout_path, c, page_height, margin, title_y, distance_from_title_y, False, last_image, x_offset)
    
    # Load image path
    final_image = str(f"{cfg.paths.analysisdir}/Cutout Distribution Across Storages.png")
    if os.path.exists(final_image):
        place_image(final_image, c, page_height, margin, title_y, distance_from_title_y, 0, 0, 0, 3)

    c.save()
    print(f"PDF saved to {output_pdf}")


def place_image(final_image, c, page_height, margin, title_y, distance_from_title_y, page_down, last_image, x_offset, scaler=1):
    final_width = 2 * inch * scaler
    img = ImageReader(final_image)
    img_width, img_height = (img.getSize())
    aspect_ratio = img_height / img_width

    image_margin = 0.25 * inch

    # Calculate height to preserve aspect ratio
    final_height = final_width * aspect_ratio

    # Check if there's enough vertical space, else start a new page
    if (page_height-(distance_from_title_y + final_height)) < 0:
        c.showPage()
        distance_from_title_y = margin

    # Y calculations
    if (page_down):
        distance_from_title_y += final_height + 0.25 * inch
    y_pos = title_y - distance_from_title_y - final_height
    if (last_image):
        distance_from_title_y += final_height + 0.25 * inch

    # X calculations
    x_pos = image_margin + x_offset
    x_offset += final_width

    c.drawImage(final_image, x_pos, y_pos, width=final_width, height=final_height, preserveAspectRatio=True)

    return distance_from_title_y, x_offset


@hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)

    generate_pdf(cfg, cfg.cutout_filters.category.common_name)

if __name__ == "__main__":
    main()
