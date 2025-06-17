import os
import hydra
import logging
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from utils.utils import count_all_files
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from omegaconf import DictConfig, OmegaConf
from reportlab.pdfbase.pdfmetrics import stringWidth

log = logging.getLogger(__name__)

class PDFDrafter():
    def __init__(self, cfg: DictConfig, num_cutouts: tuple[dict[str, int], dict[str, int]]) -> None:

        self.cfg = cfg

        # Extract species and sort them alphabetically, ignore duplicates
        species_list = list(set(name.lower() for name in cfg.cutout_filters.category.common_name))
        self.sorted_species = sorted(species_list, key=lambda s: s.lower())

        # Extract num cutouts into specified and all
        self.specified_cutouts, self.all_cutouts = num_cutouts

        # Set up canvas/pdf
        self.date_time = datetime.datetime.now()
        self.output_pdf = str(f"{cfg.paths.analysisdir}/pre_synth_analysis_{self.date_time}.pdf")
        self.pdf = canvas.Canvas(self.output_pdf, pagesize=letter)

        # PAGE INFO
        self.page_info = {
            "height": letter[1],
            "width": letter[0],
            "margin": 0.5 * inch,
            "top_of_page": letter[1] - (0.5 * inch) / 2
        }
        # FONT INFO
        self.fonts = {
            "size": {
                "title": 20,
                "author": 14,
                "heading": 12,
                "time": 10,
                "body": 8
            },
            "styles": {
                "bold": "Helvetica-Bold",
                "normal": "Helvetica"
            }
        }
        # POSITION, KEEP TRACK OF WHERE WE ARE INSERTING ON THE PAGE
        self.position_state = {
            "offset_from_top_of_page": 0,
            "offset_from_left_of_page": 0
        }

        # Title
        self.title = "Pre-Synthesis Analysis"

        # Author
        self.author = "Maintainer: PSA CV Team"

    def save_pdf(self) -> None:
        self.pdf.save()
        log.info(f"PDF saved to {self.output_pdf}")

    def add_graphs_to_pdf(self) -> None:
        '''
            Below inserts metadata graphs for each species into the final report.
        '''

        # Load in the image (graph) paths
        all_cutout_dir = str(f"{self.cfg.paths.analysisdir}/all")
        downloaded_cutout_dir = str(f"{self.cfg.paths.analysisdir}/specified")
        storage_graph_dir = str(f"{self.cfg.paths.analysisdir}/storage_location")
        all_cutout_graphs = sorted([f for f in os.listdir(all_cutout_dir) if os.path.splitext(f)[1].lower() == '.png'])
        downloaded_cutout_graphs = sorted([f for f in os.listdir(downloaded_cutout_dir) if os.path.splitext(f)[1].lower() == '.png'])
        storage_graphs = sorted([f for f in os.listdir(storage_graph_dir) if os.path.splitext(f)[1].lower() == '.png'])
        num_images = min(len(all_cutout_graphs), len(downloaded_cutout_graphs)) # should be the same length

        # Calculate how many graphs each species has
        num_of_images_on_line = 0
        number_of_species = len(self.sorted_species)
        number_of_graphs = count_all_files(all_cutout_dir)
        graphs_per_species = int(number_of_graphs/number_of_species)

        current_species_index = 0
        for i in range(num_images):
        
            # Add heading for the next batch of graphs for the next species
            if i % graphs_per_species == 0:
                if current_species_index < len(self.sorted_species):
                    species = self.sorted_species[current_species_index]
                    self.wrap_text(species.title(), self.fonts["styles"]["normal"], self.fonts["size"]["author"])
            
                    # Add size statistics for the first species
                    subj = f"{species.title()} has {self.specified_cutouts[species.upper()]} total number of cutouts. Your configs specify {self.all_cutouts[species.upper()]} of those cutouts."
                    self.wrap_text(subj, self.fonts["styles"]["normal"], self.fonts["size"]["body"])

                    current_species_index += 1

            # Load next couptle of graphs to place on pdf
            all_cutout_path = os.path.join(all_cutout_dir, all_cutout_graphs[i])
            specified_cutout_path = os.path.join(downloaded_cutout_dir, downloaded_cutout_graphs[i])

            # Keep track of image per line, to know when to add line break
            num_of_images_on_line += 1
            new_line = False

            # X, Y offset logic for new lines
            if num_of_images_on_line == 1:
                self.position_state["offset_from_left_of_page"] = 0
            if num_of_images_on_line == 2:
                new_line = True      
                num_of_images_on_line = 0

            # Place first couple of images
            self.place_image(all_cutout_path, False)
            # Place second couple of images
            self.place_image(specified_cutout_path, new_line)    

            # Place storage graphs
            if i % graphs_per_species == (graphs_per_species-1):
                if (current_species_index-1<len(storage_graphs)):
                    storage_graph_paths = os.path.join(storage_graph_dir, storage_graphs[current_species_index-1])
                    self.place_image(storage_graph_paths, True, (2,2))    
                self.position_state["offset_from_left_of_page"] = 0
                num_of_images_on_line = 0


    def initialize_heading_and_description(self, heading: str, description : str) -> None:
        '''
            Below we set the title of the report.
        '''
        self.pdf.setFont(self.fonts["styles"]["bold"], self.fonts["size"]["title"])
        self.pdf.drawCentredString(self.page_info["width"] / 2, self.page_info["top_of_page"], self.title)
        self.position_state["offset_from_top_of_page"] += self.fonts["size"]["title"]

        '''
            Below we author the report.
        '''
        self.pdf.setFont(self.fonts["styles"]["normal"], self.fonts["size"]["author"])
        self.pdf.drawCentredString(self.page_info["width"] / 2, self.page_info["top_of_page"] - self.position_state["offset_from_top_of_page"] - 4, self.author)
        self.position_state["offset_from_top_of_page"] += (self.fonts["size"]["author"]+4)

        '''
            Below we add the date to the report
        '''
        date = str(datetime.date.today())
        self.pdf.setFont(self.fonts["styles"]["normal"], self.fonts["size"]["time"])
        self.pdf.drawCentredString(self.page_info["width"] / 2, self.page_info["top_of_page"] - self.position_state["offset_from_top_of_page"] - 4, date)
        self.position_state["offset_from_top_of_page"] += (self.fonts["size"]["time"]+4)

        '''
            Below we add the heading for this section of the report
        '''
        date = str(datetime.date.today())
        self.pdf.setFont(self.fonts["styles"]["normal"], self.fonts["size"]["heading"])
        self.pdf.drawCentredString(self.page_info["width"] / 2, self.page_info["top_of_page"] - self.position_state["offset_from_top_of_page"] - 4, heading)
        self.position_state["offset_from_top_of_page"] += (self.fonts["size"]["heading"]+4)


        '''
            Below is a brief description intended to give the reader context and insight into the graphs presented in this report.
        '''
        self.wrap_text(description, self.fonts["styles"]["normal"], self.fonts["size"]["body"])
        # Add spacing between description and images
        self.position_state["offset_from_top_of_page"] += 0.15 * inch  

    def place_image(self, image_path: str, new_line: bool, image_scaler_width_and_height: tuple[int, int] = (1, 1)) -> None:
        final_width = 2 * inch * image_scaler_width_and_height[0]
        img = ImageReader(image_path)
        img_width, img_height = img.getSize()
        aspect_ratio = img_height / img_width * (image_scaler_width_and_height[1]/image_scaler_width_and_height[0])

        image_margin = 0.25 * inch

        # Calculate height to preserve aspect ratio
        final_height = final_width * aspect_ratio

        # check to see if we need to move to a new page
        self.evaluate_room_on_page(final_height)

        y_pos = self.page_info["top_of_page"] - self.position_state["offset_from_top_of_page"] - final_height

        # X calculations
        x_pos = image_margin + self.position_state["offset_from_left_of_page"]
        self.position_state["offset_from_left_of_page"] += final_width

        self.pdf.drawImage(image_path, x_pos, y_pos, width=final_width, height=final_height, preserveAspectRatio=True)

        # Y calculations
        # check to see if we need to add a new line
        if (new_line):
            self.position_state["offset_from_top_of_page"] += final_height + 0.25 * inch
        
    def wrap_text(self, sentances: str, font_style: str, font_size: int) -> None:
        """
            This function allows you to pass in any string, font style, and font size and it will 
            ensure that it fits properly on the page. The function forces center aligned.
        """
        words = sentances.split()
        lines = []
        line = ""
        max_width = self.page_info["width"] - 2 * self.page_info["margin"]
        self.pdf.setFont(font_style, font_size)

        # Wrap text
        for word in words:
            test_line = f"{line} {word}".strip()
            if stringWidth(test_line, font_style, font_size) <= max_width:
                line = test_line
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)

        # Draw each line centered
        line_spacing = 4
        for i, line in enumerate(lines):
            y_pos = self.page_info["top_of_page"] - self.position_state["offset_from_top_of_page"] - 4 - i * (font_size + line_spacing)
            self.pdf.drawCentredString(self.page_info["width"] / 2, y_pos, line)
        
        final_height = len(lines) * (font_size + line_spacing)

        # check to see if we need to move to a new page
        self.evaluate_room_on_page(final_height)

        self.position_state["offset_from_top_of_page"] += final_height

    def evaluate_room_on_page(self, height: float) -> None:
        """
            The function will determine if an inserition can fit on a page based on its height. If it cannot it will 
            move the pdf to the next page and reset the position state to the top of the page.
        """
        if (self.page_info["height"]-(self.position_state["offset_from_top_of_page"] + height)) <= self.page_info["margin"]:
            self.pdf.showPage()
            self.position_state["offset_from_top_of_page"] = 0

# for debugging to avoid going through whole pipeline
# run pipeline once to get graphs then run python3 src/utils/pdf.py, remember to fix the utils pathing
@hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    all_cutouts = {}
    specified_cutouts = {}
    species_list = list(set(name.lower() for name in cfg.cutout_filters.category.common_name))
    for species in species_list:
        species = species.upper()
        all_cutouts[species] = (0,0)
        specified_cutouts[species] = (0,0)
    num_cutouts = specified_cutouts, all_cutouts

    cfg = OmegaConf.create(cfg)
    PDFDrafter(cfg, num_cutouts)

if __name__ == "__main__":
    main()