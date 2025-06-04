import os
from from_root import from_root
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

def generate_pdf():
    all_cutout_dir = str(from_root('analyze_images/cutouts/all_cutouts'))
    downloaded_cutout_dir = str(from_root('analyze_images/cutouts/downloaded_cutouts'))
    output_pdf = 'pre_synth_analysis.pdf'

    # Accepted image extensions
    extensions = {'.png', '.jpg', '.jpeg'}

    # Sort image files
    all_cutout_graphs = sorted([f for f in os.listdir(all_cutout_dir) if os.path.splitext(f)[1].lower() in extensions])
    downloaded_cutout_graphs = sorted([f for f in os.listdir(downloaded_cutout_dir) if os.path.splitext(f)[1].lower() in extensions])
    num_images = min(len(all_cutout_graphs), len(downloaded_cutout_graphs))

    # Start PDF
    c = canvas.Canvas(output_pdf, pagesize=letter)
    page_width, page_height = letter
    margin = 0.5 * inch
    image_width = (page_width - 3 * margin) / 2  # two images + spacing
    image_height = 3 * inch  # Adjust depending on aspect ratio

    y_start = page_height - margin

    for i in range(min(num_images, 10)):
        orig_path = os.path.join(all_cutout_dir, all_cutout_graphs[i])
        exg_path = os.path.join(downloaded_cutout_dir, downloaded_cutout_graphs[i])

        if y_start - image_height < margin:
            c.showPage()
            y_start = page_height - margin

        # Draw original image
        c.drawImage(orig_path, margin, y_start - image_height, width=image_width, height=image_height, preserveAspectRatio=True)
        c.drawString(margin, y_start - image_height - 12, f"")

        # Draw EXG image next to it
        c.drawImage(exg_path, margin + image_width + margin, y_start - image_height, width=image_width, height=image_height, preserveAspectRatio=True)
        c.drawString(margin + image_width + margin, y_start - image_height - 12, f"")

        # Move down for next row
        y_start -= (image_height + 40)

    final_image = str(from_root('analyze_images/cutouts/storage_distribution.png'))
    if os.path.exists(final_image):
        c.showPage()
        final_width = page_width - 2 * margin
        final_height = 5 * inch  # Adjust as needed

        # Calculate position to center it
        x_pos = (page_width - final_width) / 2
        y_pos = (page_height - final_height) / 2

        c.drawImage(final_image, x_pos, y_pos, width=final_width, height=final_height, preserveAspectRatio=True)
        c.drawCentredString(page_width / 2, y_pos - 20, "")

    c.save()
    print(f"PDF saved to {output_pdf}")
