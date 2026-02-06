#!/usr/bin/env python3
"""
Add a legend to the substrate classification image.
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Paths
INPUT_IMAGE = "/Users/quinnfisher/river_substrate/substrate_output/substrate_classification_colored.png"
OUTPUT_IMAGE = "/Users/quinnfisher/river_substrate/substrate_output/substrate_classification_with_legend.png"

# Legend data from class_legend.txt
LEGEND_DATA = [
    (0, "NoData", "#3366CC"),
    (1, "Fines Ripple", "#DC3912"),
    (2, "Fines Flat", "#FF9900"),
    (3, "Cobble Boulder", "#109618"),
    (4, "Hard Bottom", "#990099"),
    (5, "Wood", "#0099C6"),
    (6, "Other", "#DD4477"),
    (7, "Shadow", "#66AA00"),
    (8, "Water - NoData", "#B82E2E"),
]

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def add_legend(input_path, output_path, legend_data):
    """Add a legend to the classification image."""
    
    # Open the original image
    img = Image.open(input_path)
    img_width, img_height = img.size
    
    # Legend settings
    legend_width = 220
    item_height = 35
    padding = 15
    box_size = 25
    text_offset = 40
    
    # Calculate legend height
    legend_height = len(legend_data) * item_height + 2 * padding + 40  # +40 for title
    
    # Create new image with space for legend
    new_width = img_width + legend_width
    new_height = max(img_height, legend_height + 20)
    
    new_img = Image.new('RGB', (new_width, new_height), color=(255, 255, 255))
    
    # Paste original image
    new_img.paste(img, (0, (new_height - img_height) // 2))
    
    # Create drawing context
    draw = ImageDraw.Draw(new_img)
    
    # Try to load a nice font, fall back to default
    try:
        # Try common system fonts
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSText.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        font = None
        title_font = None
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, 16)
                title_font = ImageFont.truetype(fp, 18)
                break
        if font is None:
            font = ImageFont.load_default()
            title_font = font
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # Legend background
    legend_x = img_width + 5
    legend_y = 10
    
    # Draw legend background
    draw.rectangle(
        [legend_x, legend_y, legend_x + legend_width - 10, legend_y + legend_height],
        fill=(248, 248, 248),
        outline=(200, 200, 200),
        width=1
    )
    
    # Draw title
    title = "Substrate Classes"
    draw.text((legend_x + padding, legend_y + padding), title, fill=(0, 0, 0), font=title_font)
    
    # Draw legend items
    y_offset = legend_y + padding + 35
    
    for class_id, class_name, hex_color in legend_data:
        rgb_color = hex_to_rgb(hex_color)
        
        # Draw color box
        box_x = legend_x + padding
        box_y = y_offset
        draw.rectangle(
            [box_x, box_y, box_x + box_size, box_y + box_size],
            fill=rgb_color,
            outline=(100, 100, 100),
            width=1
        )
        
        # Draw class name
        text_x = box_x + text_offset
        text_y = box_y + 4
        draw.text((text_x, text_y), class_name, fill=(0, 0, 0), font=font)
        
        y_offset += item_height
    
    # Save the new image
    new_img.save(output_path, quality=95)
    print(f"Image with legend saved to: {output_path}")
    
    return output_path

if __name__ == '__main__':
    add_legend(INPUT_IMAGE, OUTPUT_IMAGE, LEGEND_DATA)
