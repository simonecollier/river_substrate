# river_substrate

This repository contains the code for fine-tuning the PINGMapper model to segment and classify custom substrate classes from side-scan sonar data of river bottoms.

## Scott

* The python script for spliting tiff files into jpeg tiles for labelling is [`tile_geotiffs_to_jpegs.py`](tile_geotiffs_to_jpegs.py).
* The colour mapping that I found associated with the tiff files from Grand River when I opened them in GIS is stored the file [`sonar_colourmap.clr`](sonar_colourmap.clr). This can be used to colour the jpeg tiles if this makes it easier for labelling compared to the greyscale images.
