# XrdQty

## Tool to quantify crystalline Phases via X-Ray Diffraction Measurements

**XrdQty** is a lightweight Python tool that synthesizes X-ray diffraction (XRD) patterns for use in training convolutional neural networks. By generating large, labeled datasets, it enables automated quantification of mineral phases. A critical step for assessing the purity and performance of natural materials that often contain multiple co-existing phases.

## Table of contents

* Description and background
* Installation
* Getting Started
* Usage

## Description and background

Natural minerals, especially clay, are typically poorly crystalline. This low crystallinity, along with crystalline defects and amorphous fractions, diminishes and distorts diffraction peaks, making conventional Rietveld refinement cumbersone and unreliable for phase quantification. **XrdQty** circumvents these difficulties by producing synthetics patterns. The tool demonstrated on **Hectorite**, a smectite clay that can accumulate significant lithium concentrations, positioning it as a potential lithium resource. Because experimental training data are scarce, **XrdQty** derives its synthetic patterns from crystal-lattice parameters and applies appropriate distribution functions to model peak shapes and intensities. The resulting datasets have proven effective for quality-control workflows, delivering accurate phase quantification even in complex, low-crystallinity samples.

## Installation

### Prerequisites

* python 3.12.11
* pip (Python package manager)
 
### Clone the repo:
```
git clone https://github.com/mschindler779/XrdQty.git
cd XrdQty
```

### Create a virtual environment (optional but recommended)
```
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```

### Install dependencies
```
python -m pip install -r requirements.txt
```

## Getting Started

### Preparing Your Structure Folder
Add new minerals by adding a CSV file to the structure folder. Each CSV must list **Reflex Position** (2 Theta in degrees) and the relative **Peak Intensities**. **Important** The maximum intensity must be 100. This normalizes all peaks and keeps the PDF (Probability Distribution Function) generation stable.

### maxIntensity.csv
This file lists the **maximum Reference Intensity** for each mineral. It is used to normalise the synthetic patterns and to build the probability distribution. Furthermore, the **minimum Allowed Fraction** for each mineral is stored in this file. For e.g., the major phase Hectorite clay could start at 40 %, while impurities as Quartz or Hematite start at 0 percent.  

## Usage

### Import of XrdQty in python
```
import XrdQty
```

### Initiate XrdQty
Tool initializing as:
```
xrd_qty = XrdQty(start_angle = 10, stop_angle = 90, angle_steps = 8501, "model_name")
```

### Create Training Data
By default, the program generates 500 synthetic powder X‑ray diffraction patterns. If you need a different number, simply edit the `XrdQty.py` file. The diffraction intensities are written to `features.csv`, while the associated mineral compositions are stored in `label.csv`.
```
xrd_qty.create_training_data()
```

### CNN Model Training
You can now start training the convolutional neural network; the trained model will be saved as **"model_name.pth"**.
```
xrd_qty.model_training()
```

### Loading Existing Model
If the model has just been prepared, the next step can be omitted.
```
xrd_qty.load_model()
```

### Predicting Phase Quantity
The X‑ray diffraction powder pattern of interest can be analyzed to predict mineral quantities as follows.
```
xrd_qty.predict("XRD Data.csv")
```

## File Structure

├── XrdQty.py **Main application**<br/>
├── structure **Folder for Structure Data**<br/>
├── maxIntensity.csv **Reference Peak Intensities**<br/>
├── requirements.txt **Python dependencies**<br/>
├── README.md **This file**<br/>
└── LICENSE **MIT license**

## License

This project is licensed under the MIT License - see the LICENSE file for details

© 2025 Markus Schindler
