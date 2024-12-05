<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">NUTRITRACK-AI-PREDICTOR</h1></p>



##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

This repository contains a Flask-based REST API for classifying food images into one of the 101 categories from the Food-101 dataset. The API uses a fine-tuned MobileNetV3 model trained on the Food-101 dataset.

---

##  Features

#### Image Classification: Upload an image to classify it into one of the 101 food categories.
#### Pre-Trained Model: Utilizes MobileNetV3 for efficient inference.
#### CORS Support: Allows cross-origin requests for easy integration with frontend applications.
#### Error Handling: Provides meaningful error messages for invalid requests.


---

##  Project Structure

```sh
└── NutriTrack-AI-Predictor/
    ├── app.py
    ├── food101_mobilenet_v3_2.pth
    └── requirements.txt
```



---
##  Getting Started

###  Prerequisites

Before getting started with NutriTrack-AI-Predictor, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


###  Installation

Install NutriTrack-AI-Predictor using one of the following methods:

**Build from source:**

1. Clone the NutriTrack-AI-Predictor repository:
```sh
❯ git clone https://github.com/A5TA/NutriTrack-AI-Predictor
```

2. Navigate to the project directory:
```sh
❯ cd NutriTrack-AI-Predictor
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```




###  Usage
Run NutriTrack-AI-Predictor using the following command:
```sh
❯ python app.py
```


###  Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pytest
```


