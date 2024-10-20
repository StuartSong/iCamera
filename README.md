
# Clarity Sorting and Server for Medical Record Scanning

## Overview

This project is developed to automate the scanning of patient paper medical records and upload them to a server. It includes two main components:

1. **Clarity Sorting**: A Python script (`Clarity Sorting.py`) that captures images from a scanner, evaluates their clarity, and ensures only the best-quality images are retained.
2. **Server**: A Pyramid-based server that handles the image uploads and stores them for later retrieval.

## Table of Contents

- [Clarity Sorting.py](#clarity-sortingpy)
  - [Features](#features)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
- [Server Code](#server-code)
  - [Endpoints](#endpoints)
  - [Usage](#usage-1)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

---

## Clarity Sorting.py

### Features

- Captures images from a scanner using a connected camera.
- Measures image sharpness using the variance of the Laplacian method.
- Automatically sorts and removes blurry images to ensure only high-quality scans are retained.
- Compares images to avoid storing duplicate or low-quality scans.
- Uploads the final set of clear images to a server.

### Dependencies

Ensure the following Python packages are installed:

- `numpy`
- `opencv-python`
- `matplotlib`
- `argparse`
- `requests`
- `shutil`
- `glob`

To install them:

```bash
pip install numpy opencv-python matplotlib requests
```

### Usage

1. Connect the camera or scanner.
2. Adjust the resolution settings as needed in the `cap.set()` function.
3. Run the script to start capturing and evaluating the images.

```bash
python ClaritySorting.py
```

Images will be stored in the `Patient/` folder if deemed clear enough. Blurry or duplicate images will be discarded automatically.

---

## Server Code

The server is built using the Pyramid framework and uses the `cornice` package for handling image uploads. It allows authenticated users to upload medical record images via HTTP POST requests and stores them on the server.

### Endpoints

- **POST /imagecap/{username}**: Uploads an image file and stores it on the server under the `Patient/` directory.
- **GET /imagecap/{username}**: Retrieves information about the uploaded images.

### Usage

To run the server, ensure you have Pyramid installed along with `cornice`:

```bash
pip install pyramid cornice
```

Run the server using the following command:

```bash
pserve development.ini
```

To upload an image via the `POST` endpoint, use a REST client like `curl` or `Postman`. Example `curl` command:

```bash
curl -X POST -F 'realfile=@path/to/image.jpg' -F 'realname=image.jpg' http://localhost:6543/imagecap/username
```

---
