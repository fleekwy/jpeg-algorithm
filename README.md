# JPEG Compressor — Baseline JPEG Algorithm Implementation

This project implements a **Baseline JPEG compression algorithm** entirely from scratch in Python.  
It includes **standard quantization and Huffman tables**, full **RGB → YCbCr color conversion**, **chroma subsampling (4:2:0)**, **DCT**, **quantization**, **Huffman encoding**, and **JPEG file generation** — following the official JPEG specification.

---

## ✨ Features

- Full implementation of the **baseline JPEG pipeline**
- Uses **standard quantization** and **Huffman tables**
- Supports input in any **PIL-compatible format** (PNG, BMP, JPEG, etc.)
- Includes **chroma subsampling 4:2:0**
- Automatically generates a **logging.yaml** file for configurable logging
- Organized output and automatic creation of **log folders**
- Adjustable **compression quality** (1–100)
- Includes a **sample image** and **example usage** in `main.py`

---

## 🛠️ Installation & Setup

Follow these steps to set up and run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/fleekwy/jpeg-algorithm.git
cd jpeg-compressor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # on Linux/Mac
.venv\Scripts\activate           # on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file and copy the contents from `.env.example`:

```bash
cp .env.example .env            # on Linux/Mac
copy .env.example .env          # on Windows
```

The `.env` file contains configuration for logging control.

---

## 🧾 Logging Configuration

The first time you run the project, it will automatically generate a **`logging.yaml`** configuration file.  
You can customize log outputs separately for **console** and **file**, for example:

- `INFO` level for the console  
- `DEBUG` level for the file output  

### Log File Behavior

Logging behavior is controlled by the environment variable in `.env`:

| Variable | Behavior |
|-----------|-----------|
| `DISABLE_FILE_LOGGING=0` | Logs are written to both **console and file** |
| `DISABLE_FILE_LOGGING=1` | Logs are shown **only in console** (no files created) |

When file logging is enabled (`0`), logs are organized as follows:

```
logs/
└── logs_YYYY-MM-DD/
    ├── debug_log_YYYY-MM-DD_0.txt
    ├── debug_log_YYYY-MM-DD_1.txt
    └── ...
```

This structure is created automatically based on the current date.

---

## 📁 Project Structure

```
.
├── data/
│   ├── test_image.png             # Original RGB test image
│   ├── ycbcr_image.png            # Converted YCbCr version
│   ├── compressed_image_q75.jpg   # Result of JPEG compression with quality=75
│   ├── compressed_image_q20.jpg   # Result of JPEG compression with quality=20
│   └── ...
│
├── main.py                        # Example usage of the compressor
├── jpeg_compressor.py             # Core implementation of the JPEG algorithm
├── requirements.txt
├── .env
│── .gitignore
│── logging.yaml
└── README.md
```

---

## 🚀 Usage Example

Here’s a minimal example (as in `main.py`):

```python
from jpeg_compressor import JpegCompressor

if __name__ == "__main__":
    compressor = JpegCompressor()
    compressor.compress("data/test_image.png", "compressed_output.jpg", quality=85)
```

This will:
1. Load the input image  
2. Apply full JPEG compression  
3. Save the result to `data/compressed_output.jpg`  
4. Log all processing steps  

---
