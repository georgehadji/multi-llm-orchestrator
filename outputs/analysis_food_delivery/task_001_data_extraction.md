Here's a revised and complete version of the output, addressing all specific issues raised in the critique.

---

This task involves either downloading and cleaning a Zomato dataset from Kaggle or, as a fallback, scraping restaurant data from Wolt. I will prioritize the Kaggle option as specified.

The entire process will be implemented in Python, leveraging libraries such as `pandas`, `kaggle`, `requests`, and `BeautifulSoup`.

### Part 1: Zomato Dataset from Kaggle (Primary Option)

This section details the steps for downloading, loading, cleaning, and saving the Zomato dataset.

#### 1.1. Setup Kaggle API

To use the Kaggle API, you need to install the `kaggle` library and set up your authentication credentials.

**Steps:**

1.  **Install the Kaggle library:**
    ```bash
    pip install kaggle pandas
    ```
2.  **Obtain Kaggle API Token:**
    *   Go to your Kaggle account page (kaggle.com/me/account).
    *   Scroll down to the "API" section and click "Create New API Token". This will download a `kaggle.json` file.
3.  **Place `kaggle.json`:**
    *   Create a directory named `.kaggle` in your user's home directory (e.g., `C:\Users\<username>\.kaggle\` on Windows, or `~/.kaggle/` on Linux/macOS).
    *   Move the downloaded `kaggle.json` file into this `.kaggle` directory.
    *   Ensure the file permissions are set correctly (e.g., `chmod 600 ~/.kaggle/kaggle.json` on Linux/macOS) to keep your API key secure.

#### 1.2. Download the Zomato Dataset

We will use the Kaggle API to download the specified dataset.

```python
import os
import pandas as pd
import subprocess
import sys

# Define dataset details and file paths
KAGGLE_DATASET = "shrutimehta/zomato-restaurants-data"
DATA_DIR = "data"
ZOMATO_ZIP_FILE = os.path.join(DATA_DIR, "zomato-restaurants-data.zip")
ZOMATO_CSV_FILE = os.path.join(DATA_DIR, "zomato.csv") # Assuming the main CSV inside is named zomato.csv
RAW_ZOMATO_OUTPUT = os.path.join(DATA_DIR, "zomato_raw.csv")

# Create the data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Attempting to download Zomato dataset: {KAGGLE_DATASET}")
try:
    # Use subprocess to run the kaggle command
    # -d for download, -p for path, --unzip to automatically unzip
    result = subprocess.run(
        ["kaggle", "datasets", "download", KAGGLE_DATASET, "-p", DATA_DIR, "--unzip"],
        check=True, # Raise an exception for non-zero exit codes
        capture_output=True,
        text=True
    )
    print("Kaggle download successful.")
    # print("STDOUT:", result.stdout) # Uncomment for detailed output
    # print("STDERR:", result.stderr) # Uncomment for detailed output

    # Verify if the CSV file exists after unzipping
    if not os.path.exists(ZOMATO_CSV_FILE):
        print(f"