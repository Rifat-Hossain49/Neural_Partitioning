import os
import urllib.request
import zipfile
import subprocess
import sys

def print_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 100 / total_size
        sys.stdout.write(f"\r{percent:5.1f}% [{read_so_far} / {total_size} bytes]")
        sys.stdout.flush()
        if read_so_far >= total_size:
            sys.stdout.write("\n")
    else:
        sys.stdout.write(f"\rRead {read_so_far} bytes")
        sys.stdout.flush()

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=print_progress)
        print("Download complete.")
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        sys.exit(1)

def main():
    base_dir = "datasets"
    os.makedirs(base_dir, exist_ok=True)

    # 1. Arizona Dataset (from Geofabrik)
    arizona_dir = os.path.join(base_dir, "arizona_osm")
    os.makedirs(arizona_dir, exist_ok=True)
    arizona_url = "https://download.geofabrik.de/north-america/us/arizona-latest-free.shp.zip"
    arizona_zip_path = os.path.join(arizona_dir, "arizona-latest-free.shp.zip")
    if not os.path.exists(arizona_zip_path):
        download_file(arizona_url, arizona_zip_path)
    else:
        print(f"Arizona dataset already exists at {arizona_zip_path}")
    
    # Optional: Extract Arizona zip if needed
    # with zipfile.ZipFile(arizona_zip_path, 'r') as zip_ref:
    #     zip_ref.extractall(arizona_dir)

    # 2. Twitter Geospatial Data (from UCI)
    twitter_dir = os.path.join(base_dir, "twitter_geospatial")
    os.makedirs(twitter_dir, exist_ok=True)
    twitter_url = "https://archive.ics.uci.edu/static/public/1050/twitter+geospatial+data.zip"
    twitter_zip_path = os.path.join(twitter_dir, "twitter_geospatial_data.zip")
    if not os.path.exists(twitter_zip_path):
        download_file(twitter_url, twitter_zip_path)
    else:
        print(f"Twitter dataset already exists at {twitter_zip_path}")

    # 3. Chicago Crimes 2001 to Present (from Kaggle)
    chicago_crimes_dir = os.path.join(base_dir, "chicago_crimes")
    os.makedirs(chicago_crimes_dir, exist_ok=True)
    
    kaggle_dataset = "utkarshx27/crimes-2001-to-present"
    print(f"Downloading Kaggle dataset '{kaggle_dataset}' to {chicago_crimes_dir}...")
    try:
        # Run kaggle cli command
        subprocess.run(["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", chicago_crimes_dir], check=True)
        print("Kaggle download complete.")
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please ensure the kaggle CLI is installed ('pip install kaggle').")
        print("Also ensure you have set up your kaggle.json API key properly.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading Kaggle dataset: {e}")
        print("Please ensure your Kaggle API key (kaggle.json) is configured correctly.")

    print("\nAll dataset downloads finished.")

if __name__ == "__main__":
    main()
