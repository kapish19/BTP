"""
Automated Kaggle Dataset Downloader for DG-ReID.
Downloads Market-1501 and DukeMTMC-reID directly from Kaggle using credentials
from environment variables (KAGGLE_USERNAME, KAGGLE_KEY), Colab Secrets, or kaggle.json.
"""

import os
import sys
import glob
import json
import shutil
import zipfile
import subprocess


def setup_kaggle_credentials():
    """Configures Kaggle credentials from environment variables, Colab Secrets, or local files."""
    print("🔑 Configuring Kaggle credentials...")

    # 1. Try Google Colab Secrets (userdata)
    try:
        from google.colab import userdata
        if 'KAGGLE_USERNAME' in os.environ or userdata.get('KAGGLE_USERNAME'):
            os.environ['KAGGLE_USERNAME'] = os.environ.get('KAGGLE_USERNAME') or userdata.get('KAGGLE_USERNAME')
            os.environ['KAGGLE_KEY'] = os.environ.get('KAGGLE_KEY') or userdata.get('KAGGLE_KEY')
            print("  ✅ Retrieved Kaggle credentials from Google Colab Secrets.")
    except Exception:
        pass

    # 2. Check standard environment variables
    kaggle_user = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

    if kaggle_user and kaggle_key:
        with open(kaggle_json_path, 'w') as f:
            json.dump({'username': kaggle_user, 'key': kaggle_key}, f)
        os.chmod(kaggle_json_path, 0o600)
        print(f"  ✅ Kaggle credentials written from environment variables for user: {kaggle_user}")
        return True

    # 3. Check for existing kaggle.json in current directory or /content
    for cand in ['/content/kaggle.json', 'kaggle.json', kaggle_json_path]:
        if os.path.exists(cand) and cand != kaggle_json_path:
            shutil.copy(cand, kaggle_json_path)
            os.chmod(kaggle_json_path, 0o600)
            print(f"  ✅ Copied kaggle.json from {cand} to ~/.kaggle/kaggle.json")
            return True
        elif os.path.exists(cand) and cand == kaggle_json_path:
            print(f"  ✅ Found existing credentials at {kaggle_json_path}")
            return True

    print("  ⚠️ Kaggle credentials not detected in environment or files.")
    print("     Please set KAGGLE_USERNAME and KAGGLE_KEY in your environment, or place kaggle.json in ~/.kaggle/")
    return False


def run_cmd(cmd):
    """Executes a shell command and streams output."""
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"  ⚠️ Command '{cmd}' failed: {res.stderr.strip()}")
        return False
    return True


def download_and_extract_kaggle(slug_list, dest_dir, dataset_name):
    """Tries downloading a dataset from a list of candidate Kaggle slugs."""
    os.makedirs(dest_dir, exist_ok=True)
    temp_download = os.path.join(dest_dir, f"_temp_{dataset_name}")
    os.makedirs(temp_download, exist_ok=True)

    success = False
    for slug in slug_list:
        print(f"  📦 Attempting Kaggle download: {slug} ...")
        cmd = f"kaggle datasets download -d {slug} -p {temp_download}"
        if run_cmd(cmd):
            # Check for zip file
            zip_files = glob.glob(os.path.join(temp_download, "*.zip"))
            if zip_files:
                print(f"  📦 Extracting {zip_files[0]} -> {dest_dir} ...")
                with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                    zip_ref.extractall(dest_dir)
                success = True
                break

    # Clean up temp
    shutil.rmtree(temp_download, ignore_errors=True)
    return success


def prepare_datasets_kaggle(data_dir="data"):
    """Downloads Market-1501 and DukeMTMC-reID directly from Kaggle."""
    os.makedirs(data_dir, exist_ok=True)
    setup_kaggle_credentials()

    # 1. Market-1501
    market_train = glob.glob(os.path.join(data_dir, "**", "bounding_box_train", "*.jpg"), recursive=True)
    if not market_train:
        print("\n📥 Downloading Market-1501 from Kaggle...")
        market_slugs = ["pengcw1/market-1501", "priyanagda/market1501", "ouassimaazzouzi/market1501"]
        market_dest = os.path.join(data_dir, "Market-1501-v15.09.15")
        ok = download_and_extract_kaggle(market_slugs, market_dest, "market1501")
        if not ok:
            print("  ⚠️ Kaggle download failed. Attempting direct web mirror fallback...")
            run_cmd(f"wget -q -c http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip -O {data_dir}/m.zip && unzip -q {data_dir}/m.zip -d {data_dir}/ && rm -f {data_dir}/m.zip")
    else:
        print(f"  ✅ Market-1501 already downloaded ({len(market_train)} train images).")

    # 2. DukeMTMC-reID
    duke_test = glob.glob(os.path.join(data_dir, "**", "DukeMTMC-reID", "**", "bounding_box_test", "*.jpg"), recursive=True)
    if not duke_test:
        duke_test = glob.glob(os.path.join(data_dir, "**", "bounding_box_test", "*.jpg"), recursive=True)

    if not duke_test or len(duke_test) < 1000:
        print("\n📥 Downloading DukeMTMC-reID from Kaggle...")
        duke_slugs = ["whurobin/dukemtmcreid", "27wilson/dukemtmcreid", "ivkrashenyi/dukemtmc-reid"]
        duke_dest = os.path.join(data_dir, "DukeMTMC-reID")
        ok = download_and_extract_kaggle(duke_slugs, duke_dest, "dukemtmc")
        if not ok:
            print("  ⚠️ Kaggle download failed. Attempting direct GitHub release mirror fallback...")
            run_cmd(f"wget -q -c https://github.com/layumi/DukeMTMC-reID_evaluation/releases/download/v1.0/DukeMTMC-reID.zip -O {data_dir}/d.zip && unzip -q {data_dir}/d.zip -d {data_dir}/ && rm -f {data_dir}/d.zip")
    else:
        print(f"  ✅ DukeMTMC-reID already downloaded ({len(duke_test)} gallery images).")

    # Summary verification
    m_train_imgs = glob.glob(os.path.join(data_dir, "**", "bounding_box_train", "*.jpg"), recursive=True)
    d_test_imgs = glob.glob(os.path.join(data_dir, "**", "bounding_box_test", "*.jpg"), recursive=True)
    print("\n" + "=" * 60)
    print("🔍 DATASET VERIFICATION SUMMARY:")
    print("=" * 60)
    print(f"  ✅ Market-1501 Train Images:   {len(m_train_imgs):6d} images")
    print(f"  ✅ Target Gallery Images:      {len(d_test_imgs):6d} images")
    print("=" * 60)


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    prepare_datasets_kaggle(target_dir)
