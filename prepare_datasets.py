import os
import sys
import shutil
import urllib.request

def setup_data(data_dir="data", drive_dir="/content/drive/MyDrive/datasets"):
    os.makedirs(data_dir, exist_ok=True)
    print("🚀 Preparing datasets for DG-ReID...")

    # 1. Setup Kaggle credentials from environment variables or files
    kaggle_user = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")
    kdir = os.path.expanduser("~/.kaggle")
    os.makedirs(kdir, exist_ok=True)
    kdest = os.path.join(kdir, "kaggle.json")

    if kaggle_user and kaggle_key:
        import json
        with open(kdest, "w") as f:
            json.dump({"username": kaggle_user, "key": kaggle_key}, f)
        os.chmod(kdest, 0o600)
        print(f"  ✅ Kaggle API credentials written from environment variables for user: {kaggle_user}")
    else:
        for src in ["/content/kaggle.json", "kaggle.json"]:
            if os.path.exists(src):
                if not os.path.exists(kdest):
                    shutil.copy(src, kdest)
                    os.chmod(kdest, 0o600)
                    print("  ✅ Kaggle API credentials configured from file.")

    use_drive = os.path.exists("/content/drive/MyDrive")
    if use_drive:
        os.makedirs(drive_dir, exist_ok=True)
        print(f"  📁 Google Drive persistent storage enabled at: {drive_dir}")

    # Dataset configurations: (folder_name, kaggle_slug, unzip_subfolder)
    dataset_configs = [
        {
            "name": "MSMT17_V2",
            "drive_cands": ["MSMT17_V2", "MSMT17", "msmt17"],
            "kaggle": "ouassimaazzouzi/msmt17",
            "unzip_to": "MSMT17_V2",
        },
        {
            "name": "Market-1501-v15.09.15",
            "drive_cands": ["Market-1501-v15.09.15", "Market-1501", "market1501"],
            "kaggle": "pengcw1/market-1501",
            "unzip_to": "",
        },
        {
            "name": "cuhk_sysu",
            "drive_cands": ["cuhk_sysu", "cuhk-sysu"],
            "kaggle": "manaschaiaonon/cuhk-sysu",
            "unzip_to": "cuhk_sysu",
        },
        {
            "name": "cuhk03",
            "drive_cands": ["cuhk03", "cuhk-03"],
            "kaggle": "priyanagda/cuhk03",
            "unzip_to": "cuhk03",
        },
        {
            "name": "DukeMTMC-reID",
            "drive_cands": ["DukeMTMC-reID", "dukemtmc-reid", "dukemtmc"],
            "kaggle": "whurobin/dukemtmcreid",
            "unzip_to": "DukeMTMC-reID",
        },
    ]

    for cfg in dataset_configs:
        name = cfg["name"]
        local_target = os.path.join(data_dir, name)

        # Check if already linked and valid
        if os.path.exists(local_target) and len(os.listdir(local_target)) > 0:
            print(f"  ✅ {name:24s} already ready.")
            continue

        # Check Google Drive first (instant persistent link)
        found_on_drive = None
        if use_drive:
            for cand in cfg["drive_cands"]:
                cand_path = os.path.join(drive_dir, cand)
                if os.path.exists(cand_path) and len(os.listdir(cand_path)) > 0:
                    found_on_drive = cand_path
                    break

        if found_on_drive:
            if os.path.islink(local_target): os.unlink(local_target)
            os.symlink(found_on_drive, local_target)
            print(f"  ⚡ {name:24s} linked from Google Drive: {found_on_drive}")
            continue

        # If not on Drive, download via Kaggle
        if cfg.get("kaggle"):
            slug = cfg["kaggle"]
            download_dir = drive_dir if use_drive else data_dir
            dest_dir = os.path.join(download_dir, cfg.get("unzip_to") or "")
            os.makedirs(dest_dir, exist_ok=True)

            print(f"  📦 Downloading {name} ({slug})...")
            zip_dest = os.path.join(download_dir, f"{name}.zip")
            res = os.system(f"kaggle datasets download -d {slug} -p {download_dir}")
            
            # Find the downloaded zip file
            zip_files = [f for f in os.listdir(download_dir) if f.endswith(".zip")]
            for zf in zip_files:
                zpath = os.path.join(download_dir, zf)
                os.system(f"unzip -o -q {zpath} -d {dest_dir}")
                os.remove(zpath)

            # Link into data_dir if downloaded to Drive
            actual_folder = os.path.join(download_dir, name)
            if not os.path.exists(actual_folder):
                actual_folder = dest_dir

            if os.path.islink(local_target): os.unlink(local_target)
            if actual_folder != local_target:
                os.symlink(actual_folder, local_target)
            print(f"  ✅ {name:24s} downloaded and prepared.")

    # 4. Ensure CUHK03 protocol files exist
    cuhk_dir = os.path.join(data_dir, "cuhk03")
    if os.path.exists(cuhk_dir):
        m1 = os.path.join(cuhk_dir, "cuhk03_new_protocol_config_detected.mat")
        m2 = os.path.join(cuhk_dir, "cuhk03_new_protocol_config_labeled.mat")
        u1 = "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/torchreid/datasets/cuhk03_new_protocol_config_detected.mat"
        u2 = "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/torchreid/datasets/cuhk03_new_protocol_config_labeled.mat"
        if not os.path.exists(m1):
            try: urllib.request.urlretrieve(u1, m1)
            except Exception: pass
        if not os.path.exists(m2):
            try: urllib.request.urlretrieve(u2, m2)
            except Exception: pass

    # 5. Verify all 4 datasets
    print("\n" + "=" * 60)
    print("🔍 VERIFYING ALL DATASETS:")
    print("=" * 60)
    from datasets import DATASET_REGISTRY
    all_ready = True
    for ds_name in ["MSMT17", "cuhk_sysu", "cuhk03", "Market"]:
        try:
            ds = DATASET_REGISTRY.get(ds_name)(root=data_dir, verbose=False)
            n_tr = len(ds.train)
            n_te = len(ds.query) + len(ds.gallery)
            print(f"  ✅ {ds_name:15s} | Train: {n_tr:6d} imgs | Test: {n_te:6d} imgs")
        except Exception as e:
            print(f"  ❌ {ds_name:15s} | Error: {e}")
            all_ready = False
    print("=" * 60)
    if all_ready:
        print("🎉 ALL DATASETS READY FOR TRAINING!\n")
    else:
        print("⚠️ Some datasets had errors. Please check the logs above.\n")

if __name__ == "__main__":
    setup_data()
