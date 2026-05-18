#!/usr/bin/env python3
"""
Script to download StyleDrop and DreamBooth datasets.
"""

import argparse
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

STYLEDROP_URL = [
    "https://images.unsplash.com/photo-1578926078693-4eb3d4499e43?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2008&q=80",
    "https://images.unsplash.com/photo-1578927107994-75410e4dcd51?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=729&q=80",
    "https://images.unsplash.com/photo-1612760721786-a42eb89aba02?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=735&q=80",
    "https://images.unsplash.com/photo-1630476504743-a4d342f88760?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1895&q=80",
    "https://upload.wikimedia.org/wikipedia/commons/6/66/VanGogh-starry_night_ballance1.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/de/Van_Gogh_Starry_Night_Drawing.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg/1024px-Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Vincent_van_Gogh_-_Self-portrait_with_grey_felt_hat_-_Google_Art_Project.jpg/1024px-Vincent_van_Gogh_-_Self-portrait_with_grey_felt_hat_-_Google_Art_Project.jpg",
    "https://img.freepik.com/free-vector/young-woman-walking-dog-leash-girl-leading-pet-park-flat-illustration_74855-11306.jpg?w=996&t=st=1685117377~exp=1685117977~hmac=dd6cf9856bdac8715c1d5464875225286942da2a01ea3851ea3936dd95d96a44",
    "https://img.freepik.com/free-vector/biophilic-design-workspace-abstract-concept_335657-3081.jpg?w=996&t=st=1685117412~exp=1685118012~hmac=cc89e22bd6dbeb3c2fc06396035863e612149b04ed6dee90791292a7151a4dd2",
    "https://img.freepik.com/free-vector/pine-tree-sticker-white-background_1308-75956.jpg?w=826&t=st=1685117428~exp=1685118028~hmac=36f37f710de7b4b7320d32dc169459f0bd0d6081e94e972198ab8d0a479f67e2",
    "https://img.freepik.com/free-psd/abstract-background-design_1297-124.jpg?w=996&t=st=1685117527~exp=1685118127~hmac=08c82ea8b2087dff81e01c946f999ed6bfb286a222c09e396b4d3f46787b2b50",
    "https://images.unsplash.com/photo-1538836026403-e143e8a59f04?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1448&q=80",
    "https://images.rawpixel.com/image_1000/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvZnJydWluX3dpbmRvd19kZWNheV9sZWF2ZS1pbWFnZS1reWNmbmM5aC5qcGc.jpg",
    "https://images.unsplash.com/photo-1518562180175-34a163b1a9a6?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1740&q=80",
    "https://images.unsplash.com/photo-1654648663068-0093ade5069e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1160&q=80",
    "https://img.freepik.com/free-psd/three-dimensional-real-estate-icon-mock-up_23-2149729145.jpg?w=996&t=st=1685117577~exp=1685118177~hmac=2d789df87b156c2e5578c8ddb69e4a3b3176206f81b774d9faea7492a4eafc0f",
    "https://images.unsplash.com/photo-1644664477908-f8c4b1d215c4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2080&q=80",
    "https://images.unsplash.com/photo-1634926878768-2a5b3c42f139?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=912&q=80",
    "https://github.com/styledrop/styledrop.github.io/blob/main/images/assets/image_6487327_crayon_02.jpg",
    "https://images.unsplash.com/photo-1668090956076-b2c9d6193e6b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1935&q=80",
    "https://images.unsplash.com/photo-1637234852730-677079a9d718?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=735&q=80",
    "https://images.unsplash.com/photo-1636391891394-56a534be9a1b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1160&q=80",
]

# DreamBooth dataset repository URL
DREAMBOOTH_REPO_URL = "https://github.com/google/dreambooth"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def has_images(root_dir: Path) -> bool:
    """Return True if directory contains at least one image file."""
    if not root_dir.exists() or not root_dir.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in root_dir.rglob("*")
    )


def download_file(url: str, filename: str, retries: int = 3) -> bool:
    """Download a file from URL with retries."""
    for attempt in range(retries):
        try:
            print(f"Downloading {filename} (attempt {attempt + 1}/{retries})")

            # Handle special case for GitHub blob URLs
            if "github.com" in url and "/blob/" in url:
                # Convert blob URL to raw URL
                url = url.replace("github.com", "raw.githubusercontent.com").replace(
                    "/blob/", "/"
                )

            # Create request with headers to avoid blocking
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
            )

            with urllib.request.urlopen(req) as response:
                with open(filename, "wb") as f:
                    shutil.copyfileobj(response, f)

            print(f"Successfully downloaded {filename}")
            return True
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            if attempt < retries - 1:
                print(f"Retrying...")
            else:
                print(f"Failed to download {filename} after {retries} attempts")
                return False
    return False


def download_styledrop_dataset(output_dir: str, force: bool = False) -> bool:
    """Download StyleDrop dataset images."""
    styledrop_dir = Path(output_dir) / "styledrop"
    styledrop_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading StyleDrop dataset to {styledrop_dir}")

    success_count = 0
    total_count = len(STYLEDROP_URL)

    for i, url in enumerate(STYLEDROP_URL):
        filename = f"{i:02d}.png"  # Use PNG extension for all images
        filepath = styledrop_dir / filename

        # Skip if file exists and not forcing
        if filepath.exists() and not force:
            print(f"Skipping {filename} (already exists)")
            success_count += 1
            continue

        if download_file(url, str(filepath)):
            success_count += 1

    print(f"StyleDrop dataset download completed: {success_count}/{total_count} files")
    return success_count == total_count


def clone_dreambooth_dataset(output_dir: str, force: bool = False) -> bool:
    """Clone DreamBooth dataset from GitHub."""
    dreambooth_dir = Path(output_dir) / "dreambooth"

    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: git is not installed or not available in PATH")
        print("Please install git to download the DreamBooth dataset")
        return False

    # Check if DreamBooth images already exist.
    # This avoids false positives when only metadata files (e.g. captions.jsonl) are present.
    if dreambooth_dir.exists() and not force and has_images(dreambooth_dir):
        print(f"DreamBooth dataset already exists at {dreambooth_dir}")
        return True
    if dreambooth_dir.exists() and force:
        print(
            "Force mode: refreshing DreamBooth dataset without deleting existing metadata files"
        )

    print(f"Cloning DreamBooth dataset to {dreambooth_dir}")

    try:
        # Clone the repository
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_repo_dir = Path(temp_dir) / "dreambooth"

            # Clone the repository
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    DREAMBOOTH_REPO_URL,
                    str(temp_repo_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            # Find the dataset directory in the cloned repository
            # Look for common dataset directory names
            possible_dirs = ["dataset", "datasets", "data", "images"]
            dataset_source = None

            for possible_dir in possible_dirs:
                candidate = temp_repo_dir / possible_dir
                if candidate.exists() and candidate.is_dir():
                    dataset_source = candidate
                    break

            # If no standard dataset directory found, look for directories with images
            if dataset_source is None:
                for item in temp_repo_dir.iterdir():
                    if (
                        item.is_dir()
                        and any(item.glob("*.jpg"))
                        or any(item.glob("*.png"))
                    ):
                        dataset_source = item
                        break

            # If still no dataset found, copy the entire repository (excluding .git)
            if dataset_source is None:
                print("No specific dataset directory found, copying repository content")
                dataset_source = temp_repo_dir

            # Copy the dataset (merge into destination; preserves existing metadata files)
            dreambooth_dir.mkdir(parents=True, exist_ok=True)

            if dataset_source == temp_repo_dir:
                # Copy everything except .git directory
                for item in dataset_source.iterdir():
                    if item.name != ".git":
                        if item.is_dir():
                            shutil.copytree(
                                item, dreambooth_dir / item.name, dirs_exist_ok=True
                            )
                        else:
                            shutil.copy2(item, dreambooth_dir)
            else:
                # Copy the specific dataset directory content
                for item in dataset_source.iterdir():
                    if item.is_dir():
                        shutil.copytree(
                            item, dreambooth_dir / item.name, dirs_exist_ok=True
                        )
                    else:
                        shutil.copy2(item, dreambooth_dir)

        print(f"DreamBooth dataset downloaded successfully to {dreambooth_dir}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to clone DreamBooth repository: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error downloading DreamBooth dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download StyleDrop and DreamBooth datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Output directory for datasets (default: datasets)",
    )
    parser.add_argument(
        "--dataset",
        choices=["styledrop", "dreambooth", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )

    args = parser.parse_args()

    # Convert to absolute path
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    success = True

    if args.dataset in ["styledrop", "all"]:
        print("\n" + "=" * 50)
        print("DOWNLOADING STYLEDROP DATASET")
        print("=" * 50)
        if not download_styledrop_dataset(str(output_dir), args.force):
            success = False

    if args.dataset in ["dreambooth", "all"]:
        print("\n" + "=" * 50)
        print("DOWNLOADING DREAMBOOTH DATASET")
        print("=" * 50)
        if not clone_dreambooth_dataset(str(output_dir), args.force):
            success = False

    if success:
        print("\n" + "=" * 50)
        print("ALL DATASETS DOWNLOADED SUCCESSFULLY!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("SOME DOWNLOADS FAILED - CHECK LOGS ABOVE")
        print("=" * 50)
        exit(1)


if __name__ == "__main__":
    main()
