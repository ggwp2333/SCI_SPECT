import zipfile
from pathlib import Path

directory = Path("/datastore01/user-storage/huitian/PET_denoising_data/zipped_data/")
zip_files = list(directory.glob("Subject_31-36.zip"))

for file in zip_files:
    print(f'Processing {file}')
    with zipfile.ZipFile(file, 'r') as z:
        z.extractall(f'/datastore01/user-storage/huitian/PET_denoising_data/ud_challenge_dataset/')
