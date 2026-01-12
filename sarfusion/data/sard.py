import os

from roboflow import Roboflow
from PIL import Image
from torch.utils.data import Dataset

from sarfusion.data.utils import load_annotations, process_image_annotation_folders
from sarfusion.utils.structures import DataDict


def download_and_clean():
    api = os.environ.get("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=api)
    project = rf.workspace("sard").project("sardd")
    version = project.version(1)
    dataset = version.download("yolov9", location="dataset/sard")
    
    for subset in ["train", "valid", "test"]:
        subset_path = os.path.join(dataset.location, subset)
        types = ["images", "labels"]
        for t in types:
            subset_t = os.path.join(subset_path, t)
            subset_t_files = os.listdir(subset_t)
            for file in subset_t_files:
                parts = file.split(".")
                if "rf" in parts:
                    new_name = parts[0] + "." + parts[-1]
                    os.rename(os.path.join(subset_t, file), os.path.join(subset_t, new_name))
    
    return dataset.location


class YOLODataset(Dataset):
    def __init__(self, root, transform=None, return_path=False):
        self.image_paths, self.annotation_paths = process_image_annotation_folders(root)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations
        targets = load_annotations(annotation_path)

        if self.transform:
            img = self.transform(img, return_tensors="pt")['pixel_values'][0]
        data_dict = DataDict(
            images=img,
            target=targets
        )
        
        if self.return_path:
            data_dict.path = img_path

        return data_dict
    
    
class PoseClassificationDataset(Dataset):
    id2class = {
        0: "running",
        1: "walking",
        2: "laying_down",
        3: "not_defined",
        4: "seated",
        5: "stands",
    }
    def __init__(self, root, transform=None, return_path=False):
        self.image_paths = [os.path.join(root, img) for img in os.listdir(root)]
        self.transform = transform
        self.return_path = return_path
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        cls = int(img_path.split("_")[-1].split(".")[0])
        
        if self.transform:
            img = self.transform(img, return_tensors="pt")["pixel_values"][0]
        
        data_dict = DataDict(
            pixel_values=img,
            labels=cls
        )
        if self.return_path:
            data_dict.path = img_path
        return data_dict