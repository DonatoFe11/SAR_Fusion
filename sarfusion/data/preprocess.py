from io import StringIO
import os
import accelerate
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image, ImageDraw

from ultralytics.data.utils import check_det_dataset

from sarfusion.data.sard import YOLODataset, download_and_clean
from sarfusion.data.utils import build_preprocessor, is_annotation_valid
from sarfusion.data.wisard import MISSING_ANNOTATIONS, TEST_FOLDERS, TRAIN_FOLDERS, VAL_FOLDERS, VIS_ONLY, IR_ONLY, VIS_IR, WiSARDYOLODataset, generate_wisard_filelist, get_wisard_folders
from sarfusion.models import build_model
from sarfusion.utils.utils import load_yaml


def crop_bboxes(image, targets, crop_size=224):
    crop_width = crop_height = crop_size
    cropped_images = []
    # Pad the image to make sure the bounding box is not out of the image
    image_width, image_height = image.size(2), image.size(1)
    image = transforms.Pad(padding=crop_size)(image)
    for target in targets:
        class_label, x_center, y_center, width, height = target

        # Calculate the center of the bounding box
        bbox_center_x = x_center * image_width + crop_width
        bbox_center_y = y_center * image_height + crop_height

        # Calculate the crop region
        xmin = round(bbox_center_x - crop_width / 2)
        xmax = round(bbox_center_x + crop_width / 2)
        ymin = round(bbox_center_y - crop_height / 2)
        ymax = round(bbox_center_y + crop_height / 2)

        # Fix precision errors
        if xmax - xmin != crop_width:
            if (xmax - xmin) < crop_width:
                xmax += 1
            else:
                xmin += 1
        if ymax - ymin != crop_height:
            if (ymax - ymin) < crop_height:
                ymax += 1
            else:
                ymin += 1

        # Crop the image
        cropped_image = image[:, ymin:ymax, xmin:xmax]
        cropped_images.append((cropped_image, class_label))

    return cropped_images


def generate_pose_classification_dataset(output_dir):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset_location = download_and_clean()
    os.makedirs(output_dir, exist_ok=True)

    for subset in ["train", "valid", "test"]:
        print(f"Processing {subset}...")
        os.makedirs(f"{output_dir}/{subset}", exist_ok=True)
        subset_location = f"{dataset_location}/{subset}"

        dataset = YOLODataset(subset_location, transform=transform)

        bar = tqdm(dataset, total=len(dataset))

        for i, data_dict in enumerate(bar):
            image = data_dict.images
            targets = data_dict.target
            cropped_images = crop_bboxes(image, targets)
            for j, (cropped_image, class_label) in enumerate(cropped_images):
                save_path = f"{output_dir}/{subset}/{i}_{j}_{class_label}.png"
                cropped_image = cropped_image.mul(255).byte().permute(1, 2, 0).numpy()
                cropped_image = Image.fromarray(cropped_image)
                cropped_image.save(save_path)


def annotate_rgb_wisard(root, model_yaml):
    accelerator = accelerate.Accelerator()
    vis = VIS_ONLY + [f[0] for f in VIS_IR]
    params = load_yaml(model_yaml)
    model_params = params["model"]
    model = build_model(model_params)
    model.eval()
    model = accelerator.prepare(model)
    transform, detransform = build_preprocessor(params)

    print("Placing -1 in all labels...")
    # Place -1 in all labels
    subsets = list(filter(lambda x: os.path.isdir(f"{root}/{x}"), os.listdir(root)))
    for subset in tqdm(subsets):
        subset_location = f"{root}/{subset}"
        label_path = f"{subset_location}/labels"
        for label_file in os.listdir(label_path):
            with open(f"{label_path}/{label_file}", "r") as file:
                lines = file.readlines()
            for i in range(len(lines)):
                row = lines[i].split(" ")
                row[0] = "-1"
                lines[i] = " ".join(row)
            with open(f"{label_path}/{label_file}", "w") as file:
                file.writelines(lines)

    for subset in vis:
        print(f"Processing {subset}...")
        subset_location = f"{root}/{subset}"

        dataset = YOLODataset(subset_location, transform=transform, return_path=True)

        bar = tqdm(dataset, total=len(dataset))

        for i, data_dict in enumerate(bar):
            image = data_dict.images
            targets = data_dict.target
            path = data_dict.path
            cropped_images = crop_bboxes(image, targets)
            new_targets = []
            for (cropped_image, _), target in zip(cropped_images, targets):
                # Check if the annotation is valid

                if not is_annotation_valid(target):
                    print(f"Invalid annotation in {path}, {target}")
                    continue
                input_dict = {
                    "pixel_values": cropped_image.unsqueeze(0).to(accelerator.device)
                }
                result = model(**input_dict)
                class_label = result.logits.argmax().item()
                new_targets.append((class_label, *target[1:]))
            # Replace the target file with the new one
            gt_path = path.replace("images", "labels")
            gt_path, ext = os.path.splitext(gt_path)
            gt_path += ".txt"
            with open(gt_path, "w") as file:
                for target in new_targets:
                    file.write(" ".join(map(str, target)) + "\n")


def simplify_wisard(root):
    label_map = {
        0: 0,  # running
        1: 0,  # walking
        2: 1,  # laying_down
        3: 2,  # not_defined
        4: 1,  # seated
        5: 0,  # stands
    }
    vis = VIS_ONLY + [f[0] for f in VIS_IR]    
            
    for subset in vis:
        print(f"Processing {subset}...")
        subset_location = f"{root}/{subset}"

        dataset = YOLODataset(subset_location, transform=lambda x: x, return_path=True)

        bar = tqdm(dataset, total=len(dataset))

        for i, data_dict in enumerate(bar):
            image = data_dict.images
            targets = data_dict.target
            path = data_dict.path
            new_targets = []
            for target in targets:
                # Check if the annotation is valid
                if not is_annotation_valid(target):
                    print(f"Invalid annotation in {path}, {target}")
                    continue
                class_label = label_map[target[0]]
                new_targets.append((class_label, *target[1:]))
            # Replace the target file with the new one
            gt_path = path.replace("images", "labels")
            gt_path, ext = os.path.splitext(gt_path)
            gt_path += ".txt"
            with open(gt_path, "w") as file:
                for target in new_targets:
                    file.write(" ".join(map(str, target)) + "\n")
                    

def check_vis_ir(root):
    for f in VIS_IR:
        vis = f[0]
        ir = f[1]
        vis_path = f"{root}/{vis}"
        ir_path = f"{root}/{ir}"
        vis_images = os.listdir(f"{vis_path}/images")
        ir_images = os.listdir(f"{ir_path}/images")
        vis_labels = os.listdir(f"{vis_path}/labels")
        ir_labels = os.listdir(f"{ir_path}/labels")
        assert len(vis_images) == len(ir_images), f"Number of VIS and IR images do not match: {len(vis_images)} != {len(ir_images)} in {vis} and {ir}"
        assert len(vis_labels) == len(ir_labels), f"Number of VIS and IR labels do not match: {len(vis_labels)} != {len(ir_labels)} in {vis} and {ir}"
    
                
                
def wisard_to_yolo_dataset(root):
    subfolders = [os.path.join(root, f) for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    print(f"Found {len(subfolders)} subfolders.")
    for subfolder in subfolders:
        print(f"Processing {subfolder}...")
        os.makedirs(f"{subfolder}/images", exist_ok=True)
        os.makedirs(f"{subfolder}/labels", exist_ok=True)
        for image in tqdm(os.listdir(subfolder)):
            ext = os.path.splitext(image)[-1]
            if ext.lower() in ['.jpg', '.jpeg', '.png']:
                image_path = os.path.join(subfolder, image)
                label_path = image_path.replace(ext, ".txt")
                os.rename(image_path, f"{subfolder}/images/{image}")
                if os.path.exists(label_path):
                    os.rename(label_path, f"{subfolder}/labels/{image.replace(ext, '.txt')}") 
    for annotation in MISSING_ANNOTATIONS:
        print(f"Creating missing annotation {annotation}...")
        with open(f"{root}/{annotation}", 'w') as file:
            pass
        
    print("Generating VIS filelists...")
    folders = "vis"
    train_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TRAIN_FOLDERS
    ]
    generate_wisard_filelist(root, train_folders, "train_vis.txt")
    val_folders = [
        folder for folder in get_wisard_folders(folders) if folder in VAL_FOLDERS
    ]
    generate_wisard_filelist(root, val_folders, "val_vis.txt")
    test_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TEST_FOLDERS
    ]
    generate_wisard_filelist(root, test_folders, "test_vis.txt")

    folders = "ir"
    print("Generating IR filelists...")
    train_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TRAIN_FOLDERS
    ]
    generate_wisard_filelist(root, train_folders, "train_ir.txt")
    val_folders = [
        folder for folder in get_wisard_folders(folders) if folder in VAL_FOLDERS
    ]
    generate_wisard_filelist(root, val_folders, "val_ir.txt")
    test_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TEST_FOLDERS
    ]
    generate_wisard_filelist(root, test_folders, "test_ir.txt")

    folders = "vis_ir"
    print("Generating VIS_IR filelists...")
    train_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TRAIN_FOLDERS
    ]
    generate_wisard_filelist(root, train_folders, "train_vis_ir.txt")
    val_folders = [
        folder for folder in get_wisard_folders(folders) if folder in VAL_FOLDERS
    ]
    generate_wisard_filelist(root, val_folders, "val_vis_ir.txt")
    test_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TEST_FOLDERS
    ]
    generate_wisard_filelist(root, test_folders, "test_vis_ir.txt")

    folders = "vis_all_ir_sync"
    print("Generating VIS_ALL_IR_SYNC filelists...")
    train_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TRAIN_FOLDERS
    ]
    generate_wisard_filelist(root, train_folders, "train_vis_all_ir_sync.txt")
    val_folders = [
        folder for folder in get_wisard_folders(folders) if folder in VAL_FOLDERS
    ]
    generate_wisard_filelist(root, val_folders, "val_vis_all_ir_sync.txt")
    test_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TEST_FOLDERS
    ]
    generate_wisard_filelist(root, test_folders, "test_vis_all_ir_sync.txt")

    folders = "vis_sync_ir_all"
    print("Generating VIS_SYNC_IR_ALL filelists...")
    train_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TRAIN_FOLDERS
    ]
    generate_wisard_filelist(root, train_folders, "train_vis_sync_ir_all.txt")
    val_folders = [
        folder for folder in get_wisard_folders(folders) if folder in VAL_FOLDERS
    ]
    generate_wisard_filelist(root, val_folders, "val_vis_sync_ir_all.txt")
    test_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TEST_FOLDERS
    ]
    generate_wisard_filelist(root, test_folders, "test_vis_sync_ir_all.txt")

    folders = "all"
    print("Generating ALL filelists...")
    train_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TRAIN_FOLDERS
    ]
    generate_wisard_filelist(root, train_folders, "train_all.txt")
    val_folders = [
        folder for folder in get_wisard_folders(folders) if folder in VAL_FOLDERS
    ]
    generate_wisard_filelist(root, val_folders, "val_all.txt")
    test_folders = [
        folder for folder in get_wisard_folders(folders) if folder in TEST_FOLDERS
    ]
    generate_wisard_filelist(root, test_folders, "test_all.txt")


def generate_tiled_wisard(root, data_yaml, imgsz=4096, tilesize=512, stride=32):
    root_prefix = root.split("WiSARD")[0]
    outfolder = f"{root_prefix}WiSARD_Tiled_{tilesize}"
    os.makedirs(outfolder, exist_ok=True)
    
    data_dict = check_det_dataset(data_yaml)
    phases = ["train", "val", "test"]
    for phase in phases:
        img_path = data_dict[phase]
        dataset = WiSARDYOLODataset(
            img_path=img_path,
            imgsz=imgsz,
            batch_size=1,
            augment=False,
            # hyp=None, 
            rect=False,
            cache=None,
            single_cls=False,
            stride=int(stride),
            pad=0.0,
            prefix="",
            task="detect",
            classes=None,
            data=data_dict,
            fraction=1.0,
            augment_vis_ir=False,
        )
        write_dataset_to_disk(dataset, root, outfolder, imgsz, tilesize, phase)
    

def write_dataset_to_disk(dataset, root, outfolder, imgsz, tilesize, phase):
    DISCARD_THRESHOLD = 0.8
    rgb_only_list = []
    multimodal_list = []
    for i, elem in enumerate(tqdm(dataset)):
        im_file = elem["im_file"]
        im = elem["img"]
        classes = elem["cls"]
        bboxes = elem["bboxes"]
        for x in range(0, imgsz, tilesize):
            for y in range(0, imgsz, tilesize):
                tile = im[:, y:y + tilesize, x:x + tilesize]
                counts = tile.unique(return_counts=True)[1]
                if (counts / counts.sum()).max() > DISCARD_THRESHOLD: # Discard tiles with too much padding
                    continue
                tile_classes = []
                tile_bboxes = []
                for j, bbox in enumerate(bboxes):
                    x_center, y_center, width, height = bbox
                    if x_center * imgsz >= x and x_center * imgsz <= x + tilesize and y_center * imgsz >= y and y_center * imgsz <= y + tilesize:
                        tile_classes.append(classes[j].item())
                        tile_bboxes.append(bbox)
                tile = tile.permute(1, 2, 0).numpy()
                output_string = StringIO()
                for j, bbox in enumerate(tile_bboxes):
                    class_label = tile_classes[j]
                    x_center, y_center, width, height = bbox
                    x_center = (x_center * imgsz - x) / tilesize
                    y_center = (y_center * imgsz - y) / tilesize
                    width = width * imgsz / tilesize
                    height = height * imgsz / tilesize
                    output_string.write(f"{class_label} {x_center} {y_center} {width} {height}\n")
                # Get the final string
                result = output_string.getvalue()
                # Close the StringIO object
                output_string.close()
                
                if isinstance(im_file, str):
                    ext = os.path.splitext(im_file)[-1]
                    im_file_out = im_file.replace(root, outfolder).replace(ext, f"_{x}_{y}{ext}")
                    labels_file_out = im_file_out.replace("images", "labels").replace(ext, ".txt")
                    os.makedirs(os.path.dirname(im_file_out), exist_ok=True)
                    os.makedirs(os.path.dirname(labels_file_out), exist_ok=True)
                    tile = Image.fromarray(tile)
                    tile.save(im_file_out)
                    with open(labels_file_out, "w") as file:
                        file.write(result)
                    rgb_only_list.append(im_file_out)
                    multimodal_list.append(im_file_out)
                else:                 
                    ext_rgb = os.path.splitext(im_file[0])[-1]
                    ext_ir = os.path.splitext(im_file[1])[-1]
                    rgb_file_out, ir_file_out = im_file
                    rgb_file_out = rgb_file_out.replace(root, outfolder).replace(ext_rgb, f"_{x}_{y}{ext_rgb}")
                    ir_file_out = ir_file_out.replace(root, outfolder).replace(ext_ir, f"_{x}_{y}{ext_ir}")
                    labels_rgb_file_out = rgb_file_out.replace("images", "labels").replace(f"{ext_rgb}", ".txt")
                    labels_ir_file_out = ir_file_out.replace("images", "labels").replace(f"{ext_ir}", ".txt")
                    os.makedirs(os.path.dirname(rgb_file_out), exist_ok=True)
                    os.makedirs(os.path.dirname(ir_file_out), exist_ok=True)
                    os.makedirs(os.path.dirname(labels_rgb_file_out), exist_ok=True)
                    os.makedirs(os.path.dirname(labels_ir_file_out), exist_ok=True)
                    rgb_image = Image.fromarray(tile[:, :, :3])
                    ir_image = Image.fromarray(np.repeat(tile[:, :, 3:], 3, axis=2))
                    rgb_image.save(rgb_file_out)
                    ir_image.save(ir_file_out)
                    with open(labels_rgb_file_out, "w") as file:
                        file.write(result)
                    with open(labels_ir_file_out, "w") as file:
                        file.write(result)
                    rgb_only_list.append(rgb_file_out)
                    multimodal_list.append((rgb_file_out, ir_file_out))
    print(f"Finished processing {phase}...")
        
    with open(dataset.img_path.replace("_all_ir_sync", "").replace("WiSARD", f"WiSARD_Tiled_{tilesize}"), "w") as file:
        for item in rgb_only_list:
            file.write(f"{item}\n")
    print(f"Finished writing {phase} RGB only filelist...")
    with open(dataset.img_path.replace("WiSARD", f"WiSARD_Tiled_{tilesize}"), "w") as file:
        for item in multimodal_list:
            if isinstance(item, str):
                file.write(f"{item}\n")
            else:
                file.write(f"{item[0]},{item[1]}\n")
    print(f"Finished writing {phase} multimodal filelist...")
                        