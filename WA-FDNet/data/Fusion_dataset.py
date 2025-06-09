import glob
from torch.utils.data.dataset import Dataset
from scripts.util import *
from torchvision.transforms import functional as F
from basicsr.utils.registry import DATASET_REGISTRY
import cv2

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.txt"))))
    data.sort()
    filenames.sort()
    return data, filenames

def filter_images_by_size(filepath_vi, filepath_ir, filenames_vi, filenames_ir, target_size=(1024, 768)):  
    indices_to_keep = []  
    for idx, vi_path in enumerate(filepath_vi):  
        try:  
            with Image.open(vi_path) as img:  
                if img.size == target_size:  
                    indices_to_keep.append(idx)  
        except Exception as e:  
            continue  
    
    filepath_vi_new      = [filepath_vi[i] for i in indices_to_keep]  
    filepath_ir_new      = [filepath_ir[i] for i in indices_to_keep]  
    filenames_vi_new     = [filenames_vi[i] for i in indices_to_keep]  
    filenames_ir_new     = [filenames_ir[i] for i in indices_to_keep]  
    return filepath_vi_new, filepath_ir_new, filenames_vi_new, filenames_ir_new  

@DATASET_REGISTRY.register()
class VI_FusionDataset(Dataset):
    def __init__(self, opt):
        super(VI_FusionDataset, self).__init__()
        assert opt["name"] in ['train', 'val', 'test'], 'name must be "train"|"val"|"test"'
        self.opt = opt
        self.split = opt["name"]
        self.is_crop = opt["is_crop"]
        crop_size = opt["crop_size"]
        self.shape = (opt["crop_size"],opt["crop_size"])
        self.target_size = opt["target_size"]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor=1)
        self.dethr_transform = train_hr_transform(crop_size)
        self.centerhr_transform = train_centerhr_transform(crop_size)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.1)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.1)
        
        self.data_dir_vi = opt["vi_path"]
        self.data_dir_ir = opt["ir_path"]
        self.filepath_vi, self.filenames_vi = prepare_data_path(self.data_dir_vi)
        self.filepath_ir, self.filenames_ir = prepare_data_path(self.data_dir_ir)
        self.filepath_vi, self.filepath_ir, self.filenames_vi, self.filenames_ir = filter_images_by_size(  
        self.filepath_vi, self.filepath_ir, self.filenames_vi, self.filenames_ir, target_size=(self.target_size[0],self.target_size[1]))  
        self.length = min(len(self.filenames_vi), len(self.filenames_ir))

    def __getitem__(self, index):
        vi_image = Image.open(self.filepath_vi[index])
        ir_image = Image.open(self.filepath_ir[index])
        h, w = vi_image.height, vi_image.width

        if self.split == 'train':
            if self.is_crop:
                crop_size = self.dethr_transform(vi_image)
                vi_image, ir_image = F.crop(vi_image, crop_size[0], crop_size[1], crop_size[2],crop_size[3]), \
                                            F.crop(ir_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3])

            # Random horizontal flipping
            if random.random() > 0.5:
                vi_image = self.hflip(vi_image)
                ir_image = self.hflip(ir_image)

            # Random vertical flipping
            if random.random() > 0.5:
                vi_image = self.vflip(vi_image)
                ir_image = self.vflip(ir_image)

            vi_image = ToTensor()(vi_image)
            ir_image = ToTensor()(ir_image)
            cat_img = torch.cat([vi_image[:, :, :], ir_image[0:1, :, :]], axis=0)

            return {'img': cat_img, 'vi': vi_image[:, :, :], 'ir': ir_image[0:1, :, :]}

        elif self.split == 'val':
            if self.is_crop:
                crop_size = self.centerhr_transform(vi_image)
                vi_image, ir_image = F.crop(vi_image, crop_size[0], crop_size[1], crop_size[2],crop_size[3]), \
                                            F.crop(ir_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3])

            vi_image = ToTensor()(vi_image)
            ir_image = ToTensor()(ir_image)
            cat_img = torch.cat([vi_image[:, :, :], ir_image[0:1, :, :]], axis=0)

            return {'img': cat_img, 'vi': vi_image[:, :, :], 'ir': ir_image[0:1, :, :]},{"im_name": self.filenames_vi[index]}

    def __len__(self):
        return self.length
