import torch
from torch.utils.data import Dataset

class RootDataset(Dataset):
    def __init__(self, img_dir, target_dir):
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(img_dir)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        
        # Load raw input image
        img = cv2.imread(os.path.join(self.img_dir, name + ".png"), cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0  # Normalise image
        
        # Load the perfect preprocessed distance transform target map
        target = np.load(os.path.join(self.target_dir, name + "_target.npy"))
        target_tensor = torch.from_numpy(target).unsqueeze(0).float()
        
        return img_tensor, target_tensor
