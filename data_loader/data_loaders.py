from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from utils.utils import *

path = "../dataset/"




class CloudDataSet(Dataset):
  def __init__(self, df: pd.DataFrame = None, datatype: str = "train", img_ids: np.array = None,transforms = albu.Compose([albu.HorizontalFlip(),ToTensor()]),preprocessing = None):
    self.df = df
    if datatype != "test":
      self.data_folder = f'{path}/train_images'
    else:
      self.data_folder = f'{path}/test_images'
    self.img_ids = img_ids
    self.transforms = transforms
    self.preprocessing = preprocessing

  def __getitem__(self, idx):
    image_name = self.img_ids[idx]
    mask = make_masks(self.df,image_name)
    image_path = os.path.join(self.data_folder,image_name)
    img = cv2.imread(image_path)
    try: 
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      augmented = self.transforms(image = img,mask = mask)
    except:
      if isinstance(img,np.ndarray):
        print("Its an array but still error??")
      else:
        print("Not an array")
    #   print(image_name)

    img = augmented["image"]
    mask = augmented["mask"]


    if self.preprocessing:
      preprocessed = self.preprocessing(image = img,mask = mask)
      img = preprocessed["image"]
      mask = preprocessed["mask"]


    return img,mask

  def __len__(self):
    return len(self.img_ids)
