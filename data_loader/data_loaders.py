from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from utils.utils import *

path = "path/to/dataset"
train = pd.read_csv(f'{path}/train.csv')
sub = pd.read_csv(f'{path}/sample_submission.csv')


train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])

sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])

DEVICE = "cuda"


id_mask_count = train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"].apply(lambda x:x.split("_")[0]).value_counts().reset_index().rename(columns = \
 {"index":"img_id","Image_Label":"count"})



train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values




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
      print(image_name)

    img = augmented["image"]
    mask = augmented["mask"]


    if self.preprocessing:
      preprocessed = self.preprocessing(image = img,mask = mask)
      img = preprocessed["image"]
      mask = preprocessed["mask"]


    return img,mask

  def __len__(self):
    return len(self.img_ids)
