import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils.utils import plot_loss_curves
from post_process import *
## Make submission File from test.py
## Convert the masks to the RLE 

def load_model(path):
  # Load
  ENCODER = "resnet50"
  ENCODER_WEIGHTS = "imagenet"
  DEVICE = "cuda"

  ACTIVATION = None
  model = smp.Unet(encoder_name = ENCODER,encoder_weights = ENCODER_WEIGHTS,classes = 4,activation = ACTIVATION)
  model.load_state_dict(torch.load(path))
  model.eval()
  return model

def test(args):
    ## write test data loaders and import thresholding from postprocess script
    ## Use valid dataloader for thresholding and finding optimal value
    model_path = "../saved/" # CHANGE THIS!!
    ENCODER = "resnet50"
    ENCODER_WEIGHTS = "imagenet"
    DEVICE = "cuda"
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER,ENCODER_WEIGHTS)
    test_dataset = CloudDataSet(df=sub, datatype='test', img_ids=test_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    model = load_model("/content/drive/MyDrive/clouds-segmentation-dataset/cloud_segmentation_model/model_smp.pth")

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    loaders = {"test": test_loader}
    find_optimal_values()
    encoded_pixels = []
    image_id = 0
    for i, test_batch in enumerate(loaders['test']):
        runner = SupervisedRunner(model = model)
        # runner = runner.predict_loader(model = model)
        runner_out = runner.predict_batch({"features": test_batch[0]})['logits'] ##  change this!!!

        for i, batch in enumerate(runner_out):
            for probability in batch:

                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
                if num_predict == 0:
                    encoded_pixels.append('')
                else:
                    r = mask2rle(predict)
                    encoded_pixels.append(r)
                image_id += 1
    
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)


    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='kaggle-clouds-segmentation-challenge')
    parser.add_argument('-testds_path', '--dataset_path', default="~/Downloads", type=str,
                      help='Input the path to the kaggle clouds segmentation dataset')
    args = parser.parse_args()

    test(args)
