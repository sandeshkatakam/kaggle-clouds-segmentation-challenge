# Thresholding the optimal values 
# Plot of Loss Curves
import numpy as np
from train import runner
# Visualizing Results
# Making a submission File

def save_label_masks():
  with open('/content/drive/MyDrive/cloud_pred/mask_ar_valid.txt', 'w') as outfile:
      outfile.write('# Array shape: {0}\n'.format(mask_ar_resized.shape))
      for threeD_data_slice in output_ar_resized:
        for twoD_data_slice in threeD_data_slice:
          np.savetxt(outfile, twoD_data_slice, fmt='%-7.2f')
          outfile.write('# New slice\n')

def save_predicted_masks():
  with open('/content/drive/MyDrive/cloud_pred/output_ar_valid.txt', 'w') as outfile:
      outfile.write('# Array shape: {0}\n'.format(output_ar_resized.shape))

      for threeD_data_slice in output_ar_resized:
          for twoD_data_slice in threeD_data_slice:
              np.savetxt(outfile, twoD_data_slice, fmt='%-7.2f')
              outfile.write('# New slice\n')


## POSTPROCESS
# def perform_thresholding():
#   num_classes = 4
#   class_min_max = []
#   output_ar_resized = sigmoid(output_ar_resized)
#   for j in range(num_classes):
#     class_min_max.append((np.min(output_ar_resized[:,j,:,:]),np.max(output_ar_resized[:,j,:,:])))
#   threshold_class = []
#   avg_dice_score = []
#   for j in range(num_classes):
#     max_dice_score = 0
#     threshold = (class_min_max[j][0] + class_min_max[j][1])/2
#     for t in np.linspace(class_min_max[j][0],class_min_max[j][1],25):
#       dice_score = 0
#       for i in range(len(valid_ids)):
#         prediction,nums = post_process(output_ar_resized[i,j,:,:],t,10)
#         if (np.sum(prediction) == 0 and np.sum(mask_ar_resized[i,j,:,:]) == 0):
#           dice_score += 1
#         else:
#           dice_score += dice(mask_ar_resized[i,j,:,:],prediction)
#       dice_score /= len(valid_ids)
#       if dice_score > max_dice_score:
#         max_dice_score = dice_score
#         threshold = t
#     threshold_class.append(threshold)
#     avg_dice_score.append(max_dice_score)



# def predict_mask_before_thresholding():
#   device = "cpu"
#   loaders = {"infer":valid_loader}
#   runner.predict_loader(model = model,loaders = loaders,callbacks = [CheckpointCallback(logdir = logdir,resume_model= f"{logdir}/checkpoints/best.pth")])
#   image_list = []
#   mask_ar = np.zeros((555,4,320,640))
#   output_ar = np.zeros((555,4,320,640))

#   model = model.to(device)
#   for i,batch in enumerate(tqdm.tqdm(valid_loader)):
#     image,mask = batch
#     image = image.to(device)
#     mask = mask.to(device)
#     output = runner.predict_batch(batch)
#     for j in range(mask.shape[0]):
#       mask_ar[i*bs +j] = mask[j].numpy()
#       output_ar[i*bs + j] = output["logits"][j].cpu().numpy()

# def apply_threshold():
#   threshold_ar = np.zeros(output_ar_resized.shape)
#   for i in tqdm.tqdm(range(output_ar_resized.shape[0])):
#     for j in range(output_ar_resized.shape[1]):
#       threshold_ar[i,j][output_ar_resized[i,j,:,:] > threshold_class[j]] = 1
  

def thresholding():
  encoded_pixels = []
  loaders = {"infer": valid_loader}

  valid_masks = []
  probabilities = np.zeros((2220, 350, 525))
  for i, (batch, output) in enumerate(zip(
          valid_dataset, runner.callbacks[0].predictions["logits"])):
      image, mask = batch
      for m in mask:
          if m.shape != (350, 525):
              m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
          valid_masks.append(m)

      for j, probability in enumerate(output):
          if probability.shape != (350, 525):
              probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
          probabilities[i * 4 + j, :, :] = probability

def find_optimal_values():
  class_params = {}
  for class_id in range(4):
      print(class_id)
      attempts = []
      for t in range(0, 100, 5):
          t /= 100
          for ms in [0, 100, 1200, 5000, 10000]:
              masks = []
              for i in range(class_id, len(output_arr), 4): # Change this!!
                  probability = output_arr[i] # change this!!
                  predict, num_predict = post_process(sigmoid(probability), t, ms) # change this!!
                  masks.append(predict)

              d = []
              for i, j in zip(masks, valid_masks[class_id::4]):
                  if (i.sum() == 0) & (j.sum() == 0):
                      d.append(1)
                  else:
                      d.append(dice(i, j))

              attempts.append((t, ms, np.mean(d)))

      attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])


      attempts_df = attempts_df.sort_values('dice', ascending=False)
      print(attempts_df.head())
      best_threshold = attempts_df['threshold'].values[0]
      best_size = attempts_df['size'].values[0]

      class_params[class_id] = (best_threshold, best_size)
      return class_params





  