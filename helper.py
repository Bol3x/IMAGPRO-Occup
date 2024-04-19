import numpy as np
from sklearn.metrics import confusion_matrix
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns

def adaptive_filter(image, kernel_size, noise_variance):
  '''
  Applies a weiner filter to input image
  '''

  # Generate a kernel with the specified size
  kernel = np.ones(kernel_size, dtype=np.float32) / (kernel_size[0] * kernel_size[1])

  #calculate the Fourier Transform of both the input image and the kernel
  ft_image = np.fft.fft2(image)
  ft_kernel = np.fft.fft2(kernel, s=image.shape)

  #calculate the Wiener filter using the two values obtained above including the assigned noise variance
  filter = np.conj(ft_kernel) / (np.abs(ft_kernel) ** 2 + noise_variance)

  #apply Wiener filter in frequency domain
  ft_restored = filter * ft_image

  #perform the Inverse Fourier Transform to get the filtered image (back to the time domain)
  restored_image = np.fft.ifft2(ft_restored)
  restored_image = np.abs(restored_image)

  return np.uint8(restored_image)

def compute_iou(box, foreground):
	'''
	Computes the are covered by the foreground inside the box.
	'''
    
	area = (int(box[2])-int(box[0])) * (int(box[3])-int(box[1]))

	count = 0

	#loop over the box's coordinates
	for i in range(int(box[1]), int(box[3])):
		for j in range(int(box[0]), int(box[2])):
			#check if there is a foreground object via pixels
			#opencv produces binary images by literally having a greyscale image be either 0 or 255 in value
			if foreground[i][j] > 0:
				count = count + 1


	return count / area


def load_ROIs(xml_file, image_id):
    '''
    Load seat box locations as boxes (xyxy)
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    roi_boxes = []
    
    # This finds the image with the specified frame/image id
    image = root.find(f".//image[@id='{image_id}']")
    if image is not None:
        # Extract all the 'box' elements within this image
        for box in image.findall('.//box[@label="chair"]'):
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            roi_boxes.append((xtl, ytl, xbr, ybr))
    
    return roi_boxes


def load_ground_truth(xml_file):
	'''
	Load ground truth of box occupancy status to evaluate system accuracy
 	'''
  
	# Parse the XML file
	tree = ET.parse(xml_file)
	root = tree.getroot()

	# Initialize a dictionary to hold the ground truth data
	ground_truth = {}

	# Iterate over each 'track' element in the XML
	for track in root.findall('.//track'):
		# Iterate over each 'box' element within each 'track'
		for box in track.findall('box'):
			frame = int(box.get('frame'))
			roi_id = int(track.get('id'))  # Assuming 'id' attribute of 'track' is used as ROI ID
			occluded = int(box.get('occluded'))
			
			# If the frame is not in the dictionary, add it
			if frame not in ground_truth:
				ground_truth[frame] = np.zeros(22)  # Assuming 22 ROIs
			
			# Update the occupancy status for the ROI in the current frame
			ground_truth[frame][roi_id] = int(occluded)

	return ground_truth



def evaluate_performance(ground_truth, predictions):
    '''
    Function to evaluate the system's performance for each ROI
    '''
    
    # Initialize lists to store confusion matrices and accuracies for each ROI
    conf_matrices = []
    accuracies = []
    
    # Calculate confusion matrix and accuracy for each ROI
    for roi_id in range(len(ground_truth[0])):
        true_labels = [frame[roi_id] for frame in ground_truth.values()]
        pred_labels = [frame[roi_id] for frame in predictions.values()]
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
        
        conf_matrices.append(conf_matrix)
        accuracies.append(accuracy)
    
    # Calculate overall accuracy
    all_true_labels = [status for frame in ground_truth.values() for status in frame]
    all_pred_labels = [status for frame in predictions.values() for status in frame]
    overall_accuracy = np.sum(np.array(all_true_labels) == np.array(all_pred_labels)) / len(all_true_labels)
    
    return conf_matrices, accuracies, overall_accuracy



def plot_and_save_confusion_matrix(conf_matrix, roi_id):
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if type(roi_id) == str:
        plt.title(f'Confusion Matrix for ROI')
        plt.savefig(f'conf_matrix_roi_{roi_id}.jpeg', format='jpeg')
    else:
        plt.title(f'Confusion Matrix for ROI {roi_id + 1}')
        plt.savefig(f'conf_matrix_roi_{roi_id+1}.jpeg', format='jpeg')
    plt.show()
    

    # Extracting true positives, etc. from the confusion matrix
    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')