import cv2 as cv
import numpy as np
import pickle
import helper

# Read the ROI boxes' coordinates
xml_file_roi = 'ROI_annotations.xml'
image_id = "35"
roi_boxes = helper.load_ROIs(xml_file_roi, image_id)

# Read the ground truth data for evaluation 
xml_file_gt = 'annotations.xml'
ground_truth_data = helper.load_ground_truth(xml_file_gt)

#load video
video = cv.VideoCapture('cafe.mp4')
framerate = video.get(cv.CAP_PROP_FPS)

# image differencing > foreground + data
# determine foreground to ROI overlap

# Initialize a dictionary to store predictions
predictions = {}
frame_counter = 0
occupied_status = np.zeros((len(roi_boxes),))
threshold = 0.20 #threshold for IOU

#KNN model
backSub = cv.createBackgroundSubtractorKNN()
backSub.setHistory(200)

while True:
	ret, frame = video.read()

	if ret:
		#update the background model
		fgMask = backSub.apply(frame)
		
		# adapative filter (using Wiener Filtering) on the foreground mask
		noise_var = 5 # this is adjustable based on the noise characteristics
		fgMask_af = helper.adaptive_filter(fgMask, (5,5), noise_var)
		fgMask_af = cv.threshold(fgMask_af, 5, 255, cv.THRESH_BINARY)[1]

		#add contours to increase area of foreground objects
		contours, hierarchy = cv.findContours(fgMask_af, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
		cv.drawContours(fgMask_af, contours, -1, (0,255,0), 3)

		#find foreground IOU with ROI boxes to determine occupancy status
		#count occupied ROI boxes in video
		
		fgMask_display = cv.cvtColor(fgMask_af, cv.COLOR_GRAY2BGR)
		occupied_status = np.zeros((len(roi_boxes),))
		for i, box in enumerate(roi_boxes):
			coverage = helper.compute_iou(box, fgMask_af)
	
			#update display frame
			cv.rectangle(fgMask_display, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 1)
	
			#label the frame
			label = str(i) + ":  %.4f" % coverage
			(w, h), _ = cv.getTextSize(
					label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)
			cv.rectangle(fgMask_display, (int(box[0]), int(box[1]) - 20), (int(box[0]) + w, int(box[1])), (75,154,255), -1)
			cv.putText(fgMask_display, label, (int(box[0]), int(box[1])),
						cv.FONT_HERSHEY_SIMPLEX, 0.6, (175,50,50), thickness=1)
	
			if coverage >= threshold:
				occupied_status[i] = 1
				cv.rectangle(fgMask_display, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 1)
			else:
				cv.rectangle(fgMask_display, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 1)

		# Store the occupancy status for the current frame
		predictions[frame_counter] = occupied_status.copy()

		num_occupied = np.count_nonzero(occupied_status)
		display_status = "Occupied seats: " + str(num_occupied)
		cv.rectangle(fgMask_display, (10, 2), (200,20), (255,255,255), -1)
		cv.putText(fgMask_display, display_status, (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
	
		thresh_display = "Threshold: " + str(threshold)
		cv.rectangle(fgMask_display, (210, 2), (350,20), (255,255,255), -1)
		cv.putText(fgMask_display, thresh_display, (215, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

		cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
		cv.putText(frame, str(video.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

		frame_counter += 1
		cv.imshow('original', frame)
		#cv.imshow('foreground', fgMask)
		cv.imshow('adaptive', fgMask_display)
  
		cv.waitKey(1)
  
	else:
		break


# save prediction dictionary as pickle file
with open('prediction_dict.pkl', 'wb') as file:
    pickle.dump(predictions, file)
    
    
#evaluate performance
confusion_matrices, accuracies, overall_accuracy = helper.evaluate_performance(ground_truth_data, predictions)
#plot confusion matrix for each ROI 

# Initialize an empty matrix for the overall confusion matrix
overall_conf_matrix = np.zeros_like(confusion_matrices[0])


for i, matrix in enumerate(confusion_matrices):
    #helper.plot_and_save_confusion_matrix(matrix, i)
    overall_conf_matrix += matrix

for i, accuracy in enumerate(accuracies):
    print(f"ROI #{i+1} Accuracy: {accuracy}")
print(f"Overall System Accuracy: {overall_accuracy}")
# Plot and save the overall confusion matrix
helper.plot_and_save_confusion_matrix(overall_conf_matrix, 'overall')

