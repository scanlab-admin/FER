import nilearn, sys, os, pathlib
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.image import mean_img
from nilearn.image import index_img
from nilearn.decoding import Decoder
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.input_data import NiftiMasker

#~~~~~~contrast guide ~~~~~~~~~~~~~~~~~~~~~
#_contrasts_1 #all_correct
#_contrasts_2 #all_incorrect
#_contrasts_3 #correct > incorrect
#_contrasts_4 #incorrect > correct

#_height_threshold_0.05 #p<0.05
#_height_threshold_0.01 #p<0.01
#_height_threshold_0.001 #p<0.001

#_level2FDRthreshold0 #con group mean
#_level2FDRthreshold1 #con group mean NEGATIVE
#_level2FDRthreshold2 #asd group mean 
#_level2FDRthreshold3 #asd group mean NEGATIVE
#_level2FDRthreshold4 #con > asd 
#_level2FDRthreshold5 #asd > con
#_level2FDRthreshold6 #2 group mean
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


 
 			'EMO0001',
 			'EMO0002',
 			'EMO0003',
 			'EMO0004',
 			'EMO0006',
			'EMO0007',
			'EMO0008',
			'EMO0009',
			'EMO0010',
			'EMO0012',
			'EMO0013',
			'EMO0016',
			'EMO0017',
			'EMO0018',
			'EMO0019',
			'EMO0021', 
 			'EMO0022',
 			'EMO0024', 
 			'EMO0025', 
 			'EMO0026',  
 			'EMO0027',
 			'FER0001',
 			'FER0002',
 			'FER0003',
 			'FER0004',
 			'FER0005',
 			'FER0006',
 			'FER0007',
 			'FER0008',
 			'FER0009',
 			'FER0010',
 			'FER0011',
 			'FER0012',
 			'FER0013',
 			'FER0014',
 			'FER0015', 
 			'FER0016',
 			'FER0017',
 			'FER0018',
 			'FER0019',
 			'FER0020',
 				
				]

L1_contrasts=['_contrasts_1','_contrasts_2' ,'_contrasts_3' ,'_contrasts_4'] 
thresholds= ['_height_threshold_0.05','_height_threshold_0.001']
L2_contrasts=['_level2FDRthreshold1','_level2FDRthreshold2' ,'_level2FDRthreshold1','_level2FDRthreshold2','_level2FDRthreshold3','_level2FDRthreshold4','_level2FDRthreshold5','_level2FDRthreshold6']

for subject in subjects:
	for L1_contrast in L1_contrasts:
		for threshold in thresholds: 
			for L2_contrast in L2_contrasts:
				for i in range(25,38+1,1):
					
					ROI_number= "%s"%i
					path2data='/Volumes/ActiveStorage-11TB/FER.01/Analysis/nilearn/%s/'%subject
					
					file = pathlib.Path(path2data+'/roi_transforms/' + L1_contrast + '/' + threshold + '/' + L2_contrast + '/_' +ROI_number+'_y_pred_accuracy_correct_v_incorrect.txt')
	
					if file.exists ():
						print (file)
					
					#comment out the "else" statement to overwrite. 
					#else:
							
					try: 
						
						print (subject,L1_contrast,threshold,L2_contrast,ROI_number)
						#~~~~~~~~~ Load roi data ~~~~~~~~~~~~~~~~~~~~~~~~

						#this is the ROI number in subject-space that you want to probe. Get this number in freeview by examining the group-level ennumerated cluster map such as this one:
						#/Volumes/ActiveStorage-11TB/FER.01/Analysis/nilearn/L2_contrasts/_contrasts_1/_height_threshold_0.05/_level2FDRthreshold0/ennumerated_cluster.nii.gz
						#when automated, this script will cycle through all the L2 rois for that group contrast that are now in subject space. 
						mask_filename =path2data+'/roi_transforms/' + L1_contrast + '/' + threshold + '/' + L2_contrast + '/' +ROI_number+'_in_'+subject+'_funcspace_bin_masked.nii.gz'

						#~~~~~~~~~ Load anatomical data ~~~~~~~~~~~~~~~~~~~~~~~~

						anat='/Volumes/ActiveStorage-11TB/FER.01/Analysis/ANTS/'+'%s_brain.nii.gz'%subject

						#~~~~~~~~~ Load fmri data ~~~~~~~~~~~~~~~~~~~~~~~~

						func_train= '/Volumes/ActiveStorage-11TB/FER.01/Analysis/nipype/l1pipeline_HDR_2_sec_w_both_derivs_2021/_subject_id_%s/realign/rf_st.nii'%subject
						func_test= '/Volumes/ActiveStorage-11TB/FER.01/Analysis/nipype/l1pipeline_HDR_2_sec_w_both_derivs_2021_run2/_subject_id_%s/realign/rf_st.nii'%subject

						#~~~~~~~~~ Load behavioral data ~~~~~~~~~~~~~~~~~~~~~~~~

						session_target_train=path2data+'/run1/'+'parsed.txt'
						session_target_test=path2data+'/run2/'+'parsed.txt'

						behavioral_train = pd.read_csv(session_target_train, delimiter=' ')
						behavioral_test = pd.read_csv(session_target_test, delimiter=' ')

						conditions_train = behavioral_train['labels']
						conditions_test = behavioral_test['labels']

						# Restrict to desired conditions
						condition_mask_train = conditions_train.isin(['correct','incorrect'])
						condition_mask_test = conditions_test.isin(['correct','incorrect'])

						# Split data into train and test samples, using the chunks. Set to <=2 if using all data. 
						condition_mask_train = (condition_mask_train) & (behavioral_train['chunks'] <= 2)
						condition_mask_test = (condition_mask_test) & (behavioral_test['chunks'] <=2)

						# Apply this sample mask to X (fMRI data) and y (behavioral labels)
						X_train = index_img(func_train, condition_mask_train)
						X_test = index_img(func_test, condition_mask_test)
						y_train = conditions_train[condition_mask_train].values
						y_test = conditions_test[condition_mask_test].values

						#~~~~~specify predictive model ~~~~~~~~~~

						from nilearn.decoding import FREMClassifier
						decoder = FREMClassifier(
										mask=mask_filename,
										estimator='svc',
										cv=10,
										smoothing_fwhm=5,
										#n_jobs=2,
										scoring='balanced_accuracy',
										verbose=1,
										)
		
						# Fit model on train data and predict on test data
						decoder.fit(X_train, y_train)
						y_pred = decoder.predict(X_test)
						accuracy = (y_pred == y_test).mean() * 100.
						
						print(y_test)
						print(y_pred)

						#~~~~~assess chance levels~~~~~
						from sklearn.model_selection import cross_val_score
						from sklearn.dummy import DummyClassifier

						null_cv_scores = cross_val_score(
		
											DummyClassifier(), 
											y_test, 
											y_pred, 
											cv=10,
											scoring='balanced_accuracy'
			
											)
						'''
						#~~~~~~~ plot the svc weights ~~~~~~~~~~~~~~~~~~~

						#turn this off if running in a loop. Will produce a bunch of images that we dont need. 
						#turn this on, if inspecting the plausibility of the masks and svc weights. 

						#Compute the mean epi to be used for the background plotting
						from nilearn.image import mean_img
						background_img = mean_img(func_test)

						plotting.plot_roi(
							mask_filename, 
							bg_img=anat, 
							black_bg=False,
							title="L2 binary mask",
							cmap='Paired')

						plotting.plot_stat_map(
							coef_img_correct, 
							bg_img=background_img,
							threshold=0.01, 
							#vmax=6, 
							colorbar=True, 
							black_bg=False,
							cmap='inferno', 
							title="FREM: accuracy %g%%, 'face coefs'"%accuracy,)

						'''
						#~~~~~~~  print and store results  ~~~~~~~~~~~~~~~~~~~

						print("Chance level accuracy: {:.3f}".format(null_cv_scores.mean() *100))
						print("FREM classification accuracy : %g%%" % accuracy)

						#adjust the name of the output file, depending on the conditions being examined. 
						y_pred_file=path2data+'/roi_transforms/' + L1_contrast + '/' + threshold + '/' + L2_contrast + '/_' +ROI_number+'_y_pred_accuracy_correct_v_incorrect.txt'
						y_test_file=path2data+'/roi_transforms/' + L1_contrast + '/' + threshold + '/' + L2_contrast + '/_' +ROI_number+'_y_test_accuracy_correct_v_incorrect.txt'
						np.savetxt(y_pred_file,y_pred,delimiter=',',fmt='%s')
						np.savetxt(y_test_file,y_test,delimiter=',',fmt='%s')
	
			
					except Exception as e: print(e)