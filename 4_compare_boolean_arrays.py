import sys,os
import pandas as pd
import numpy as np
from pathlib import Path

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

subjects = [
 
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
thresholds= ['_height_threshold_0.001']
L2_contrasts=['_level2FDRthreshold1','_level2FDRthreshold2' ,'_level2FDRthreshold1','_level2FDRthreshold2','_level2FDRthreshold3','_level2FDRthreshold4','_level2FDRthreshold5','_level2FDRthreshold6']
events=['correct_v_none','incorrect_v_none','correct_v_incorrect']

for subject in subjects:
	for L1_contrast in L1_contrasts:
		for threshold in thresholds: 
			for L2_contrast in L2_contrasts:
				for event in events:
					for i in range(1,4+1,1):
						try:
							ROI='%s'%i
							directory1='/Volumes/ActiveStorage-11TB/FER.01/Analysis/nilearn/'+subject+'/roi_transforms/'+L1_contrast+'/'+threshold+'/'+L2_contrast+'/__boolean_matrix/'
							if not os.path.exists(directory1):
								Path(directory1).mkdir(parents=True, exist_ok=True)
					


							path2file1='/Volumes/ActiveStorage-11TB/FER.01/Analysis/nilearn/'+subject+'/roi_transforms/'+L1_contrast+'/'+threshold+'/'+L2_contrast+'/_'+ROI+'_y_test_accuracy_'+event+'.txt'
							path2file2='/Volumes/ActiveStorage-11TB/FER.01/Analysis/nilearn/'+subject+'/roi_transforms/'+L1_contrast+'/'+threshold+'/'+L2_contrast+'/_'+ROI+'_y_pred_accuracy_'+event+'.txt'

							m1=pd.read_csv(path2file1,delimiter=',')
							m2=pd.read_csv(path2file2,delimiter=',')

							df=pd.concat(
								[

								m1,m2 

								],

								axis=1,
								join="outer",
								ignore_index=True,
								keys=None,
								levels=None,
								names=None,
								verify_integrity=False,
								copy=True,
								)

							df.columns=[
										'y_test',
										'y_pred',
	
										]
					
							if event == 'correct_v_none':
					
								total_available_hit=[(df["y_test"].isin(["correct"]))]
								total_available_none=[(df["y_test"].isin(["none"]))]
								total_hit_count=np.count_nonzero(total_available_hit) #the number of available 'correct' volumes
								total_none_count=np.count_nonzero(total_available_none)	#the number of available 'none' volumes

								true_hit 		= [(df["y_test"].isin(["correct"])) & (df["y_pred"].isin(["correct"]))] #the number of true hits (called correct correct)
								true_none		= [(df["y_test"].isin(["none"])) & (df["y_pred"].isin(["none"]))]		#the number of true nones (called none none)
								type_1_error 	= [(df["y_test"].isin(["none"])) & (df["y_pred"].isin(["correct"]))]	#the number of false positives (called none correct)
								type_2_error 	= [(df["y_test"].isin(["correct"])) & (df["y_pred"].isin(["none"]))]	#the number of false negatives (called correct none)

							if event == 'incorrect_v_none':
					
								total_available_hit=[(df["y_test"].isin(["correct"]))]
								total_available_none=[(df["y_test"].isin(["none"]))]
								total_hit_count=np.count_nonzero(total_available_hit) #the number of available 'incorrect' volumes
								total_none_count=np.count_nonzero(total_available_none)	#the number of available 'none' volumes

								true_hit 		= [(df["y_test"].isin(["incorrect"])) & (df["y_pred"].isin(["incorrect"]))] #the number of true hits (called incorrect incorrect)
								true_none		= [(df["y_test"].isin(["none"])) & (df["y_pred"].isin(["none"]))]		#the number of true nones (called none none)
								type_1_error 	= [(df["y_test"].isin(["none"])) & (df["y_pred"].isin(["incorrect"]))]	#the number of false positives (called none incorrect)
								type_2_error 	= [(df["y_test"].isin(["incorrect"])) & (df["y_pred"].isin(["none"]))]	#the number of false negatives (called incorrect none)

							if event == 'correct_v_incorrect':
					
								total_available_hit=[(df["y_test"].isin(["correct"]))]
								total_available_none=[(df["y_test"].isin(["incorrect"]))]
								total_hit_count=np.count_nonzero(total_available_hit) #the number of available 'correct' volumes
								total_none_count=np.count_nonzero(total_available_none)	#the number of available 'incorrect' volumes

								true_hit 		= [(df["y_test"].isin(["correct"])) & (df["y_pred"].isin(["correct"]))] 		#the number of true hits (called correct correct)
								true_none		= [(df["y_test"].isin(["incorrect"])) & (df["y_pred"].isin(["incorrect"]))]		#the number of true nones (called incorrect incorrect)
								type_1_error 	= [(df["y_test"].isin(["incorrect"])) & (df["y_pred"].isin(["correct"]))]		#the number of false positives (called incorrect correct)
								type_2_error 	= [(df["y_test"].isin(["correct"])) & (df["y_pred"].isin(["incorrect"]))]		#the number of false negatives (called correct incorrect)

					
					
							true_hit_count = np.count_nonzero(true_hit)
							true_none_count = np.count_nonzero(true_none)

							type_1_error_count = np.count_nonzero(type_1_error)
							type_2_error_count = np.count_nonzero(type_2_error)

							total_events=true_hit_count+true_none_count+type_1_error_count+type_2_error_count  		#the total number of volumes in the series

							true_hit_rate=(true_hit_count/total_hit_count)*100
							true_none_rate=(true_none_count/total_none_count)*100
							false_positive_rate=((total_hit_count-true_hit_count)/total_hit_count)*100
							false_negative_rate=((total_none_count-true_none_count)/total_none_count)*100

				
							print('true hit',		true_hit_count,						'out of',		total_hit_count,	'or ',		true_hit_rate ,		'of "hit" events were called "hit"',event)
							print('true_none',		true_none_count,					'out of',		total_none_count,	'or ', 		true_none_rate,		'of "none" events were called "none"',event)
							print('type_1_error',	(total_hit_count-true_hit_count),	'out of',  		total_hit_count,	'or ', 		false_positive_rate,'of "hit" events were called "none"',event)
							print('type_2_error',	(total_none_count-true_none_count),	'out of',		total_none_count,	'or ', 		false_negative_rate,'of "none" events were called "hit"',event)
							print('\n')
							print((true_hit_count/total_events)*100,'% of all events were true hits')
							print((true_none_count/total_events)*100,'% of all events were true nones')

							#print(((type_1_error_count/total_events))*100,'% of all events were type 1 error / false positives')
							#print(((type_2_error_count/total_events))*100,'% of all events were type 2 error / false negatives')
						
							true_hit_percentage=true_hit_rate
							true_none_percentage=true_none_rate
							type_1_error_percentage=false_positive_rate
							type_2_error_percentage=false_negative_rate					
						
							true_hit_file = open(directory1+'/'+ROI+'true_hit_percentage_'+event+'.txt',"w+")
							true_hit_file.write("%s"%true_hit_percentage)
							true_hit_file.truncate() #dont append file, just overwrite. 
							true_hit_file.close()
						
							true_none_file= open(directory1+'/'+ROI+'true_none_percentage_'+event+'.txt',"w+")
							true_none_file.write("%s"%true_none_percentage)
							true_none_file.truncate() #dont append file, just overwrite. 
							true_none_file.close()
												
							type_1_error_file=open(directory1+'/'+ROI+'type_1_error_percentage_'+event+'.txt',"w+")
							type_1_error_file.write("%s"%type_1_error_percentage)
							type_1_error_file.truncate() #dont append file, just overwrite. 
							type_1_error_file.close()

							type_2_error_file=open(directory1+'/'+ROI+'type_2_error_percentage_'+event+'.txt',"w+")
							type_2_error_file.write("%s"%type_2_error_percentage)
							type_2_error_file.truncate() #dont append file, just overwrite. 
							type_2_error_file.close()

							'''
							#sanity check
							print(total_events)	
							confirm_total_percent=(((true_hit_count/total_events)*100)+ ((true_none_count/total_events)*100)+((type_1_error_count/total_events)*100)+((type_2_error_count/total_events)*100))
							print(confirm_total_percent)
							'''
						except Exception as e: print(e,event,subject,ROI)
							