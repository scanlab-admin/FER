

"""
FER 2nd level onesample T test (Nipype 0.7)
- Level 1 in subjects own functional space
- Coregister output to freesurfer anatomy
- ANTS normalization done offline (ANTS_batch.sh, WIMT_batch.sh)
- Level 2 using ANTS normalized con images

Created:		05-24-2021	# based on Domain pipeline script L2 (J.A.R.)
Code Revised:	??-??-????
"""

import os                                    # system functions
import nipype.algorithms.modelgen as model   # model generation
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.io as nio           # i/o routines
import nipype.interfaces.matlab as mlab      # how to run matlab
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
#from nipype.utils.filemanip import loadflat	 # some useful stuff for debugging
import scipy.io as sio
import numpy as np
from nipype.interfaces.base import Bunch
from copy import deepcopy
import sys

#uncomment to turn on verbose logging for debugging
# from nipype import config, logging
# config.enable_debug_mode()
# logging.update_logging(config)

###### CONFIGURABLE INPUTS ######

experiment 	='FER.01' 
between_groups = True

			
subjects_list = [
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
cons	=		[
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
patients=		[	
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
				'EMO0026',
				'EMO0027',
				]

#################################

#indicate group1 and group 2, change when appropriate
myGroup1 = cons  
myGroup2 = patients

#from nipype.utils import config
#config.set('execution', 'remove_unnecessary_outputs', 'false')
#config.enable_debug_mode()

# Tell freesurfer what subjects directory to use
subjects_dir = '/Volumes/ActiveStorage-11TB/%s/Analysis/nipype/'%experiment
fs.FSCommand.set_default_subjects_dir(subjects_dir)

# Set the way matlab should be called
mlab.MatlabCommand.set_default_matlab_cmd("/Volumes/Untitled/Applications/MATLAB_R2016a.app/bin/matlab -nodesktop -nosplash")
#If SPM is not in your MATLAB path you should add it here
mlab.MatlabCommand.set_default_paths('/Volumes/ActiveStorage-11TB/packages/spm12/')
# Set up how FSL should write nifti files:
fsl.FSLCommand.set_default_output_type('NIFTI')


def ordersubjects(files, subj_list):
    import sys
    outlist = []
    for s in subj_list:
        subj_found = False
        for f in files:
	    #print f
            if '%s'%s in f:
                outlist.append(f)
                subj_found = True
                continue
        if subj_found == False:
            # Fail hard if expected con images are missing
            sys.stderr.write("Con images for subject %s could not be found!"%(s))
            sys.exit("Con images for subject %s could not be found!"%(s))
    print ('===============',outlist)
    return outlist

def list2tuple(listoflist):
    return [tuple(x) for x in listoflist]


"""
Level 2 Pipeline -- ANTS normalized anatomy and con images
"""

#initialize the 2nd level pipeline
l2pipeline = pe.Workflow(name='l2pipeline_HDR_2_sec_w_both_derivs_between_groups_2021')

# Input node for second level (group analysis) pipeline
l2inputnode = pe.Node(

	interface					=util.IdentityInterface(fields=['contrasts']),
	
				iterables 		= [('contrasts', range(1,4+1))], #range of the first level contrasts
				name			='inputnode')


# Source information for group analysis data
l2source = pe.Node(

		interface=nio.DataGrabber(

			infields			=['l1con_id'],
			outfields			=['l1con']),
			
					name		='l2source')
					
					
l2source.inputs.base_directory = os.path.abspath('/Volumes/ActiveStorage-11TB/%s/Analysis/nipype/l1pipeline_HDR_2_sec_w_both_derivs_2021/'%experiment)
l2source.inputs.template = '*'
l2source.inputs.sort_filelist=True
l2source.inputs.field_template = dict(l1con='/Volumes/ActiveStorage-11TB/FER.01/Analysis/nipype/l1pipeline_HDR_2_sec_w_both_derivs_2021/_subject_id_*/warp_T/mapflow/_warp_T*/spmT_%04d_out_warped_wimt.nii')
l2source.inputs.template_args = dict(l1con=[['l1con_id']])

if between_groups == True:
	# setup a 2-sample t-test
	twosamplettestdes = pe.Node(interface=spm.TwoSampleTTestDesign(), name="twosamplettestdes")
	twosamplettestdes.inputs.explicit_mask_file = os.path.abspath('/Volumes/ActiveStorage-11TB/%s/Analysis/ANTS/MNI152_T1_1mm_brain_uncompressed.nii'%experiment)
	l2estimate = pe.Node(interface=spm.EstimateModel(), name="level2estimate")
	l2estimate.inputs.estimation_method = {'Classical' : 1}

	l2conestimate = pe.Node(interface = spm.EstimateContrast(), name="level2conestimate")
	L2cont1 = ('Group1 Mean','T', 		['Group_{1}','Group_{2}'],	[1,0])
	L2cont2 = ('Group1 -Mean','T', 		['Group_{1}','Group_{2}'],	[-1,0])
	L2cont3 = ('Group2 Mean','T', 		['Group_{1}','Group_{2}'],	[0,1])
	L2cont4 = ('Group2 -Mean','T', 		['Group_{1}','Group_{2}'],	[0,-1])
	L2cont5 = ('Group1 > Group2','T', 	['Group_{1}','Group_{2}'],	[1,-1])
	L2cont6 = ('Group2 > Group1','T', 	['Group_{1}','Group_{2}'],	[-1,1])
	L2cont7 = ('Group1+Group2 Mean','T',['Group_{1}','Group_{2}'],	[0.5,0.5])
	l2conestimate.inputs.contrasts = [L2cont1, L2cont2, L2cont3, L2cont4, L2cont5, L2cont6, L2cont7]
	l2conestimate.inputs.group_contrast = True

if between_groups == False:
	
	# setup a 1-sample t-test
	onesamplettestdes = pe.Node(interface=spm.OneSampleTTestDesign(), name="onesamplettestdes")
	onesamplettestdes.inputs.explicit_mask_file = os.path.abspath('/Volumes/ActiveStorage-11TB/%s/Analysis/ANTS/MNI/MNI152_T1_1mm_brain_uncompressed.nii'%experiment)
	
	l2estimate = pe.Node(interface=spm.EstimateModel(), name="level2estimate")
	l2estimate.inputs.estimation_method = {'Classical' : 1}

	l2conestimate = pe.Node(interface = spm.EstimateContrast(), name="level2conestimate")
	L2cont1 = ('Group','T', ['mean'],[1])
	L2cont2 = ('Group','T', ['mean'],[-1])
	
	l2conestimate.inputs.contrasts = [L2cont1,L2cont2]
	l2conestimate.inputs.group_contrast = True
	


l2FDRthresh = pe.MapNode(interface = spm.Threshold(), name="level2FDRthreshold", iterfield = ['stat_image','contrast_index'])
l2FDRthresh.iterables = [('height_threshold', [0.05, 0.01, 0.001, 0.0001,0.00001])]
l2FDRthresh.inputs.extent_fdr_p_threshold = 0.05
l2FDRthresh.inputs.extent_threshold = 20
l2FDRthresh.inputs.contrast_index = [1,2,3,4,5,6,7] #the group level contrasts
l2FDRthresh.inputs.use_fwe_correction = False
l2FDRthresh.inputs.use_topo_fdr = True

l2pipeline.base_dir = os.path.abspath('/Volumes/ActiveStorage-11TB/%s/Analysis/nipype/'%experiment)

if between_groups is True:

	l2pipeline.connect([
						(l2inputnode,l2source,[('contrasts','l1con_id')]),
						(l2source,twosamplettestdes,[(('l1con',ordersubjects,myGroup1),'group1_files'),
													(('l1con',ordersubjects,myGroup2),'group2_files')]),
													
						(twosamplettestdes,l2estimate,[('spm_mat_file','spm_mat_file')]),
						(l2estimate,l2conestimate,[('spm_mat_file','spm_mat_file'),
													('beta_images','beta_images'),
													('residual_image','residual_image')]),
						(l2conestimate,l2FDRthresh,[('spm_mat_file','spm_mat_file'),
													('spmT_images','stat_image')]),
			])

if between_groups is False:
	l2pipeline.connect([
						(l2inputnode,l2source,[('contrasts','l1con_id')]),
						(l2source,onesamplettestdes,[('l1con','in_files')]),
						
												
						(onesamplettestdes,l2estimate,[('spm_mat_file','spm_mat_file')]),
						(l2estimate,l2conestimate,[('spm_mat_file','spm_mat_file'),
													('beta_images','beta_images'),
													('residual_image','residual_image')]),
						(l2conestimate,l2FDRthresh,[('spm_mat_file','spm_mat_file'),
													('spmT_images','stat_image')]),
			])


l2pipeline.config['execution'] = {'remove_unnecessary_outputs':'False'}
#l2pipeline.write_graph()
l2pipeline.run()#plugin='MultiProc', plugin_args={'n_procs' : 12})
