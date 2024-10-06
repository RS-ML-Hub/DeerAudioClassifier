import os
import re
import sys
import gc      
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import itertools
import numpy as np
from scipy import signal
from scipy.io import wavfile
from itertools import combinations
from collections import defaultdict
from pydub import AudioSegment	
from operator import itemgetter
from natsort import natsorted   
from datetime import datetime, timedelta

Work_dir_NN = os.getcwd()
Input_Sound_Main_Dir = os.path.join(Work_dir_NN, "Recordings")
#################################################################################
###... <<<  Must Check Parameters   >>>
#... (( exclude very short detected  sound ))..
Threshold_NonDeer = 516                        # 516 *128/44100 = 1.5 Sec   >> The shorter sound will be removed  
Long_Sound_Threshold = 7000                    # 7000 *128/44100 = 20.3 Sec >> The longer sound will be removed  

#... (( Detecting deer sound from spectrum sum ))..	
threshold = 1.25                               # 1.15 Oze, tested 1.25 Taki   >> value to multiply in median of Spectrum sumation

#... (( Low Pass filter  >> 
# VIP Note: Should be applied in Taki to remove field cockroach with high noise
Apply_Low_Pass_Filter = "Yes"                  #  "Yes"   "No"
cutoff_LPF = 3000                              # in hertz for Low Pass filter to remove field cockroach with high noise
No_Repeat_Low_pass_Filter = 2                  # Low-pass filter need to repeated several time to remove noise	

###... <<<  Othere conditions   >>>
Amplify        = "Yes"                  # "Yes"   "No"

Increase_Analysis_Time_Range = "Yes"    # "Yes"   "No"
Time_before         = 0.5               # in sec
Time_after          = 1.5               # in sec

Time_befr_DNN       = 2.7               # in sec
Time_aftr_DNN       = 3.8               # in sec
	
sampling_rate = 44100	                # in hertz  (sampling frequency or sampling rate, fs)	

## High Pass Filter 	
cutoff_HPF = 1000                       # in hertz for High Pass filter to remove white noise
#cutoff_bandstop = [3000, 5000]         # in hertz for Band Stop filter to remove field cockroach with high noise
filter_order = 4	
# Savitzky-Golay smotheen filtering 	
window_len = 511            			# 511
poly_order = 2	

sound_Speed = 340.0	                     # Note: when I tried Sound speed=300m/sec there was no big difference 136.430396 
earth_R     = 6371000.0


#################################################################################
#################################################################################
#################################################################################
###... 


def CreateFolder_(Folder_dir):
	if not os.path.exists(Folder_dir):
		os.makedirs(Folder_dir)


def Main_Code_f(Hourly_dir, Files_List):
	
	#################################################################################
	## "dc" is the main dictionary where the keys are ['POI1', 'POI2', 'POI3', 'POI5']
	#   dc{
	#   "POI1"{
	#		    'fn'							: 'POI1_220911-001_02am.wav' 
	#		    'Rec_duration'					: 3570.70
	#		    'SignalAry'						: array([ 0.,  0.,  0., ..., -1., -2., -3.])
	#		    'SignalAry_filtered'			: array([ 0.,  0.,  0., ..., -1., -2., -3.]) 
	#		    'Score_0_OR_1'					: array([0, 1, 1, ..., 0, 0, 0]) 
	#		    'sum_spectrum'					: array([  0. ,   0.  ,   0.  , ..., 166.61,45.85,  28.53])
	#		    'sum_spectrum_HPFilter'			: array([  0. ,   0.  ,   0.  , ..., 166.61,45.85,  28.53]) 
	#		    'sum_spectrum_HPFilter_smoothen': array([  0. ,   0.  ,   0.  , ..., 166.61,45.85,  28.53])
	#		    'Amplify_dB_tot'				: Amplify_dB_tot		
	#		}
	#		} 	
	dc = {}   # >> keys are ['POI1', 'POI2', 'POI3', 'POI5']
	for fn in Files_List:
		
		#################################################################################
		# Extract date and time from file name
		# >> input file name  > "POI4_181013-003_6am_amplified_exact_14dB_MONO.wav"  >  "POI4_181013-003_6am.wav"
		Date_ = int( "20" + re.search('\d{6}', fn).group(0) ) # 20181013
		Hour_am_pm = re.search('\d*.m', fn).group(0)          # 6am
		
		Hour_12 = datetime.strptime( Hour_am_pm, "%I%p" )     # %I is the 12 hour clock , %p qualifies if it is AM or PM.
		Hour_   = int( datetime.strftime(Hour_12, "%H")  )    # %H is the 24 hour clock, 
			
		#################################################################################
		# create dictionary for each POI
		POIx = fn.split("_")[0]  # POI4
		dc[POIx] = {}
		dc[POIx]["fn"] = fn
		
		
		
	#################################################################################
	#################################################################################
	#################################################################################
	###... <<<  Create output folder and Array to store deer information  >>>
	Output_Hourly_Dir = os.path.join( Hourly_dir, r"Potential_Deer_Clips")
	CreateFolder_(Output_Hourly_Dir) 	# Create output folder if not exist
	
	#################################################################################
	#################################################################################
	#################################################################################
	###... <<<  1st Stage - Signal Processing and Automatically Extract deer Cries  >>>
	print("1st Stage - Signal Processing and Automatically Extract deer Cries")

	## The following function does the following
	##   (1) convert mp3 to wav, (2)convert stero to mono, (3) Amplification, (4) High Bass Filter  
	##   (5) Compute the spectrogram, (6) Apply Savitzky-Golay smotheen filtering
	def Deer_sound_detection_f(Hourly_dir, fn):
		global cutoff_HPF, cutoff_LPF, cutoff_bandstop,  filter_order, window_len, poly_order, threshold
		
		fn_Dir = os.path.abspath( os.path.join (Hourly_dir, fn))
		print(fn_Dir)
		#################################################################################
		# (1) Convert mp3 to wav,  (2) save the wav file into folder, and (3) return the wav file name
		if ".mp3" in fn:
			fn_wav     = re.search('(.+).mp3', fn    ).group(1) + ".wav"	## group(1): to extract file name without .MP3
			fn_Dir_wav = re.search('(.+).mp3', fn_Dir).group(1) + ".wav"	## group(1): to extract file name without .MP3

			# convert mp3 to wav.. Then save wav file 
			sound = AudioSegment.from_mp3(fn_Dir)  ## Please note that "pydub.AudioSegment" required "ffmpeg" to be installed ## we could use anaconada Navigator to install "ffmpeg" package  or "conda install -c conda-forge ffmpeg"
			sound.export(fn_Dir_wav, format="wav")
			del sound
		if ".wav" in fn:
			fn_wav     = fn	
			fn_Dir_wav = fn_Dir
		#################################################################################
		# (2) read input sound data
		samplingFrequency, SignalAry = wavfile.read(fn_Dir_wav)  # samplingFrequency = 44100   SignalAry shape (157468032, 2)
		SignalAry = SignalAry.astype("int16")  # WAV Format: 16-bit integer PCM    # min:-32768  max:+32767
		Rec_duration = SignalAry.shape[0] / float(samplingFrequency)

		# compare input sample rate with the recording sample rate
		if samplingFrequency != sampling_rate:
			print(("Check!! -> input sampling_rate is (%s) but recording sample rate is (%s)" %(sampling_rate, samplingFrequency) )) 
			sys.exit()

		# number of dimension (1 for mono, 2 for stereo)
		n_dimension = SignalAry.ndim

		# number of audio frames
		n_frm_tot_each_Ch = SignalAry.shape[0]
		
		#################################################################################
		# (3) convert stero to mono
		if n_dimension == 2:
			######################
			##.. (( know the channel with more data "higher sample value"))
			sum_abs_Ch1 = np.sum(np.abs(SignalAry[:,0], dtype="float64"), dtype="float64")
			sum_abs_Ch2 = np.sum(np.abs(SignalAry[:,1], dtype="float64"), dtype="float64")
			
			##. ( Select MONO channel that have more data)
			if sum_abs_Ch1 > sum_abs_Ch2:
				SignalAry = SignalAry[:,0] 
				print ("Channel 1 was selected")
			else:
				SignalAry = SignalAry[:,1] 
				print ("Channel 2 was selected")

			SignalAry = SignalAry.astype("float64")  # 

		#################################################################################
		### (4) perform High Pass Filter over the input sound 
		## Design digital filter with N order
		## we should 1st design the filter by using (scipy.signal.butter) --> the outputs of butter are the filter coefficients (a, b) of i4R filter
		## 2nd step is to filter the input sound wav using (scipy.signal.filtfilt)
		## High pass filter will reduce white noise "Tree leaves"
		#cutoff_HPF = 1000                 ## in hertz
		#filter_order = 4            ## order 4 equivalent to -48dB
		b, a = signal.butter(filter_order, cutoff_HPF, btype='highpass', fs = sampling_rate, output='ba') # we should write the "sampling Frequency"
		SignalAry_filtered = signal.filtfilt(b, a, SignalAry).astype("float64")

		#################################################################################
		### (5) perform Low Pass Filter over the input sound 
		## Design digital filter with N order
		## we should 1st design the filter by using (scipy.signal.butter) --> the outputs of butter are the filter coefficients (a, b) of i4R filter
		## 2nd step is to filter the input sound wav using (scipy.signal.filtfilt)
		## low-pass filter will reduce high niose from insects "e.g., field cockroach "
		#cutoff_bandstop = [3000, 5000]    ## in hertz  
		#cutoff_LPF = 3000                 ## in hertz
		#filter_order = 4                  ## order 4 equivalent to -48dB		
		if Apply_Low_Pass_Filter == "Yes":
			b, a = signal.butter(filter_order, cutoff_LPF, btype='lowpass', fs = sampling_rate, output='ba') # we should write the "sampling Frequency"
			#b, a = signal.butter(filter_order, cutoff_bandstop, btype='bandstop', fs = sampling_rate, output='ba') # we should write the "sampling Frequency"
			for ii in range(No_Repeat_Low_pass_Filter):  # # Low-pass filter need to repeated several time to remove noise
				SignalAry_filtered = signal.filtfilt(b, a, SignalAry_filtered).astype("float64")
		
		
		#################################################################################
		## (6) 2nd Amplification 
		## I amplified the Filtered (HPF) signal
		if Amplify == "Yes":

			max_abs_input = np.max( np.abs(SignalAry_filtered))	## to get max positive or negative  # e.g., 3575.7
			Amplify_dB_2 = abs( 20 * np.log10( max_abs_input / 32768.) )       # e.g., 19 dB
			print("2nd amplification after HPF (dB) = %.2f" %(Amplify_dB_2) )
			
			SignalAry_filtered = SignalAry_filtered * 32768.0 / max_abs_input
			

		else:
			Amplify_dB_2 = 0
		
		#Amplify_dB_tot = Amplify_dB_1 + Amplify_dB_2
		Amplify_dB_tot =  Amplify_dB_2
		#################################################################################
		## (7) Compute the spectrogram of of original signal
		Pxx, freqs, times, im = plt.specgram(SignalAry, Fs=sampling_rate, mode = 'magnitude', noverlap=128 , NFFT= 256)	#mode = 'magnitude' , 'psd'
		plt.close("all")    ## we do not need to show the spectrogram
		
		## compute the spectrogram of High Pass Filtered data
		Pxx_filter, freqs_filter, x_time_spec, im_filter = plt.specgram(SignalAry_filtered, Fs=sampling_rate, mode = 'magnitude', noverlap=128 , NFFT= 256)	#mode = 'magnitude' , 'psd'
		plt.close("all")    ## we do not need to show the spectrogram
		## sum the spectrogram intensity 
		sum_spectrum          = np.sum(Pxx[2:,:], axis=0)        # skip the first 2 rows during summation
		sum_spectrum_HPFilter = np.sum(Pxx_filter[2:,:], axis=0) #  skip the first 2 rows during summation

		
		#################################################################################
		## (8) Apply Savitzky-Golay smotheen filtering method
		##savgol_filter(x, window_length, polyorder, axis=-1)
		window_len = window_len   ## 511
		poly_order = poly_order
		sum_spectrum_HPFilter_smoothen = signal.savgol_filter(sum_spectrum_HPFilter, window_length=window_len, polyorder=poly_order, axis=-1)

		#################################################################################
		## (9) Detecting deer sound in the Input sound
		threshold = threshold  # value to multiply in median
		Score_0_OR_1         = np.zeros(len(sum_spectrum_HPFilter_smoothen), dtype="int")
		Median_SavGly_Filter = np.median(sum_spectrum_HPFilter_smoothen)

		## replace very low values of smoothen HPF spectrum to median >>
		##   >> I will used "power of 10 from median"  >> for example, median = 1000 thus any value lower that the 1000/10=100 will be converted to ".95*median"
		sum_spectrum_HPFilter_smoothen[sum_spectrum_HPFilter_smoothen < (Median_SavGly_Filter/10)] = 0.95 * Median_SavGly_Filter
		
		Score_0_OR_1 = np.where(sum_spectrum_HPFilter_smoothen > threshold * Median_SavGly_Filter, 1, 0)

		return (Rec_duration, SignalAry, SignalAry_filtered, Score_0_OR_1, 
					  sum_spectrum, sum_spectrum_HPFilter, sum_spectrum_HPFilter_smoothen, Amplify_dB_tot)

	## the following function receive (0~1) array and return only (1) list
	## 1> means possible deer sound    0> means no deer sound
	def Slice_deer_sound_by_keeping_1_remove_0_f (Score_0_OR_1):
		print(Score_0_OR_1)
		Ones_Indx_Ary = np.nonzero(Score_0_OR_1)[0]   ## nonzero returns the indices of non zero element
		## nonzero return tuples >> Thus, [0] was used to ectract the nonzero array
		## example >> Score_0_OR_1  = [0,0,1,1,1,1,0,0,1,0,0,1,1,1,0] 
		##         >> Ones_Indx_Ary = [ 2,  3,  4,  5,  8, 11, 12, 13]
		print(Ones_Indx_Ary)
		# convert array to list
		Ones_Indx_Lst = Ones_Indx_Ary.tolist()
		## Slice the list of Ones to several lists where the sequence of numbers was interrupted by zeros
		## example >> Ones_Indx_Ary = [ 2,  3,  4,  5,  8, 11, 12, 13] >> Ones_Indx_Lst_Slice [ [2,  3,  4,  5], [8], [11, 12, 13] ]
		Ones_Indx_Lst_Slice = []
		for k, g in itertools.groupby(enumerate(Ones_Indx_Lst), lambda x: x[0]-x[1]):
			Ones_Indx_Lst_Slice.append(list(dict(g).values() ))
		return Ones_Indx_Lst_Slice

	#################################################################################
	#################################################################################
	##... Process Input Signal to perform the following
	##       (1) convert mp3 to wav, (2)convert stero to mono, (3) Amplification, (4) High Bass Filter  
	##       (5) Compute the spectrogram, (6) Apply Savitzky-Golay smotheen filtering	
	for POIx in dc.keys():
		fn = dc[POIx]["fn"] 
		
		(dc[POIx]["Rec_duration"], 
	     dc[POIx]["SignalAry"], 
		 dc[POIx]["SignalAry_filtered"], 
		 dc[POIx]["Score_0_OR_1"],
		 dc[POIx]["sum_spectrum"], 
		 dc[POIx]["sum_spectrum_HPFilter"],
		 dc[POIx]["sum_spectrum_HPFilter_smoothen"], 
		 dc[POIx]["Amplify_dB_tot"]
		 ) = Deer_sound_detection_f(Hourly_dir, fn)
		
		Score_len = dc[POIx]["Score_0_OR_1"].shape[0]
		print(f"Finish processing {POIx} Input Sound recording")
		print("###########################################")
		# Clear memory used by python
		gc.collect()		


	Min_Rec_duration_Rec_all = min( [dc[POIx]["Rec_duration"] for POIx in dc.keys()] ) 	
	#Min_Rec_duration_Rec_all = min(Rec_duration_R1st, Rec_duration_R2nd, Rec_duration_R3rd)

	#################################################################################
	#################################################################################
	## combined the detected deer sound along the three recorders
	## each Score array "e.g., Score_0_OR_1_R1st" has (0 and 1)--> 0>means no deersound   1>means possible deer sound
	## "Score_0_OR_1_Rec_all" combines all "1" accross all recoreders 
	## consider the following example
	## Score_0_OR_1_R1st      = [0,0,0,1,1,1,0,0,0,0,0,1,1,1,0]  --> length = 15
	## Score_0_OR_1_R2nd      = [0,0,0,1,1,1,0,0,1,0,0,1,1,1,0]  --> length = 15
	## Score_0_OR_1_R3rd      = [0,0,1,1,1,1,0,0,0,0,0,1,1,1,0]  --> length = 15
	## Score_0_OR_1_Rec_all   = [0,0,1,1,1,1,0,0,1,0,0,1,1,1,0]  >> Combined all possible deer sound with "1" value 
	
	
	combined = [ dc[POIx]["Score_0_OR_1"] for POIx in dc.keys()]   # append lists to one list
	# combined = [Score_0_OR_1_R1st, Score_0_OR_1_R2nd, Score_0_OR_1_R3rd]
	Score_0_OR_1_Rec_all = np.asarray([max(elem) for elem in zip(*combined)], dtype="int")

	#################################################################################
	## extract 1 only from (0 and 1 array)
	## example >> Score_0_OR_1_Rec_all = [0,0,1,1,1,1,0,0,1,0,0,1,1,1,0] >> Ones_Indx_Lst_Slice_spectrum_128Step = [ [1,1,1,1], [1], [1,1,1]  ]
	Ones_Indx_Lst_Slice_spectrum_128Step = Slice_deer_sound_by_keeping_1_remove_0_f(Score_0_OR_1_Rec_all) 
	print("Initial No of deer sound detected = ", len(Ones_Indx_Lst_Slice_spectrum_128Step))
	for i4 in range (len(Ones_Indx_Lst_Slice_spectrum_128Step) ):
		print( i4+1, "len=%.0f , Start=%.3f , end=%.3f, Dur=%.3f"  %( len(Ones_Indx_Lst_Slice_spectrum_128Step[i4]), Ones_Indx_Lst_Slice_spectrum_128Step[i4][0]*128/44100, Ones_Indx_Lst_Slice_spectrum_128Step[i4][-1]*128/44100, len(Ones_Indx_Lst_Slice_spectrum_128Step[i4])*128/44100 ))
	print("###############################")


	#################################################################################
	## remove the short slices where mainly do not have deer sound
	## example >> if threshold_nonDeer = 2  and   Ones_Indx_Lst_Slice_spectrum_128Step = [ [2,  3,  4,  5], [8], [11, 12, 13] ]  please note that "Ones_Indx_Lst_Slice_spectrum_128Step" return the indices of "1" elements 
	##          >> the second element [8] will be removed as its length < 2
	##          >> Ones_Indx_Lst_Slice_spectrum_128Step = [ [2,  3,  4,  5], [11, 12, 13] ]
	threshold_nonDeer = Threshold_NonDeer    ## 516 *128/44100 = 1.5 Sec >> The short sound will be removed
	for i3 in range( len(Ones_Indx_Lst_Slice_spectrum_128Step) - 1, -1, -1) :
		if ( (len(Ones_Indx_Lst_Slice_spectrum_128Step[i3]) < threshold_nonDeer) or     ## condition to check short time   #e.g., 1.5 sec
		     (len(Ones_Indx_Lst_Slice_spectrum_128Step[i3]) > Long_Sound_Threshold) ):  ## condition to check long time    #e.g., 20 sec
			Ones_Indx_Lst_Slice_spectrum_128Step.pop(i3)   ## remove short time

	No_detected_deer_sound = len(Ones_Indx_Lst_Slice_spectrum_128Step)
	print("Final No of deer sound detected = ", No_detected_deer_sound)
	# if No of Deer Cry is Zero, save empty dataframe and then end 
	if No_detected_deer_sound == 0:
		df_col = ["DeerNo", "Date", "hh", "StartCryTime(Sec)", "EndCtyTime(Sec)", "CryDuration(Sec)", "AnalysisStartTime", "AnalysisEndTime", "DNN_StartTime", "DNN_EndTime", "time_delay_POI1_POI2", "time_delay_POI1_POI3", "time_delay_POI2_POI3", "Lat_DD", "Lon_DD", "lat_tol_m", "lon_tol_m", "lat_tol_deg", "lon_tol_deg", "Dis_Deer_POI1(m)", "Dis_Deer_POI2(m)", "Dis_Deer_POI3(m)", "Dis_Deer_POI4(m)", "Dis_Deer_POI5(m)", "Dis_Deer_POI6(m)", "Dis_Deer_POI7(m)", "Dis_Deer_POI8(m)", "Dis_Deer_POI9(m)", "Dis_Deer_POI10(m)", "time_delay_TryErr_POI1_POI2", "time_delay_TryErr_POI1_POI3", "time_delay_TryErr_POI2_POI3"]
		df = pd.DataFrame( columns=df_col )
		Output_f_n  = os.path.join( Output_Hourly_Dir, f"DeerCryInformation_{Date_}_{Hour_}hh.csv" )
		df.to_csv(Output_f_n, index=False)
		return None
	#################################################################################
	## recreate Score_0_OR_1 array 
	##  example >> if the original length = 15   and  Ones_Indx_Lst_Slice_spectrum_128Step = [ [2,  3,  4,  5], [11, 12, 13] ]  
	##          >> Score_0_OR_1_Rec_all_updated = [0,0,1,1,1,1,0,0,0,0,0,1,1,1,0]
	Score_0_OR_1_Rec_all_updated = np.zeros( Score_len , dtype="int" )
	for i1 in range( len(Ones_Indx_Lst_Slice_spectrum_128Step) ):
		for i2 in (Ones_Indx_Lst_Slice_spectrum_128Step[i1]):
			Score_0_OR_1_Rec_all_updated[i2] = 1

	#################################################################################
	#################################################################################
	#################################################################################
	###... <<<  Saving deer sound clips  >>>
	print("Start Saving deer sound clips")


	##.. function to save deer cry in wav format
	def Save_Deer_Cry_Sliced_Wav_f (wav_ary_slice, Rec_ID, Date_, Hour_, StartCryTime, EndCryTime, DurationCry, Deer_Sound_no, Output_Hourly_Dir):
		
		wav_ary_slice = np.asarray(wav_ary_slice, dtype = "int16")
		#########################################################
		## Saving slice audio sound 
		Output_wav_f_n = os.path.join( Output_Hourly_Dir   , f"DeerSoundNo{round(Deer_Sound_no, 3)}_{Rec_ID}_{Date_}_{Hour_}_S{round(StartCryTime, 2)}_E{round(EndCryTime, 2)}_D{round(DurationCry, 2)}.wav" ) 
		wavfile.write(Output_wav_f_n, sampling_rate, wav_ary_slice)

		
	## Please note that >> (1 index of Spectrum) = (128 index of sound wave) based on (noverlap=128 , NFFT= 256) in the  >> plt.specgram(.... )
	## Saving Deer Sound Clips
	df = pd.DataFrame( index=pd.RangeIndex(start=0, stop=No_detected_deer_sound, step=1) )		

	for i1 in range ( No_detected_deer_sound ):
		Deer_Sound_no = i1+1
		CryLength     = len(Ones_Indx_Lst_Slice_spectrum_128Step[i1])
		CRyDuration   = CryLength * 128 / 44100.
		Start_Index   = (Ones_Indx_Lst_Slice_spectrum_128Step[i1][0])*128
		StartTime     = Start_Index/44100.
		End_Index     = (Ones_Indx_Lst_Slice_spectrum_128Step[i1][-1])*128
		EndTime       = End_Index/44100.

		
		## consider more time during Cross Correlation analysis
		Start_Index_more_time = int( Start_Index - Time_before * 44100 )   ## 44100 is the sampling rate "frame rate" "Sampling frequency"
		End_Index_more_time   = int( End_Index   + Time_after  * 44100 )
		if Start_Index_more_time < 0:   ## to avoid "-ve" values. This may occurs if the deer sound detected at very early stage of recording 
			Start_Index_more_time = 0
		
		elif End_Index_more_time > Min_Rec_duration_Rec_all * 44100:   ## ## to avoid value bigger that total sound frames. This may occers if the deer sound detected at the end of recording started
			End_Index_more_time = int( Min_Rec_duration_Rec_all * 44100 )
		
		StartTime_more_time = Start_Index_more_time / 44100.
		EndTime_more_time   = End_Index_more_time   / 44100.
		Duration_more_time  = (End_Index_more_time - Start_Index_more_time +1) / 44100.


		## consider Fixed time (e.g., 6 sec from the middle frame) to clip sound signal to train DNN model
		Time_tot_DNN = Time_befr_DNN + Time_aftr_DNN
		if CRyDuration > Time_tot_DNN:    ## Skip this cry during the training of DNN
			Start_Index_DNN = -999
			StartTime_DNN   = -999
			End_Index_DNN   = -999
			EndTime_DNN     = -999
			Duration_DNN    = -999
		else:
			Mid_Index = int( (Start_Index + End_Index) / 2. )     ## int >> to avoid "0.5" during middle
			Start_Index_DNN = int( Mid_Index - Time_befr_DNN * 44100 )
			End_Index_DNN   = int( Mid_Index + Time_aftr_DNN * 44100 )
			
			if Start_Index_DNN < 0:   ## to avoid "-ve" values. This may occurs if the deer sound detected at very early stage of recording 
				Start_Index_DNN = 0
				End_Index_DNN   = int( Start_Index_DNN + Time_tot_DNN * 44100 )
		
			elif End_Index_DNN > Min_Rec_duration_Rec_all * 44100:   ## to avoid value bigger that total sound frames. This may occers if the deer sound detected at the end of recording started
				End_Index_DNN   = int( Min_Rec_duration_Rec_all  * 44100 )
				Start_Index_DNN = int( End_Index_DNN - Time_tot_DNN * 44100 )
		
			StartTime_DNN = Start_Index_DNN / 44100.
			EndTime_DNN   = End_Index_DNN   / 44100.	
			Duration_DNN  = (End_Index_DNN - Start_Index_DNN +1) / 44100.

		## Save Indeices into Array
		print( i1+1, "/", No_detected_deer_sound, " Cry Time >> Start=%.3f , end=%.3f, Dur=%.3f "           %( StartTime, EndTime, CRyDuration ) )
		print(       "      More Time >> Start=%.3f , end=%.3f, Dur=%.3f "  %( StartTime_more_time, EndTime_more_time, Duration_more_time ) )
		print(       "      DNN Time >> Start=%.3f , end=%.3f, Dur=%.3f "         %( StartTime_DNN, EndTime_DNN, Duration_DNN ) )

		df.loc[i1, "DeerNo"]            = Deer_Sound_no
		df.loc[i1, "Date"]	            = Date_
		df.loc[i1, "hh"]	            = Hour_
		df.loc[i1, "StartCryTime(Sec)"] = StartTime
		df.loc[i1, "EndCtyTime(Sec)"]   = EndTime
		df.loc[i1, "CryDuration(Sec)"]  = CRyDuration
		df.loc[i1, "AnalysisStartTime"] = Start_Index_more_time
		df.loc[i1, "AnalysisEndTime"]   = End_Index_more_time
		df.loc[i1, "DNN_StartTime"]	    = Start_Index_DNN
		df.loc[i1, "DNN_EndTime"]       = End_Index_DNN



		##################################################################################################################
		## Save Sound Signal Sliced (High Pass Filter Sound Signal)			
		if CRyDuration < Time_tot_DNN:	
			for POIx in dc.keys():
				wav_ary_slice = dc[POIx]["SignalAry_filtered"][Start_Index_DNN:End_Index_DNN]
				Save_Deer_Cry_Sliced_Wav_f (wav_ary_slice, POIx, Date_, Hour_, StartTime_DNN, EndTime_DNN, Duration_DNN, Deer_Sound_no, Output_Hourly_Dir)

		print("Finish Saving Sound Clip of Deer No. %d......"    %(i1+1))

	print("###########################################")



	#################################################################################
	#################################################################################
	#################################################################################
	###... <<<  2nd Stage - Cross Correlation to estimate Time Delay  >>>
	print("2nd Stage - Cross Correlation to estimate Time Delay")

	def Cross_Correlation_for_Time_Delay_f (wav_ary_1st, wav_ary_2nd):

		# using any input Signal Array to estimate No. of Frames and Duration of One Sound signal
		no_frm_one_sound_slice = wav_ary_1st.shape[0]
		duration_one_sound_slice = no_frm_one_sound_slice / sampling_rate

		# No of frames of Cross Correlation is Almost twice of input sound signal's frame
		no_frm_Cross_Corr_one_sound_slice = no_frm_one_sound_slice * 2 - 1

		##... Cross Correlation between two sound signal
		Cross_Corr_Ary = signal.correlate(wav_ary_1st, wav_ary_2nd, mode='full')
		time_x_axis    = np.linspace(-duration_one_sound_slice, duration_one_sound_slice,  no_frm_Cross_Corr_one_sound_slice)


		## Time Delay from Cross Correlation
		max_cross_corr = np.max(Cross_Corr_Ary)
		Indx_max_cross_corr = np.argmax(Cross_Corr_Ary)
		time_delay = ( time_x_axis[ Indx_max_cross_corr ] )

		return time_delay

		
		
	#################################################################################
	#################################################################################
	##.. Loop for each deer Cry to estimate Time Delay using Cross Correlation
	for i1 in range ( No_detected_deer_sound ):

		## indices to Slice Deer Cry
		Start_Index = (Ones_Indx_Lst_Slice_spectrum_128Step[i1][0])*128
		End_Index   = (Ones_Indx_Lst_Slice_spectrum_128Step[i1][-1])*128
		
		if Increase_Analysis_Time_Range == "Yes":   ## incease	analysis time 	
			## consider more time during Cross Correlation analysis
			Start_Index = int( Start_Index - Time_before * 44100 )   ## 44100 is the sampling rate "frame rate" "Sampling frequency"
			End_Index   = int( End_Index   + Time_after  * 44100 )
			if Start_Index < 0:   ## to avoid "-ve" values. This may occurs if the deer sound detected at very early stage of recording 
				Start_Index = 0
			
			elif End_Index > Min_Rec_duration_Rec_all * 44100:   ## ## to avoid value bigger that total sound frames. This may occers if the deer sound detected at the end of recording started
				End_Index = int( Min_Rec_duration_Rec_all * 44100 )
			
		StartTime = Start_Index / 44100.
		EndTime   = End_Index   / 44100.
		
		# store wave array of each deer cry
		wav_ary_HPF_slice = {}
		for POIx in dc.keys():
			wav_ary_HPF_slice[POIx] = dc[POIx]["SignalAry_filtered"][Start_Index:End_Index]
			
		## Cross Correlation to estimate time delay
		POIx_list = list( dc.keys() )              # e.g., [POI1', 'POI2', 'POI3']
		# combinations of POIx without repeatation # e.g., [('POI1', 'POI2'), ('POI1', 'POI3'), ('POI2', 'POI3')]
		POIx_comb = combinations( POIx_list, 2)  # 2: means combination length  e.g., (POI1, POI2)
		POIx_comb = [i for i in POIx_comb]   # e.g.,  [('POI1', 'POI2'), ('POI1', 'POI3'), ('POI2', 'POI3')]
		
		for POI_1st, POI_2nd  in POIx_comb:
			#print(POI_1st, POI_2nd) 
			time_delay_n = f"time_delay_{POI_1st}_{POI_2nd}"  # e.g., time_delay_POI1_POI2'
			wav_ary_1st  = wav_ary_HPF_slice[POI_1st]
			wav_ary_2nd  = wav_ary_HPF_slice[POI_2nd]
			
			time_delay_              = Cross_Correlation_for_Time_Delay_f (wav_ary_1st, wav_ary_2nd)
			df.loc[i1, time_delay_n] = time_delay_
			#print( f"{time_delay_n} = {time_delay_}" )

		print("Finish Time Delay of Deer No. %d out of %d."    %(i1+1, No_detected_deer_sound))
		print("###########################################")
		
	Output_f_n  = os.path.join( Output_Hourly_Dir, f"DeerCryInformation_{Date_}_{Hour_}hh.csv" )
	df.to_csv(Output_f_n, index=False)

################################################################################# 
###... Read Hourly Recorders 
List_of_Folders = next(os.walk(Input_Sound_Main_Dir))[1] 
List_of_Folders = natsorted(List_of_Folders)

for Hourly_Folder_ in List_of_Folders:

	print("############################################################")
	print("current folder is  : ", Hourly_Folder_)

	#Hourly_dir = r"%s\%s"     %(Input_Sound_Main_Dir, Hourly_Folder_)  ## "r" will make string as raw string. Thus, if we have (r"\n") this will not make a new line
	Hourly_dir = os.path.join(Input_Sound_Main_Dir, Hourly_Folder_)     
	#List_of_POIxx_Folders = next(os.walk(Hourly_dir))[1]    ## os.walk(Hourly_Folder_) return 3-things <directory, folders, and files>
														 ##    > thus, next()[1] will return only the root 1 folders 
	
	List_of_output_folder = next(os.walk(Hourly_dir))[1]          ## next()[1] will return only the folders in this folder

	Files_List = next(os.walk(Hourly_dir))[2]          ## next()[2] will return only the files in this folder	
	Files_List = [file_ for file_ in Files_List if (file_.endswith(".mp3")) | (file_.endswith(".mp3"))]		
	
	Main_Code_f (Hourly_dir, Files_List)
			
	# Clear memory used by python
	gc.collect()
			
	# pause for 5 sec
	time.sleep(60) # Seconds
			
	#input("Debug")

				
################################################################################# 
###... remove wav File from duplicate same name files with different extension (i.e., mp3 & wav) 

# e.g., r"N:\Deer_Project_Taki\211014-008.mp3"   and   r"N:\Deer_Project_Taki\211014-008.wav"
# as we know, if the original sound recordings is in mp3, we will create a wav file
# for this case, we need to remove wav file as its size is 10 times bigger than MP3 file

EXTENSIONS = {'.mp3', '.wav'}

for Hourly_Folder_ in List_of_Folders:
	
	print("##############################")
	print("Remove Wav in folder : ", Hourly_Folder_)

	Hourly_dir = os.path.join( Input_Sound_Main_Dir, Hourly_Folder_ )
	grouped_files = defaultdict(int)  # create a new dictionary-like object.
	
	for file_ in os.listdir(Hourly_dir):
		f_n_dir, ext = os.path.splitext(os.path.join(Hourly_dir, file_))
		if ext in EXTENSIONS:
			grouped_files[f_n_dir] += 1
	for f_n_dir in grouped_files:
		if grouped_files[f_n_dir] == len(EXTENSIONS):  # 2 files 
				f_n_dir_ext = f_n_dir + ".wav"
				print(f_n_dir_ext)
				os.remove(f_n_dir_ext)
				print("File Removed!")				



