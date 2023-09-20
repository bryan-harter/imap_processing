from .. import common_cdf_attrs
import numpy as np

# Set IDEX software version here for now
software_version = '01'

# Global Attributes
idex_l1_global_attrs = {'Data_type': ['l1'],
                        'Data_version': [software_version],
                        'TEXT': ["Greg Newcomb, the flight software (FSW) engineer for the Interstellar Dust Experiment (IDEX) aboard the Interstellar Mapping and Acceleration Probe (IMAP) mission has passed a sawtooth function through IDEX's 6 channels for the science data pipeline verification. "],
                        'Mission_group': ['IMAP'],
                        'Logical_source': ['imap_idex_l1'],
                        'Logical_file_id': [f'imap_idex_l1_YYYYMMDD_v{software_version}'],
                        'Logical_source_description': ['L1 files for the IDEX instrument on the IMAP mission']} | common_cdf_attrs.global_base

idex_l2_global_attrs = {'Data_type': ['l2'],
                        'Data_version': [software_version],
                        'TEXT': ["Greg Newcomb, the flight software (FSW) engineer for the Interstellar Dust Experiment (IDEX) aboard the Interstellar Mapping and Acceleration Probe (IMAP) mission has passed a sawtooth function through IDEX's 6 channels for the science data pipeline verification. "],
                        'Mission_group': ['IMAP'],
                        'Logical_source': ['imap_idex_l2'],
                        'Logical_file_id': [f'imap_idex_l2_YYYYMMDD_v{software_version}'],
                        'Logical_source_description': ['L2 files for the IDEX instrument on the IMAP mission']} | common_cdf_attrs.global_base

# L1 variables base dictionaries (these are not complete)
l1_data_base = {'DEPEND_0': 'Epoch', 
                'DISPLAY_TYPE': 'spectrogram', 
                'FILLVAL': np.array([-1.e+31], dtype=np.float32), 
                'FORMAT': 'E12.2', 
                'UNITS': 'dN', 
                'VALIDMIN': np.array([-1.e+31], dtype=np.float32), 
                'VALIDMAX': np.array([1.e+31], dtype=np.float32), 
                'VAR_TYPE': 'data', 
                'SCALETYP': 'linear'}

l1_tof_base = {'DEPEND_1': 'Time_High_SR',
               'LABLAXIS': 'Time_[dN]'} | l1_data_base

l1_target_base = {'DEPEND_1': 'Time_Low_SR',
                  'LABLAXIS': 'Amplitude_[dN]'} | l1_data_base

sample_rate_base = {'DEPEND_0': 'Epoch', 
                    'FILLVAL': np.array([-1.e+31], dtype=np.float32), 
                    'FORMAT': 'E12.2', 
                    'LABLAXIS': 'Time_[dN]', 
                    'UNITS': 'microseconds', 
                    'VALIDMIN': np.array([-1.e+31], dtype=np.float32), 
                    'VALIDMAX': np.array([1.e+31], dtype=np.float32), 
                    'VAR_TYPE': 'support_data', 
                    'SCALETYP': 'linear'}

trigger_base = {'DEPEND_0': 'Epoch',
                'DISPLAY_TYPE': 'no_plot',  
                'FILLVAL': np.array([-1.e+31], dtype=np.float32), 
                'FORMAT': 'E12.2', 
                'LABLAXIS': 'Trigger_Info', 
                'UNITS': '', 
                'VALIDMIN': 0, 
                'VALIDMAX': np.array([1.e+31], dtype=np.float32), 
                'VAR_TYPE': 'metadata'}

# L1 Attribute Dictionaries
low_sr_attrs = {'CATDESC': 'Time_Low_SR', 
                'FIELDNAM': 'Time_Low_SR', 
                'VAR_NOTES': 'The Low sample rate in microseconds.  Steps are approximately 1/4.025 nanoseconds in duration.'} | sample_rate_base

high_sr_attrs = {'CATDESC': 'Time_High_SR', 
                'FIELDNAM': 'Time_High_SR', 
                'VAR_NOTES': 'The High sample rate in microseconds.  Steps are approximately 1/260 nanoseconds in duration.'} | sample_rate_base

tof_high_attrs = {'CATDESC': 'TOF_High', 
                  'FIELDNAM': 'TOF_High', 
                  'VAR_NOTES': 'This is the high gain channel of the time-of-flight signal.'} | l1_tof_base

tof_mid_attrs = {'CATDESC': 'TOF_Mid', 
                  'FIELDNAM': 'TOF_Mid', 
                  'VAR_NOTES': 'This is the mid gain channel of the time-of-flight signal.'} | l1_tof_base

tof_low_attrs = {'CATDESC': 'TOF_Low', 
                  'FIELDNAM': 'TOF_Low', 
                  'VAR_NOTES': 'This is the low gain channel of the time-of-flight signal.'} | l1_tof_base

target_low_attrs = {'CATDESC': 'Target_Low', 
                    'FIELDNAM': 'Target_Low', 
                    'VAR_NOTES': "This is the low gain channel of IDEX's target signal."} | l1_target_base

target_high_attrs = {'CATDESC': 'Target_High',
                     'FIELDNAM': 'Target_High',
                     'VAR_NOTES': "This is the high gain channel of IDEX's target signal."} | l1_target_base

ion_grid_attrs = {'CATDESC': 'Ion_Grid',  
                  'FIELDNAM': 'Ion_Grid', 
                  'VAR_NOTES': 'This is the ion grid signal from IDEX.'} | l1_target_base

# Level 2 variable base dictionaries 
model_base = {'DEPEND_0': 'Epoch', 
              'DISPLAY_TYPE': 'time_series', 
              'FILLVAL': np.array([-1.e+31], dtype=np.float32), 
              'FORMAT': 'E12.2', 
              'VALIDMIN': np.array([-1.e+31], dtype=np.float32), 
              'VALIDMAX': np.array([1.e+31], dtype=np.float32), 
              'VAR_TYPE': 'data', 
              'SCALETYP': 'linear'}

model_amplitude_base = {'UNITS': 'dN'} | model_base
model_time_base = {'UNITS': 's'} | model_base
model_dimensionless_base = {'UNITS': ''} | model_base

tof_model_base = {'DEPEND_1': 'mass_number'} | model_base
tof_model_amplitude_base = {'UNITS': 'dN'} | tof_model_base
tof_model_time_base = {'UNITS': 's'} | tof_model_base
tof_model_dimensionless_base = {'UNITS': ''} | tof_model_base

mass_number_attrs = {'CATDESC': 'Mass_Number', 
                    'DISPLAY_TYPE': 'no_plot', 
                    'FIELDNAM': 'Mass_Number', 
                    'FILLVAL': '', 
                    'FORMAT': 'E12.2', 
                    'LABLAXIS': 'Mass_Number', 
                    'UNITS': '', 
                    'VALIDMIN': '1', 
                    'VALIDMAX': '50', 
                    'VAR_TYPE': 'support_data', 
                    'VAR_NOTES': 'These represent the peaks of the TOF waveform'}