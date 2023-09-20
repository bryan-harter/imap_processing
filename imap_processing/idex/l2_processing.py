import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, find_peaks
from scipy.special import erfc
import numpy as np
import logging
import lmfit
import xarray as xr
from cdflib.xarray import cdf_to_xarray, xarray_to_cdf
from . import idex_cdf_attrs

class L2Processor:
    """

    Example
    -------
    >>> from imap_processing.idex.idex_packet_parser import PacketParser
    >>> from imap_processing.idex.l2_processing import L2Processor

    >>> l0_file = "imap_processing/idex/tests/imap_idex_l0_20230725_v01-00.pkts"
    >>> l1_data = PacketParser(l0_file)
    >>> l1_data.write_cdf_file("20230725")

    >>> l2_data = L2Processor('imap_idex_l1_20230725_v01.cdf')
    >>> l2_data.write_l2_cdf()
    """

    TARGET_HIGH_CALIBRATION = .00135597 
    ION_GRID_CALIBRATION = .00026148
    time_of_impact_init = 20                           
    constant_offset_init = 0.                            
    rise_time_init = 3.71
    discharge_time_init = 37.1
    FILLVAL = -1.e+31
    TOF_Low_Peak_Prominence = 7
    
    def __init__(self, l1_file: str):
        self.l1_file = l1_file
        self.l1_data = cdf_to_xarray(l1_file)

        target_signal_model_dataset = self.model_fitter('Target_High', self.TARGET_HIGH_CALIBRATION, butterworth_filter=False)
        ion_grid_model_dataset =  self.model_fitter('Ion_Grid', self.TARGET_HIGH_CALIBRATION, butterworth_filter=True)
        tof_model_dataset = self.fit_TOF_model('TOF_Low', peak_prominence=self.TOF_Low_Peak_Prominence)

        self.l2_data = xr.merge([self.l1_data, target_signal_model_dataset, ion_grid_model_dataset, tof_model_dataset])


    def model_fitter(self, variable: str, amplitude_calibration: float, butterworth_filter: bool =False):
        model_fit_list = []
        for impact in self.l1_data[variable]:
            epoch_xr = xr.DataArray(name="Epoch", data=[impact['Epoch'].data], dims=("Epoch"))
            x = impact[impact.attrs['DEPEND_1']].data
            y = impact.data - np.mean(impact.data[0:10])
            try:
                model = lmfit.Model(self.idex_response_function)
                params = model.make_params(time_of_impact = self.time_of_impact_init, 
                                           constant_offset = self.constant_offset_init, 
                                           amplitude = max(y), 
                                           rise_rime = self.rise_time_init, 
                                           discharge_time = self.discharge_time_init)
                params['rise_time'].min = 5
                params['rise_time'].max = 10000

                if butterworth_filter:
                    y = self.butter_lowpass_filter(y, x)

                result = model.fit(y, params, x=x)

                param = result.best_values

                _, param_cov = curve_fit(self.idex_response_function, x, y)
                fit_uncertainty = np.linalg.det(param_cov)
            
                time_of_impact_fit = param['time_of_impact']
                constant_offset_fit = param['constant_offset']
                amplitude_fit = amplitude_calibration*param['amplitude']
                rise_time_fit = param['rise_time']
                discharge_time_fit = param['discharge_time']

            except Exception as e:
                logging.warning("Error fitting Models, resorting to FILLVALs: " + str(e)) 
                time_of_impact_fit = self.FILLVAL
                constant_offset_fit = self.FILLVAL
                amplitude_fit = self.FILLVAL
                rise_time_fit = self.FILLVAL
                discharge_time_fit = self.FILLVAL
                fit_uncertainty = self.FILLVAL

            time_of_impact_fit_xr = xr.DataArray(name=f"{variable}_Model_Time_Of_Impact",
                                                data=[time_of_impact_fit],
                                                dims=("Epoch"))
            constant_offset_fit_xr = xr.DataArray(name=f"{variable}_Model_Constant_Offset",
                                                data=[constant_offset_fit],
                                                dims=("Epoch"))
            amplitude_fit_xr = xr.DataArray(name=f"{variable}_Model_Amplitude",
                                            data=[amplitude_fit],
                                            dims=("Epoch"))
            rise_time_fit_xr = xr.DataArray(name=f"{variable}_Model_Rise_time",
                                            data=[rise_time_fit],
                                            dims=("Epoch"))
            discharge_time_xr = xr.DataArray(name=f"{variable}_Model_Discharge_time",
                                            data=[discharge_time_fit],
                                            dims=("Epoch"))
            fit_uncertainty_xr = xr.DataArray(name=f"{variable}_Model_Uncertainty",
                                            data=[fit_uncertainty],
                                            dims=("Epoch"))
            
            model_fit_list.append(xr.Dataset(
                                             data_vars={
                                                 f"{variable}_Model_Time_Of_Impact": time_of_impact_fit_xr,
                                                 f"{variable}_Model_Constant_Offset": constant_offset_fit_xr,
                                                 f"{variable}_Model_Amplitude": amplitude_fit_xr,
                                                 f"{variable}_Model_Rise_Time": rise_time_fit_xr,
                                                 f"{variable}_Model_Discharge_Time": discharge_time_xr,
                                                 f"{variable}_Model_Uncertainty": fit_uncertainty_xr,
                                             },
                                             coords={
                                                 "Epoch": epoch_xr
                                             },
                                            )
            )
        return xr.concat(model_fit_list, dim="Epoch")

    @staticmethod
    def idex_response_function(x, time_of_impact, constant_offset, amplitude, rise_time, discharge_time):
        heaviside = np.heaviside(x-time_of_impact, 0) 
        exponent_1 = (1.0 - np.exp(-(x-time_of_impact)/rise_time))
        exponent_2 = np.exp( -(x-time_of_impact)/discharge_time)
        return constant_offset + (heaviside * amplitude *  exponent_1 * exponent_2)
    
    # Create a model for exponentially modified Gaussian
    @staticmethod
    def expgaussian(x, amplitude, center, sigma, gamma):
        dx = center - x
        return amplitude * np.exp(gamma * dx) * erfc(dx / (np.sqrt(2) * sigma))
    
    @staticmethod
    def butter_lowpass_filter(data, time):
        # Filter requirements.
        T = time[1] - time[0]       # |\Sample Period (s)
        fs = (time[-1] - time[0])/T # ||sample rate, Hz
        cutoff = 10                 # ||desired cutoff frequency of the filter, Hz
        nyq = 0.5 * fs              # ||Nyquist Frequency
        order = 2                   # ||sine wave can be approx represented as quadratic
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    def fit_TOF_model(self, variable, peak_prominence):
        mass_number_xr = xr.DataArray(name="mass_number", 
                                      data=np.linspace(1,50,50), 
                                      dims=("mass_number"),
                                      attrs=idex_cdf_attrs.mass_number_attrs)
        
        tof_model_parameters_list = []
        for impact in self.l1_data[variable]:
            epoch_xr = xr.DataArray(name="Epoch", data=[impact['Epoch'].data], dims=("Epoch"))
            
            mass_amplitudes = np.full(50, self.FILLVAL)
            mass_centers = np.full(50, self.FILLVAL)
            mass_sigmas = np.full(50, self.FILLVAL)
            mass_gammas = np.full(50, self.FILLVAL)

            x = impact[impact.attrs['DEPEND_1']].data 
            y = impact.data

            peaks, _ = find_peaks(y, prominence=peak_prominence)
            i=0
            for peak in peaks:
                try:
                    i+=1
                    fit_params = self.fit_expgaussian(x[peak-10: peak+10], y[peak-10: peak+10])
                    mass_amplitudes[i], mass_centers[i], mass_sigmas[i], mass_gammas[i] = tuple(fit_params.values())
                except Exception as e:
                    logging.warning("Error fitting TOF Model.  Defaulting to FILLVALS. " + str(e))

            amplitude_xr = xr.DataArray(name=f"{variable}_Model_Masses_Amplitude",
                                        data=[mass_amplitudes],
                                        dims=("Epoch", "mass_number"))
            center_xr = xr.DataArray(name=f"{variable}_Model_Masses_Center",
                                    data=[mass_centers],
                                    dims=("Epoch", "mass_number"))
            sigma_xr = xr.DataArray(name=f"{variable}_Model_Masses_Sigma",
                                    data=[mass_sigmas],
                                    dims=("Epoch", "mass_number"))
            gamma_xr = xr.DataArray(name=f"{variable}_Model_Masses_Gamma",
                                    data=[mass_gammas],
                                    dims=("Epoch", "mass_number"))

            tof_model_parameters_list.append(xr.Dataset(
                                                    data_vars={
                                                        f"{variable}_Model_Masses_Amplitude": amplitude_xr,
                                                        f"{variable}_Model_Masses_Center" : center_xr,
                                                        f"{variable}_Model_Masses_Sigma" : sigma_xr,
                                                        f"{variable}_Model_Masses_Gamma" : gamma_xr,
                                                    },
                                                    coords={
                                                        "Epoch": epoch_xr,
                                                        "mass_number": mass_number_xr
                                                    },
                                                )
            )
            
        return xr.concat(tof_model_parameters_list, dim="Epoch")

    # Fit the exponentially modified Gaussian
    def fit_expgaussian(self, x, y):
        model = lmfit.Model(self.expgaussian)
        params = model.make_params(amplitude=max(y), center=x[np.argmax(y)], sigma=10.0, gamma=10.0)
        result = model.fit(y, params, x=x)
        return result.best_values
    
    def write_l2_cdf(self):
        self.l2_data.attrs = idex_cdf_attrs.idex_l2_global_attrs

        for var in self.l2_data:
            if "_Model_Amplitude" in var:
                self.l2_data[var].attrs = {"CATDESC": var,
                                           "FIELDNAM": var,
                                           "LABLAXIS": var,
                                           "VAR_NOTES": f"The amplitude of the response for {var.replace('_Model_Amplitude', '')}"} | idex_cdf_attrs.model_amplitude_base
            if "_Model_Uncertainty" in var:
                self.l2_data[var].attrs = {"CATDESC": var,
                                           "FIELDNAM": var,
                                           "LABLAXIS": var,
                                           "VAR_NOTES": f"The uncertainty in the model of the response for {var.replace('_Model_Amplitude', '')}"} | idex_cdf_attrs.model_dimensionless_base
            if "_Model_Constant_Offset" in var:
                self.l2_data[var].attrs = {"CATDESC": var,
                                           "FIELDNAM": var,
                                           "LABLAXIS": var,
                                           "VAR_NOTES": f"The constant offset of the response for {var.replace('_Model_Constant_Offset', '')}"} | idex_cdf_attrs.model_amplitude_base
            if "_Model_Time_Of_Impact" in var:
                self.l2_data[var].attrs = {"CATDESC": var,
                                           "FIELDNAM": var,
                                           "LABLAXIS": var,
                                           "VAR_NOTES": f"The time of impact for {var.replace('_Model_Time_Of_Impact', '')}"} | idex_cdf_attrs.model_time_base
            if "_Model_Rise_Time" in var:
                self.l2_data[var].attrs = {"CATDESC": var,
                                           "FIELDNAM": var,
                                           "LABLAXIS": var,
                                           "VAR_NOTES": f"The rise time of the response for {var.replace('_Model_Rise_Time', '')}"} | idex_cdf_attrs.model_time_base
            if "_Model_Discharge_Time" in var:
                self.l2_data[var].attrs = {"CATDESC": var,
                                           "FIELDNAM": var,
                                           "LABLAXIS": var,
                                           "VAR_NOTES": f"The discharge time of the response for {var.replace('_Model_Discharge_Time', '')}"} | idex_cdf_attrs.model_time_base
            if "_Model_Masses_Amplitude" in var:
                self.l2_data[var].attrs = {"CATDESC": var,
                                           "FIELDNAM": var,
                                           "LABLAXIS": var,
                                           "VAR_NOTES": f"The amplitude of the first 50 peaks in {var.replace('_Model_Masses_Amplitude', '')}"} | idex_cdf_attrs.tof_model_amplitude_base
            if "_Model_Masses_Center" in var:
                self.l2_data[var].attrs = {"CATDESC": var,
                                           "FIELDNAM": var,
                                           "LABLAXIS": var,
                                           "VAR_NOTES": f"The center of the first 50 peaks in {var.replace('_Model_Masses_Center', '')}"} | idex_cdf_attrs.tof_model_dimensionless_base
            if "_Model_Masses_Sigma" in var:
                self.l2_data[var].attrs = {"CATDESC": var,
                                           "FIELDNAM": var,
                                           "LABLAXIS": var,
                                           "VAR_NOTES": f"The sigma of the fitted exponentially modified gaussian to the first 50 peaks in {var.replace('_Model_Masses_Sigma', '')}"} | idex_cdf_attrs.tof_model_dimensionless_base
            if "_Model_Masses_Gamma" in var:
                self.l2_data[var].attrs = {"CATDESC": var,
                                           "FIELDNAM": var,
                                           "LABLAXIS": var,
                                           "VAR_NOTES": f"The gamma of the fitted exponentially modified gaussian to the first 50 peaks in {var.replace('_Model_Masses_Gamma', '')}"} | idex_cdf_attrs.tof_model_dimensionless_base

        l2_file_name = self.l1_file.replace('_l1_', '_l2_')

        xarray_to_cdf(self.l2_data, l2_file_name)

        return l2_file_name