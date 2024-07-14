import pydicom
import numpy as np
from pydicom.data import get_testdata_files

class DataAcquisition:
    def __init__(self, filename):
        self.filename = filename

    def load_dicom(self):
        ds = pydicom.dcmread(self.filename)
        return ds

if __name__ == "__main__":
    filename = get_testdata_files("CT_small.dcm")[0]
    data_acquisition = DataAcquisition(filename)
    dicom_data = data_acquisition.load_dicom()
    print(dicom_data)
