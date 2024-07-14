import nibabel as nib
import numpy as np

class DataManagement:
    def __init__(self, dicom_data):
        self.dicom_data = dicom_data

    def dicom_to_nifti(self, output_path):
        nifti_img = nib.Nifti1Image(self.dicom_data.pixel_array, affine=np.eye(4))
        nib.save(nifti_img, output_path)
        return output_path


if __name__ == "__main__":
    from data_acquisition import DataAcquisition
    filename = get_testdata_files("CT_small.dcm")[0]
    data_acquisition = DataAcquisition(filename)
    dicom_data = data_acquisition.load_dicom()
    
    data_management = DataManagement(dicom_data)
    output_path = data_management.dicom_to_nifti('output_image.nii')
    print(f"NIFTI file saved at: {output_path}")
