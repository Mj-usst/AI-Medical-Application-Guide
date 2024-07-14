## 通用影像组学流程模块化代码

我们希望创建一个模块化且适应不同情况的影像组学流程代码，将整个流程分为几个关键模块：
研究设计、数据采集、数据管理、图像处理和分割、特征提取、模型构建与评估。

### 1. 研究设计模块
```python
class ResearchDesign:
    def __init__(self, research_question, prediction_target, roi_definition, analysis_strategy):
        self.research_question = research_question
        self.prediction_target = prediction_target
        self.roi_definition = roi_definition
        self.analysis_strategy = analysis_strategy

    def display_design(self):
        print(f"Research Question: {self.research_question}")
        print(f"Prediction Target: {self.prediction_target}")
        print(f"ROI Definition: {self.roi_definition}")
        print(f"Analysis Strategy: {self.analysis_strategy}")

if __name__ == "__main__":
    design = ResearchDesign(
        research_question="Predicting disease progression using MRI data",
        prediction_target="Progression-Free Survival",
        roi_definition={"organ": "liver", "region": "tumor"},
        analysis_strategy="Use random forest for classification"
    )
    design.display_design()
```
### 2. 数据采集模块
```python
# data_acquisition.py
import pydicom
import numpy as np
from pydicom.data import get_testdata_files

class DataAcquisition:
    def __init__(self, filename):
        self.filename = filename

    def load_dicom(self):
        ds = pydicom.dcmread(self.filename)
        return ds

# 示例用法
if __name__ == "__main__":
    filename = get_testdata_files("CT_small.dcm")[0]
    data_acquisition = DataAcquisition(filename)
    dicom_data = data_acquisition.load_dicom()
    print(dicom_data)
```
### 3. 数据管理模块
```python
# data_management.py
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
```
### 4. 图像处理和分割模块
```python
# image_processing.py
import cv2

class ImageProcessing:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, 0)

    def apply_gaussian_filter(self, kernel_size=(5, 5), sigma=0):
        filtered_image = cv2.GaussianBlur(self.image, kernel_size, sigma)
        return filtered_image

    def resample(self, new_spacing=[1, 1, 1]):
        # 假设在这里进行重采样操作
        resampled_image = self.image  # 示例，不实际重采样
        return resampled_image

    def register_images(self, fixed_image, moving_image):
        # 假设在这里进行图像配准操作
        registered_image = moving_image  # 示例，不实际配准
        return registered_image


if __name__ == "__main__":
    image_path = 'output_image.nii'
    image_processing = ImageProcessing(image_path)
    filtered_image = image_processing.apply_gaussian_filter()
    cv2.imwrite('filtered_image.nii', filtered_image)
```
### 5. 特征提取模块
```python
# feature_extraction.py
from skimage.feature import greycomatrix, greycoprops

class FeatureExtraction:
    def __init__(self, image):
        self.image = image

    def extract_glcm_features(self, distances=[5], angles=[0]):
        glcm = greycomatrix(self.image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'contrast')
        return contrast

if __name__ == "__main__":
    from image_processing import ImageProcessing
    image_path = 'filtered_image.nii'
    image_processing = ImageProcessing(image_path)
    filtered_image = image_processing.apply_gaussian_filter()
    
    feature_extraction = FeatureExtraction(filtered_image)
    features = feature_extraction.extract_glcm_features()
    print("Contrast:", features)
```
### 6. 模型构建与评估模块
```python
# modeling.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

class Modeling:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def build_and_evaluate_model(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=test_size)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy

if __name__ == "__main__":
    # 假设features和labels已定义
    features = np.random.rand(100, 10)
    labels = np.random.randint(2, size=100)
    
    modeling = Modeling(features, labels)
    model, accuracy = modeling.build_and_evaluate_model()
    print("Model Accuracy:", accuracy)
```
### 主脚本
```python
# main.py
from research_design import ResearchDesign
from data_acquisition import DataAcquisition
from data_management import DataManagement
from image_processing import ImageProcessing
from feature_extraction import FeatureExtraction
from modeling import Modeling

# 研究设计
design = ResearchDesign(
    research_question="Predicting disease progression using MRI data",
    prediction_target="Progression-Free Survival",
    roi_definition={"organ": "liver", "region": "tumor"},
    analysis_strategy="Use random forest for classification"
)
design.display_design()

# 数据采集
filename = get_testdata_files("CT_small.dcm")[0]
data_acquisition = DataAcquisition(filename)
dicom_data = data_acquisition.load_dicom()

# 数据管理
data_management = DataManagement(dicom_data)
output_path = data_management.dicom_to_nifti('output_image.nii')

# 图像处理和分割
image_processing = ImageProcessing(output_path)
filtered_image = image_processing.apply_gaussian_filter()
cv2.imwrite('filtered_image.nii', filtered_image)

# 特征提取
feature_extraction = FeatureExtraction(filtered_image)
features = feature_extraction.extract_glcm_features()

# 模型构建与评估
# 假设labels已经存在
labels = np.random.randint(2, size=len(features))
modeling = Modeling(np.array(features).reshape(-1, 1), labels)
model, accuracy = modeling.build_and_evaluate_model()
print("Model Accuracy:", accuracy)
```






