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
