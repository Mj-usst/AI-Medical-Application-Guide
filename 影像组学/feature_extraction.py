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
