import cv2

class ImageProcessing:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, 0)

    def apply_gaussian_filter(self, kernel_size=(5, 5), sigma=0):
        filtered_image = cv2.GaussianBlur(self.image, kernel_size, sigma)
        return filtered_image

    def resample(self, new_spacing=[1, 1, 1]):
        # 重采样操作
        resampled_image = self.image  
        return resampled_image

    def register_images(self, fixed_image, moving_image):
        # 图像配准操作
        registered_image = moving_image  
        return registered_image

if __name__ == "__main__":
    image_path = 'output_image.nii'
    image_processing = ImageProcessing(image_path)
    filtered_image = image_processing.apply_gaussian_filter()
    cv2.imwrite('filtered_image.nii', filtered_image)
