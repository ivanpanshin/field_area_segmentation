import albumentations as A
import albumentations.pytorch as AT
import cv2
import torchvision.transforms as T



def to_tensor():
    # Constructing the additional targets for {crop_size} images and masks

    return A.Compose(
        [
            AT.ToTensorV2(),
        ],
    )


def resize(img_size):
    # Constructing the additional targets for {crop_size} images and masks

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            AT.ToTensorV2(),
        ],
    )


def regular():
    # Constructing the additional targets for {crop_size} images and masks

    return A.Compose(
        [
            A.ShiftScaleRotate(
                p=0.7,
                shift_limit_x=(-0.1, 0.1),
                shift_limit_y=(-0.1, 0.1),
                scale_limit=(-0.25, 0.25),
                rotate_limit=(-25, 25),
                border_mode=cv2.BORDER_CONSTANT,
                # rotate_method="largest_box",
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.25, 0.25),
                p=0.5,
            ),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OneOf(
                [
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
                ],
                p=0.2,
            ),
            AT.ToTensorV2(),
        ],
    )




def regular_resize(img_size):
    # Constructing the additional targets for {crop_size} images and masks

    return A.Compose(
        [
            A.ShiftScaleRotate(
                p=0.7,
                shift_limit_x=(-0.1, 0.1),
                shift_limit_y=(-0.1, 0.1),
                scale_limit=(-0.25, 0.25),
                rotate_limit=(-25, 25),
                border_mode=cv2.BORDER_CONSTANT,
                # rotate_method="largest_box",
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.25, 0.25),
                p=0.5,
            ),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OneOf(
                [
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
                ],
                p=0.2,
            ),
            A.Resize(img_size, img_size),
            AT.ToTensorV2(),
        ],
    )

