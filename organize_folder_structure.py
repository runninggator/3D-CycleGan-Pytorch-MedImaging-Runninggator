
import os
import shutil
from time import time
import re
import argparse
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from utils.NiftiDataset import *
import json


def numericalSort(value):
    # Get all numbers from string
    num = ''.join(re.findall(r'(\d+)', value))

    if num:
        return int(num)
    else:
        return 0


def lstFiles(Path):

    images_list = []  # create an empty list, the raw image data files is stored here
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            filename_lower = filename.lower()
            if any([file_extension in filename_lower for file_extension in [".nii.gz", ".nii", ".mhd"]]):
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)
    return images_list


def Registration(image, label, save_path=''):

    image, image_sobel, label, label_sobel,  = image, image, label, label

    Gaus = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    image_sobel = Gaus.Execute(image_sobel)
    label_sobel = Gaus.Execute(label_sobel)

    fixed_image = label_sobel
    moving_image = image_sobel

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    image = sitk.Resample(image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())

    # Save transform
    if save_path:
        sitk.WriteTransform(final_transform, save_path)

    return image, label


parser = argparse.ArgumentParser()
parser.add_argument('--images', default='./Data_folder/T1', help='path to the images a (early frames)')
parser.add_argument('--labels', default='./Data_folder/T2', help='path to the images b (late frames)')
parser.add_argument('--result', default='./Data_folder/', help='path to save the train/test/validation/transformation images')
parser.add_argument('--resolution', default=(1.6,1.6,1.6), help='new resolution to resample the all data')
parser.add_argument('--config_file',  help='JSON file with a ' 
                                            + '"test" attribute (list of all test subjects), ' 
                                            + '"train" attribute (list of all train subjects), '
                                            + '"validation" attribute (list of all validation subjects) '
                                            + '"file_extension" attribute and ' 
                                            + '"reg_ref" attribute (name of subject to register all subjects with)')
parser.add_argument('--no_reg', action='store_true', help='Whether to skip the registration step')
args = parser.parse_args()

if __name__ == "__main__":

    # Read json config file
    config_options = json.load(open(args.config_file))

    # Check config file has the required attributes
    required_attributes = ['file_extension', 'train', 'test']
    if not all(attribute in config_options for attribute in required_attributes):
        raise Exception(f'Config file is missing attributes. Required attributes: {str(required_attributes)}')

    list_images = lstFiles(args.images)
    list_labels = lstFiles(args.labels)

    # setting a reference image to have all data in the same coordinate system
    reference_image = None

    if not args.no_reg:
        if "reg_ref" in config_options:
            for filename in list_labels:
                if f'{config_options["reg_ref"]}.{config_options["file_extension"]}' in filename:
                    reference_image = filename
                    break

            if reference_image is None:
                raise Exception('Could not find the reference image in the label list.')
            
            reference_image = sitk.ReadImage(reference_image)
            reference_image = resample_sitk_image(reference_image, spacing=args.resolution, interpolator='linear')
        else:
            raise Exception('Config file is missing attribute "reg_ref".')

    for split in ['train', 'test', 'validation']:
        for filename in config_options[split]:

            save_directory_images = os.path.join(args.result, split, 'images')
            save_directory_labels = os.path.join(args.result, split, 'labels')
            save_directory_transforms = os.path.join(args.result, split, 'transforms')

            for dir_name in [save_directory_images, save_directory_labels, save_directory_transforms]:
                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)

            a = os.path.join(args.images, f"{filename}.{config_options['file_extension']}")
            b = os.path.join(args.labels, f"{filename}.{config_options['file_extension']}")

            print(a)

            label = sitk.ReadImage(b)
            image = sitk.ReadImage(a)

            if not args.no_reg and "reg_ref" in config_options:
                transform_path = os.path.join(save_directory_transforms, f'{filename}.tfm')

                image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear')
                label = resample_sitk_image(label, spacing=args.resolution, interpolator='linear')

                label, _ = Registration(label, reference_image, transform_path)
                image, label = Registration(image, label)

            label_directory = os.path.join(str(save_directory_labels), f'{filename}.nii')
            image_directory = os.path.join(str(save_directory_images), f'{filename}.nii')

            sitk.WriteImage(image, image_directory)
            sitk.WriteImage(label, label_directory)