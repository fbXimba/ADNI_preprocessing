"""Author: Riccardo Biondi"""

import os
import itk
import logging
import argparse
import functools

LOG_LEVELS = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG
}

default_logger = logging.getLogger(__file__)

def parse_args():
    
    parser = argparse.ArgumentParser(description="Script to Identify the head region starting from the T1 weighted MRI")

    _ = parser.add_argument(
        "-in",
        "--input",
        dest="input",
        action="store",
        required=False,
        type=str,
        help="Path to the input MRI scan to correct. Input scan should be 2D or 3D and in .nii or .nii.gz format"
    )
    _ = parser.add_argument(
        "-out",
        "--output",
        dest="output",
        action="store",
        required=False,
        type=str,
        help="Path to save the resulting mask. Output scan format must be in .nii or .nii.gz"
    )

    #
    # Argument to control verbosity level
    #

    _ = parser.add_argument('-v', '--verbose', dest="verbose", action='count', default=0)

    args = parser.parse_args()
    return args




def update(func):
    """
    Decorator to automatically update an itk pipeline. The pipeline must be
    initlaized with the input/s images as *args and other as kwargs.
    The pipeline must return an itk filter, not an image.

    To deactivate the usage of the dex  corator, simply specify: upadte=False
    as kwargs in the function.

    Example
    -------
    >>> import itk
    >>> from ipt.decorators import update
    >>>
    >>> # Create a decorated function containing the pipeline to update
    >>>
    >>> @update
    >>> def pipeline(image, radius=1, **kwargs):
    >>>   median_filter = itk.MedianImageFilter[type(image), type(image)].New()
    >>>   _ = median_filter.SetInput(image)
    >>>   _ = median_filter.SetRadius(radius)
    >>>
    >>>   return median_filter
    >>>
    >>> def main():
    >>>
    >>>   image = itk.imread('path/to/input/image')
    >>>   filtered = median_filter(image)
    >>>   _ = itk.imwrite('path/to/output', filtered.GetOutput())
    >>>
    >>> if __name__ == '__main__':
    >>>   main()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        pipeline = func(*args, **kwargs)
        logger = kwargs.get("logger", default_logger)
        if kwargs.get('update', True):
            logger.debug(f'Updating {func.__name__}')

            _ = pipeline.Update()

        return pipeline
    return wrapper


def infer_itk_image_type(image, desidered_type=None, **kwargs):
    '''
    Infer the desidered image type: if default type is None, will return the
    type of the specified image, otherwise will return desidered_type
s
    Parameters
    ----------
    image: itk.Image
        itk Image from which infer the type

    desidered_type: itk ImageType
        type to return instead of the one of image. Default: None

    Return
    ------
    image_type: itk Image type (i.e. itk.Image[itk.UC, 2])
        inferred image type
    '''
    logger = kwargs.get("logger", default_logger)
    if desidered_type is not None:
        return desidered_type

    pixel_type, dimension = itk.template(image)[1]
    image_type = itk.Image[pixel_type, dimension]

    return image_type



@update
def itk_multi_otsu_threshold(image, number_of_thresholds=3, input_type=None, **kwargs):

    logger = kwargs.get("logger", default_logger)
    logger.debug(f"Multi Otsu Threshold: - number of thresholds={number_of_thresholds}")
    input_image_type = infer_itk_image_type(image, input_type)

    motsu = itk.OtsuMultipleThresholdsImageFilter[input_image_type, input_image_type].New()
    _ = motsu.SetInput(image)
    _ = motsu.SetNumberOfThresholds(number_of_thresholds)

    return motsu


@update
def itk_binary_morphological_closing(image, radius=1, foreground_value=1, input_type=None, **kwargs):
    logger = kwargs.get("logger", default_logger)
    logger.debug(f"Binary Morphological Closing: - radius={radius} - foreground_value={foreground_value} - structuring element=Ball")
    input_image_type = infer_itk_image_type(image, input_type)
    _, dimension = itk.template(image)[1]

    StructuringElementType = itk.FlatStructuringElement[dimension]
    structuring_element = StructuringElementType.Ball(radius)

    ClosingFilterType = itk.BinaryMorphologicalClosingImageFilter[input_image_type, input_image_type, StructuringElementType]
    closing = ClosingFilterType.New()
    closing.SetInput(image)
    closing.SetKernel(structuring_element)
    closing.SetForegroundValue(foreground_value)  # Intensity value to erode

    return closing


@update
def itk_cast(image, new_type=itk.UC, **kwargs):
    '''
    Cast image voxel type to new_type. Preserve image dimensions

    Parameters
    ----------
    image: itk.Image
        Image to cast
    new_type: itk voxel type (i.e. itk.UC)
        new voxel type
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    cast: itk.CastImageFilter
        filter is updated by default.
        To not update the instance pecify update=False as kwargs.
    '''
    logger = kwargs.get("logger", default_logger)
    pixel_type, dimension = itk.template(image)[1]
    logger.debug(f'Casting image from {pixel_type} to {new_type}')
    input_image_type = itk.Image[pixel_type, dimension]
    output_image_type = itk.Image[new_type, dimension]

    cast = itk.CastImageFilter[input_image_type, output_image_type].New()
    _ = cast.SetInput(image)

    return cast

@update
def itk_binary_threshold(image, lower_thr=0, upper_thr=0, inside_value=1,
                        outside_value=0, input_type=None, output_type=None,
                         **kwargs):
    '''
    Apply a threshold in a specified interval and return a binary image. The
    values outside the inteval are setted to outside_value, the ones inside to
    inside_value.

    Parameters
    ----------
    image: itk.Image
        itk image to process
    lower_thr: PixelType
        lower threshold value
    upper_thr: PixelType
        upper threshold value
    inside_value: PixelType
        value to which set the voxels inside the specified inteval
    outside_value: PixelType
        value to which set the voxels outside the specified inteval
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
        input image type. If not specified it is iferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
        output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators
    Return
    ------
    thr: itk.BinaryThresholdImageFilter
        New instance of binary threshold filter. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.
    '''
    logger = kwargs.get("logger", default_logger)
    logger.debug(f'Binary Threshold: -Upper thr: {upper_thr} - Lower \
    thr: {lower_thr}')

    input_image_type = infer_itk_image_type(image, input_type)
    output_image_type = infer_itk_image_type(image, output_type)

    thr = itk.BinaryThresholdImageFilter[
                                        input_image_type,
                                        output_image_type
                                        ].New()
    _ = thr.SetInput(image)
    _ = thr.SetLowerThreshold(lower_thr)
    _ = thr.SetUpperThreshold(upper_thr)
    _ = thr.SetInsideValue(inside_value)
    _ = thr.SetOutsideValue(outside_value)

    return thr

@update
def itk_add_images(image1, image2, input1_type=None, input2_type=None, output_type=None, **kwargs):
    """
    """
    logger = kwargs.get("logger", default_logger)
    logger.debug("Add Images")
    input1_image_type = infer_itk_image_type(image1, input1_type)
    input2_image_type = infer_itk_image_type(image2, input1_type)
    output_image_type = infer_itk_image_type(image1, output_type)


    add_image = itk.AddImageFilter[input1_image_type, input2_image_type, output_image_type].New()
    _ = add_image.SetInput1(image1)
    _ = add_image.SetInput2(image2)

    return add_image

@update
def itk_slice_by_slice(image, pipeline, slicing_dimension=2, **kwargs):
    logger = kwargs.get("logger", default_logger)
    logger.debug(f"Slice By Slice: slicing_dimension={slicing_dimension}")
    pixel_type, dimension = itk.template(image)[1]
    image_type = itk.Image[pixel_type, dimension]


    filter_ = itk.SliceBySliceImageFilter[image_type, image_type].New()
    _ = filter_.SetInput(image)
    _ = filter_.SetFilter(pipeline)
    _ = filter_.SetDimension(slicing_dimension)

    return filter_

@update
def itk_invert_intensity(image, maximum=1, input_type=None, output_type=None, **kwargs):
    """
    """
    logger = kwargs.get("logger", default_logger)
    logger.debug(f"Invert Intensity: maximum={maximum}")
    input_image_type = infer_itk_image_type(image, input_type)
    output_image_type = infer_itk_image_type(image, output_type)


    inverter = itk.InvertIntensityImageFilter[input_image_type, output_image_type].New()
    _ = inverter.SetInput(image)
    _ = inverter.SetMaximum(maximum)

    return inverter


@update
def flood_fill_2d(image, **kwargs):
    logger = kwargs.get("logger", default_logger)
    logger.debug("Fllod Fill 2d")

    PixelType, Dimension = itk.template(image)[1]

    invert = itk_invert_intensity(image)
    filled = itk.ConnectedComponentImageFilter[itk.Image[PixelType, 2], itk.Image[PixelType, 2]].New()

    filled = itk_slice_by_slice(invert.GetOutput(), filled)

    filled = itk_binary_threshold(filled.GetOutput(), upper_thr=700, lower_thr=2)

    filled = itk_add_images(filled.GetOutput(), image)

    return filled

@update
def itk_relabel_components(image,
                           sort_by_object_size=True,
                           minimum_object_size=None,
                           number_of_object_to_print=None,
                           input_type=None, output_type=None,
                           **kwargs):
    '''
    Relabel the components in an image such that consecutive labels are used.

    Parameters
    ----------
    image: itk.Image
        label image to relabel
    sort_by_object_size: bool
        specify if sort the object by their size
    minimum_object_size: int
        Set the minimum size in pixels for an object. All objects smaller than
        this size will be discarded and will not appear in the output label map
    number_of_object_to_print: int
        Set the number of objects enumerated and described when the filter is
        printed.
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
         output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    relabeler: itk::RelabelComponentImageFilter
        itk::RelabelComponentImageFilter instance. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.

    '''
    logger = kwargs.get("logger", default_logger)
    logger.debug(f'Relabel Components. - Sort by Size: {sort_by_object_size}  \
    - minimum size: {minimum_object_size} - number of objects to print: {number_of_object_to_print}')

    InputType = infer_itk_image_type(image, input_type)
    OutputType = infer_itk_image_type(image, output_type)

    relabeler = itk.RelabelComponentImageFilter[InputType, OutputType].New()
    _ = relabeler.SetInput(image)
    _ = relabeler.SetSortByObjectSize(sort_by_object_size)

    return relabeler


@update
def itk_connected_components(image, fully_connected=False, background_value=0,
                             input_type=None, output_type=None, **kwargs):
    '''
    Label the object of a binary image. Assign a Unique Label to each distinct
    object.

    Parameters
    ----------
    image: itk.Image
        binary image to process
    fully_connected: bool
        Set whether the connected components are defined strictly by face
        connectivity or by face+edge+vertex connectivity
    background_value: voxel type
        Set the pixel intensity to be used for background (non-object)
        regions of the image in the output
    input_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
        input image type. If not specified it is inferred from the input image
    output_type : itk.Image type (i.e.itk.Image[itk.UC, 2])
        output image type. If not specified it is iferred from the input image
    kwargs:
        keyword arguments to control the behaviour of deorators

    Return
    ------
    connected: itk.ConnectedComponentImageFilter
        itk.ConnectedComponentImageFilter instance. As default the instance is
        updated. To not update the instance pecify update=False as kwargs.
    '''
    logger = kwargs.get("logger", default_logger)
    logger.debug(
        f'Computing Connected Components: - fully_connected: {fully_connected} - background_value: {background_value}')

    input_image_type = infer_itk_image_type(image, input_type)
    output_image_type = infer_itk_image_type(image, output_type)

    connected = itk.ConnectedComponentImageFilter[
                                                input_image_type,
                                                output_image_type].New()
    _ = connected.SetInput(image)
    _ = connected.SetFullyConnected(fully_connected)
    _ = connected.SetBackgroundValue(background_value)

    return connected

def run(image, logger):

    _, dimension = itk.template(image)[1]

    if dimension not in [2, 3]:
        logger.error(f"Scan Dimension should be 2D or 3D, found {dimension}D instead")

    logger.info("Define the head region")
    brain = itk_multi_otsu_threshold(image,  1, logger=logger)
    brain = itk_cast(brain.GetOutput(), itk.SS, logger=logger)
    brain = itk_binary_morphological_closing(brain.GetOutput(), 5, logger=logger)
    brain = flood_fill_2d(brain.GetOutput(), logger=logger)
    brain = itk_connected_components(brain.GetOutput(), logger=logger)
    brain = itk_relabel_components(brain.GetOutput(), logger=logger)
    brain = itk_binary_threshold(brain.GetOutput(), lower_thr=1, upper_thr=1, logger=logger)
    brain = itk_cast(brain.GetOutput(), itk.UC, logger=logger)
    return brain


def main():

    args = parse_args()

    logger = logging.getLogger(__file__)
    logging.basicConfig(level=LOG_LEVELS[min(args.verbose, 3)])

    logger.info(f"Reading scan from {args.input}")
    scan = itk.imread(args.input, itk.F)
    
    brain = run(scan, logger)

    logger.info(f"Writing the Results on {args.output}")

    _ = itk.imwrite(brain.GetOutput(), args.output)

if __name__ == "__main__":
    main()