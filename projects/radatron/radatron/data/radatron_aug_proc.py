import numpy as np
from detectron2.config import *
from fvcore.transforms.transform import Transform, TransformList
from detectron2.data.transforms.augmentation import Augmentation, AugmentationList
from typing import List, Union, Optional


def apply_augmentations_twostream(augmentations: List[Union[Transform, Augmentation]], inputs):
    """
    Use ``T.AugmentationList(augmentations)(inputs)`` instead.
    """
    if isinstance(inputs["image1"], np.ndarray):
        # handle the common case of image-only Augmentation, also for backward compatibility
        image_only = True
        inputs = RadatronAugInput(inputs)
    else:
        image_only = False
    tfms = inputs.apply_augmentations(augmentations)
    return inputs.image if image_only else inputs, tfms


class RadatronAugInput:
    """
    Input that can be used with :meth:`Augmentation.__call__`.
    This is a standard implementation for the majority of use cases.
    This class provides the standard attributes **"image", "boxes", "sem_seg"**
    defined in :meth:`__init__` and they may be needed by different augmentations.
    Most augmentation policies do not need attributes beyond these three.

    After applying augmentations to these attributes (using :meth:`AugInput.transform`),
    the returned transforms can then be used to transform other data structures that users have.

    Examples:
    ::
        input = AugInput(image, boxes=boxes)
        tfms = augmentation(input)
        transformed_image = input.image
        transformed_boxes = input.boxes
        transformed_other_data = tfms.apply_other(other_data)

    An extended project that works with new data types may implement augmentation policies
    that need other inputs. An algorithm may need to transform inputs in a way different
    from the standard approach defined in this class. In those rare situations, users can
    implement a class similar to this class, that satify the following condition:

    * The input must provide access to these data in the form of attribute access
      (``getattr``).  For example, if an :class:`Augmentation` to be applied needs "image"
      and "sem_seg" arguments, its input must have the attribute "image" and "sem_seg".
    * The input must have a ``transform(tfm: Transform) -> None`` method which
      in-place transforms all its attributes.
    """

    # TODO maybe should support more builtin data types here
    def __init__(
        self,
        image: dict,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
    ):
        """
        Args:
            image (ndarray): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255]. The meaning of C is up
                to users.
            boxes (ndarray or None): Nx4 float32 boxes in XYXY_ABS mode
            sem_seg (ndarray or None): HxW uint8 semantic segmentation mask. Each element
                is an integer label of pixel.
        """
        #_check_img_dtype(image)
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)

    def apply_augmentations(
        self, augmentations: List[Union[Augmentation, Transform]]
    ) -> TransformList:
        """
        Equivalent of ``AugmentationList(augmentations)(self)``
        """
        return AugmentationList(augmentations)(self)


