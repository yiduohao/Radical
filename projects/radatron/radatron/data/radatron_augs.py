
import cv2
import numpy as np
from numpy import matlib
from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.config import *
from detectron2.data.transforms.augmentation_impl import RandomCrop, RandomRotation, RandomContrast, RandomBrightness, RandomFlip
from detectron2.data.transforms.transform import RotationTransform, NoOpTransform
from fvcore.transforms.transform import CropTransform, BlendTransform, HFlipTransform


class RadatronRandomFlip(RandomFlip):
    def __init__(self, prob=0.5, *, horizontal=True, vertical=False, gt_grid_width:int):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())
        self.gt_grid_width = gt_grid_width
        
    def get_transform(self, image):
        if isinstance(image, dict):
            h, w = image["image1"].shape[:2]
        else:
            h, w = image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return RadatronHFlip(w, gt_grid_width=self.gt_grid_width)
            else:
                return NoOpTransform()
        else:
            return NoOpTransform()

class RadatronHFlip(HFlipTransform):
    def __init__(self, width: int, gt_grid_width:int):
        self._set_attributes(locals())
        self.gt_grid_width=gt_grid_width
        
    def apply_image(self, img):
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        if isinstance(img, dict):
            out = {}
            if img["image1"].ndim <= 3:  # HxW, HxWxC
                out["image1"] = np.flip(img["image1"], axis=1).copy()
                out["image2"] = np.flip(img["image2"], axis=1).copy()
            else:
                out["image1"] = np.flip(img["image1"], axis=-2).copy()
                out["image2"] = np.flip(img["image2"], axis=-2).copy()

            return out

        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=1).copy()
        else:
            return np.flip(img, axis=-2).copy()
        
    def apply_rotated_box(self, rotated_boxes):
        rotated_boxes[:, 0] = self.gt_grid_width - rotated_boxes[:, 0]
        # Transform angle
        rotated_boxes[:, 4] = -rotated_boxes[:, 4]
        return rotated_boxes

class RadatronBlendTransform(BlendTransform):
    def apply_rotated_box(self, rotated_boxes):

        return rotated_boxes

class RadatronRandomBrightness(RandomBrightness):
    def get_transform(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return RadatronBlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)

class RadatronRandomContrast(RandomContrast):
    def get_transform(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return RadatronBlendTransform(src_image=image.mean(), src_weight=1 - w, dst_weight=w)


class RadatronRandomRotation(RandomRotation):
    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None, fill_zeros=False, gt_grid_h=None, gt_grid_w=None, gt_center=None):
            """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
            
            assert sample_style in ["range", "choice"], sample_style
            self.is_range = sample_style == "range"
            if isinstance(angle, (float, int)):
                angle = (angle, angle)
            #if center is not None and isinstance(center[0], (float, int)):
            #    center = (center, center)
            self._init(locals())
            self.fill_zeros=fill_zeros
            self.gt_grid_h = gt_grid_h
            self.gt_grid_w = gt_grid_w
            self.gt_center = gt_center
    def get_transform(self, image):
        if isinstance(image, dict):
            h, w = image["image1"].shape[:2]
        else:
            h, w = image.shape[:2]
        center = None
        angle = np.random.uniform(self.angle[0], self.angle[1])
        if self.center is not None:
            center = self.center

        if angle % 360 == 0:
            return NoOpTransform()

        
        return RadatronRotationBF(h, w, angle, expand=self.expand, center=center, interp=self.interp, fill_zeros=self.fill_zeros, gt_grid_h=self.gt_grid_h, gt_grid_w=self.gt_grid_w, gt_center=self.gt_center)


class RadatronRotationTransform(RotationTransform):
    def __init__(self, h, w, angle, expand=True, center=None, interp=None, fill_zeros=False):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        #super().__init__(h,w,angle)
        self.angle = angle
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand is False and fill_zeros is True:
            self.fill_zeros = True
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix_gt()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def create_rotation_matrix_gt(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
 
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        rot_im = cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)
        rot_im = rot_im[:,:,np.newaxis] if len(rot_im.shape)==2 else rot_im
        if self.expand is False and self.fill_zeros is True:
            zero_indices = np.where(rot_im==0)
            rot_im_nz = rot_im[rot_im!=0]
            fill_rand = np.random.rand(len(zero_indices[0]))*(np.percentile(rot_im_nz,10)-np.percentile(rot_im_nz, 1))+np.percentile(rot_im_nz,1) #np.random.normal(loc=np.mean(rot_im), scale=np.std(rot_im[rot_im!=0]), size=len(zero_indices[0]))#np.random.uniform(low=min_im, high=med_im, size=len(zero_indices[0]))
            rot_im[rot_im==0] = fill_rand
        return rot_im


    def apply_rotated_box(self, rotated_boxes):

        

        rotated_boxes[:,:2] = self.apply_coords(rotated_boxes[:,:2])
        rotated_boxes[:,4] += self.angle
        rotated_boxes[:,4] = (rotated_boxes[:,4] + 180) % 360 -180

        for idx, rot_box in enumerate(rotated_boxes):
            bb_xyxy = self.cwha2xyxy(rot_box)
            bb_y = np.sort(bb_xyxy[:,1])
            if np.max(bb_xyxy[:,0])<0 or np.min(bb_xyxy[:,0])>self.bound_w or np.min(bb_xyxy[:,1])>self.bound_h or np.max(bb_xyxy[:,1])<0:# or bb_y[1]<0: #or bb_xyxy[:,1].flatten().sort()[1]<0:
                rotated_boxes=np.delete(rotated_boxes, idx, 0)

        return rotated_boxes

    def cwha2xyxy(self, bb_cwha):
        if len(bb_cwha.shape)>2:
            bb_cwha = np.squeeze(bb_cwha)

        bb_xyxy = [[bb_cwha[0]-bb_cwha[2]/2, bb_cwha[1]-bb_cwha[3]/2],
                [bb_cwha[0]-bb_cwha[2]/2, bb_cwha[1]+bb_cwha[3]/2],
                [bb_cwha[0]+bb_cwha[2]/2, bb_cwha[1]+bb_cwha[3]/2],
                [bb_cwha[0]+bb_cwha[2]/2, bb_cwha[1]-bb_cwha[3]/2]];

        bb_xyxy = self.rotateRect(bb_xyxy, bb_cwha[4])

        return bb_xyxy


    def rotateRect(self, bb_xyxy, theta):
        if len(bb_xyxy)>2:
            bb_xyxy = np.squeeze(bb_xyxy)

        theta = theta*np.pi/180
        rot_mat = [[np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)]]

        center = np.mean(bb_xyxy, axis=0)
        bb_xyxy = bb_xyxy - np.matlib.repmat(center, 4, 1)
        bb_xyxy = np.matmul(bb_xyxy, rot_mat)

        bb_rot = bb_xyxy + np.matlib.repmat(center, 4, 1)

        return bb_rot


class RadatronRotationBF(RadatronRotationTransform):        
    def __init__(self, h, w, angle, expand=True, center=None, interp=None, fill_zeros=False, gt_grid_h=None, gt_grid_w=None, gt_center=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        #super().__init__(h,w,angle)
        self.angleBF = int(192*angle/180)
        self.angle = self.angleBF*180/192
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand is False and fill_zeros is True:
            self.fill_zeros = True
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self.gt_grid_h = gt_grid_h
        self.gt_grid_w = gt_grid_w
        self.gt_center = gt_center
        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix_gt()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)
        
        
    
    def create_rotation_matrix_gt(self, offset=0):
        center = (self.gt_center[0] + offset, self.gt_center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), -self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm


    def apply_image(self, img):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if isinstance(img, dict):
            img1 = img["image1"]
            img2 = img["image2"]
            if len(img1) == 0 or self.angle % 360 == 0:
                return img
            assert img1.shape[:2] == (self.h, self.w)
            az_trans = self.angleBF
            # T = np.float32([[1, 0, az_trans], [0, 1, 0]])
            # rotated_img = cv2.warpAffine(img, T, (self.w, self.h))
            rotated_img1 = np.roll(img1, az_trans, axis=1)
            rotated_img2 = np.roll(img2, az_trans, axis=1)

            return {
                "image1" : rotated_img1,
                "image2" : rotated_img2
            }
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        az_trans = self.angleBF
        # T = np.float32([[1, 0, az_trans], [0, 1, 0]])
        # rotated_img = cv2.warpAffine(img, T, (self.w, self.h))
        rotated_img = np.roll(img, az_trans, axis=1)

        return rotated_img
        
    
    def apply_rotated_box(self, rotated_boxes):
    
        

        rotated_boxes[:,:2] = self.apply_coords(rotated_boxes[:,:2])
        rotated_boxes[:,4] -= self.angle
        #rotated_boxes[:,4] = (rotated_boxes[:,4] + 180) % 360 -180

        for idx, rot_box in enumerate(rotated_boxes):
            bb_xyxy = self.cwha2xyxy(rot_box)
            if np.max(bb_xyxy[:,0])<=0 or np.min(bb_xyxy[:,0])>=self.gt_grid_w or np.min(bb_xyxy[:,1])>=self.gt_grid_h or np.max(bb_xyxy[:,1])<=0:# or bb_y[1]<0: #or bb_xyxy[:,1].flatten().sort()[1]<0:
                rotated_boxes=np.delete(rotated_boxes, idx, 0)

        return rotated_boxes

class RadatronRandomCrop(RandomCrop):

    def get_transform(self, image):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = int((h - croph + 1)/2)
        w0 = int((w - cropw + 1)/2)
        return CropTransform(w0, h0, cropw, croph)

def transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
  if annotation["bbox_mode"] == BoxMode.XYWHA_ABS:
    annotation["bbox"] = np.squeeze(transforms.apply_rotated_box(np.asarray([annotation["bbox"]])))
  else:
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_box([bbox])[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

  return annotation


