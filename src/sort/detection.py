# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = self.xcycwh_to_tlwh(tlwh)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def xcycwh_to_tlwh(self,bbox_xywh):
        """Convert bounding box to format `(x1, y1, width,height)`, i.e.,
        `(top,shape)`.
        """
        bbox_xywh =np.asarray(bbox_xywh, dtype=np.float)
        bbox_xywh[0] = bbox_xywh[0] - bbox_xywh[2]/2.
        bbox_xywh[1] = bbox_xywh[1] - bbox_xywh[3]/2.
        return bbox_xywh

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
