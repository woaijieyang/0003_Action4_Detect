"""Shared fixed ROI coefficient; square sampling remains in action4_dataset."""
import math
from numbers import Real
DEFAULT_EXPANSION=2.25
SWEEP_EXPANSIONS=(1.0,1.1,1.2,1.25,1.5,2.25)
def validate_expansion(value):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or value<1:
        raise ValueError("Finite ROI expansion >=1 required")
    return float(value)

from dataclasses import dataclass
import numpy as np
from data import DataAdapter

@dataclass(frozen=True)
class RoiSpec:
    """One complete-body extent expanded once in native continuous pixel space."""
    full_body_bbox: tuple
    expansion: float
    output_size: int
    continuous_bounds: tuple

    @classmethod
    def from_full_box(cls,box,expansion,output_size=448):
        from data.action4_dataset import virtual_roi
        expansion=validate_expansion(expansion)
        if type(output_size) is not int or output_size<=0 or output_size%32:
            raise ValueError("ROI output size must be a positive multiple of32")
        bounds=virtual_roi(box,expansion)
        return cls(tuple(map(float,box)),expansion,output_size,tuple(map(float,bounds)))

    def __post_init__(self):
        from data.action4_dataset import virtual_roi
        expected=virtual_roi(self.full_body_bbox,validate_expansion(self.expansion))
        if (type(self.output_size) is not int or self.output_size<=0 or self.output_size%32
                or type(self.full_body_bbox) is not tuple or type(self.continuous_bounds) is not tuple
                or not np.array_equal(expected,self.continuous_bounds)):
            raise ValueError("RoiSpec must retain its unchanged full-body extent and single expansion")

    def affine(self,size=None):
        size=self.output_size if size is None else size
        if type(size) is not int or size<=0:raise ValueError("Positive integer destination grid required")
        if size==self.output_size:
            return DataAdapter.crop_affine(self.continuous_bounds,(size,size))
        down,_=DataAdapter.letterbox_affine((self.output_size,self.output_size),(size,size))
        return down@self.affine()

    def camera(self,K_native,size=None):
        K=np.asarray(K_native,dtype=float)
        if K.shape!=(3,3) or not np.isfinite(K).all():raise ValueError("Finite native3x3 intrinsics required")
        return self.affine(size)@K

    def record(self):
        return dict(schema="action4.full_body_roi_spec.v1",full_body_bbox=list(self.full_body_bbox),
            expansion=self.expansion,output_size=self.output_size,continuous_bounds=list(self.continuous_bounds),
            pixel_convention="native_integer_centres_continuous_crop_edges",
            padding="zero_before_normalization",clip_to_image=False)
