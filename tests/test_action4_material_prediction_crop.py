import numpy as np
from eval.action4_material_track_probe import prediction_crop

def test_initial_prediction_exact_legacy_parity():
    b=np.array([1880.2,900.7,2444.3,1456.2]);center=(b[:2]+b[2:])/2;size=np.ceil((b[2:]-b[:2])*1.5).astype(int);left=np.floor(center-size/2).astype(int);c=prediction_crop([b])
    assert (c.x0,c.y0,c.width,c.height)==(*left,*size)

def test_moving_object_covered_and_negative_padding_kept():
    boxes=np.array([[-20.,10.,80.,80.],[200.,900.,320.,990.]])
    c=prediction_crop(boxes);assert c.x0<0
    assert np.all(boxes[:,:2]>=np.array([c.x0,c.y0]))
    assert np.all(boxes[:,2:]<=np.array([c.x0+c.width,c.y0+c.height]))
    xy=np.array([[-10.,20.],[280.,950.]])
    assert np.allclose(c.to_native(c.to_model(xy)),xy,atol=1e-10)

def test_invalid_boxes_fail_closed():
    for boxes in [[],[[1,2,0,3]],[[float('nan'),0,1,2]]]:
        try:prediction_crop(boxes)
        except ValueError:pass
        else:raise AssertionError('invalid crop accepted')
