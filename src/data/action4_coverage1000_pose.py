"""Rigid view-axis arm inclination, preserving the wrist/body mounting matrix.

This broadens whole-arm image inclination; it is not an extra rotation of the
DJI relative to the wrist, an image warp, or scaling of any physical component.
"""
import math
import numpy as np
from data.action4_phrase2_geometry import axis_angle,rotation_for_arm
from data.action4_coverage1000_contract import SIDES

def augment_hints(runtime,cfg,slot,centers,widths,hints,rng):
    policy=cfg.get('view_axis_pose_augmentation')
    if not policy:return hints,{s:{} for s in SIDES},None
    size=np.array([runtime.scene.render.resolution_x,runtime.scene.render.resolution_y]);result=[];notes={}
    for i,side in enumerate(SIDES):
        RB,roll,direction=hints[i];old_axis=RB@runtime.arm_axis_B[side]
        # Far slots retain the broad physical footprint rather than being
        # implicitly pushed close by a narrow in-image body orientation.
        far=slot.get('depth_bands_by_side',{}).get(side)
        enabled=rng.random()<(policy['enabled_probability_far'] if far else policy['enabled_probability'])
        theta=0.;target=None
        if enabled:
            strong=rng.random()<(policy['strong_probability_small'] if widths[i]<.06 else policy['strong_probability_other'])
            magnitude=rng.uniform(*(policy['strong_azimuth_deg'] if strong else policy['regular_azimuth_deg']))
            sign=(-1 if side=='left' else 1)*(1 if rng.random()<policy['outward_probability'] else -1)
            target=sign*magnitude;old=math.degrees(math.atan2(old_axis[0],old_axis[1]))
            theta=old-target;ray=np.linalg.inv(runtime.K)@np.r_[np.array(centers[i])*size,1.]
            RB=axis_angle(ray,math.radians(theta))@RB
            axis=RB@runtime.arm_axis_B[side]
            if axis[1]<=1e-5:return None,None,side+':inclination_points_arm_upward'
            direction=axis/axis[1]
            zero=rotation_for_arm(runtime,side,0.,direction);rot=RB@zero.T;n=axis/np.linalg.norm(axis)
            sine=np.dot(n,[rot[2,1]-rot[1,2],rot[0,2]-rot[2,0],rot[1,0]-rot[0,1]])/2
            roll=math.degrees(math.atan2(sine,(np.trace(rot)-1)/2))
            if not np.allclose(rotation_for_arm(runtime,side,roll,direction),RB,atol=1e-8,rtol=0):raise ValueError('Rigid inclination decomposition failed')
        result.append((RB,float(roll),np.asarray(direction).tolist()))
        axis=RB@runtime.arm_axis_B[side]
        notes[side]={'view_axis_roll_deg':float(theta),'requested_arm_azimuth_deg':target,'actual_arm_axis_azimuth_deg':float(math.degrees(math.atan2(axis[0],axis[1]))),
            'pose_augmentation_applied':bool(enabled),'fixed_mount_changed':False}
    return result,notes,None
