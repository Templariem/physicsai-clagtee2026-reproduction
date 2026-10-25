import collections
import numpy as np
def average(records,key):
    groups=collections.defaultdict(list)
    for r in records:
        if r.get(key) is not None:groups[r['group']].append(r[key])
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
def summarize(records):
    keys=['air_mae_V','air_rmse_V','effect_mae_V','without_ruler_effect_mae_V','grounded_rule_mae_V','supported_mae_V','extrapolated_mae_V','mean_map_mae_V','direction_median_deg']
    result={key:average(records,key) for key in keys}
    result['images']=len(records);result['groups']=len(set(r['group'] for r in records))
    a,b=result['effect_mae_V'],result['without_ruler_effect_mae_V'];result['effect_improvement_vs_without_ruler']=None if a is None or not b else 1-a/b
    a,b=result['air_mae_V'],result['mean_map_mae_V'];result['improvement_vs_mean_map']=None if a is None or not b else 1-a/b
    result['by_camera']={c:{key:average([r for r in records if r['camera']==c],key) for key in keys} for c in sorted(set(r['camera'] for r in records))}
    result['by_ruler_presence']={str(present):dict(images=sum(r.get('ruler_present')==present for r in records),air_mae_V=average([r for r in records if r.get('ruler_present')==present],'air_mae_V')) for present in [True,False]}
    return result
def per_image(pred,truth,before,phi,valid,air,rule,meanmap,supported,row,direction=False):
    air=air.astype(bool);valid=valid.astype(bool);rule=rule.astype(bool)&valid;effect=(abs(before-truth)>.1)&air
    error=abs(pred-truth)
    def mae(region):return float(error[region].mean()) if region.any() else None
    record=dict(id=row['id'],camera=row['camera'],group=row['group'],ruler_present=row['scene']['mode']=='ground',air_pixels=int(air.sum()),effect_pixels=int(effect.sum()),
                air_mae_V=mae(air),air_rmse_V=float(np.sqrt(((pred-truth)[air]**2).mean())),effect_mae_V=mae(effect),
                without_ruler_effect_mae_V=float(abs(before-truth)[effect].mean()) if effect.any() else None,
                grounded_rule_mae_V=mae(rule),supported_mae_V=mae(air&supported),extrapolated_mae_V=mae(air&~supported),
                mean_map_mae_V=float(abs(meanmap-truth)[air].mean()),direction_median_deg=None,
                finite=bool(np.isfinite(pred).all()))
    if direction:
        import cv2
        # Both derivatives are in image coordinates, V/pixel; not measured V/m.
        gy,gx=np.gradient(truth);py,px=np.gradient(pred);norm=np.hypot(gx,gy);pn=np.hypot(px,py)
        interior=cv2.erode(air.astype('uint8'),np.ones((3,3),np.uint8))>0;good=interior&(norm>.005)&(pn>1e-5)
        if good.sum()>30:
            cos=np.clip((gx[good]*px[good]+gy[good]*py[good])/(norm[good]*pn[good]),-1,1)
            record['direction_median_deg']=float(np.median(np.degrees(np.arccos(cos))))
    return record
