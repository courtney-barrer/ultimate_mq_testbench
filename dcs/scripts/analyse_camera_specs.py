#!/usr/bin/env python3
"""Compare prompted dark/flat measurements with the saved per-camera references.
Run from repo root: python dcs/scripts/analyse_camera_specs.py data/spec_tests/camera_1/RUN_DIRECTORY
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIT_MIN_ADU = 500
FIT_MAX_ADU = 35000
MAX_PAIR_MEAN_CHANGE = 0.01       # Initial source-stability screen, not a correction


def setting_matches(key, actual, expected):
    # FITS decimal cards and JSON can round the same exposure differently.
    # Allow only serialization-scale differences, not camera-setting changes.
    if key == 'EXPTIME':
        try:
            actual, expected = float(actual), float(expected)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(actual) and np.isfinite(expected)
                    and np.isclose(actual, expected, rtol=1e-12, atol=1e-15))
    return actual == expected


def signatures_match(left, right):
    return left.keys() == right.keys() and all(
        setting_matches(key, left[key], right[key]) for key in left)


def statistics(folder, names, expected):
    mean=m2=None; n=0; pair_variances=[]; pair_changes=[]; maximum=0; signature=None
    for name in names:
        with fits.open(folder/name,memmap=False) as h:
            head=h[0].header; data=h[0].data
            if data.ndim!=3 or len(data)<2 or len(data)%2 or head['NFRAMES']!=len(data):
                raise ValueError('Expected complete even-sized frame cubes')
            for key,value in expected.items():
                if not setting_matches(key, head.get(key), value):
                    raise ValueError(f'{name}: mismatched {key}: FITS={head.get(key)!r}, expected={value!r}')
            # Match all geometry and the main processing settings across files.
            keys=['MODEL','SERIAL','SCANMODE','READID','EXPTIME','XSTART','XEND','YSTART','YEND','XBINNING','YBINNING']
            sig={k:head.get(k) for k in keys}
            for k in head:
                if k.lower().startswith('dcam ') and any(s in k.lower() for s in ['sensor_mode','bit_per_channel','defect_correct','sensitivity','binning','trigger_source']):
                    sig[k]=head[k]
            if signature is None:signature=sig
            elif not signatures_match(sig,signature):raise ValueError('Configuration changed within sequence')
            if mean is not None and data.shape[1:]!=mean.shape:raise ValueError('Frame shape changed')
            maximum=max(maximum,int(data.max()))
            for frame in data:
                f=frame.astype(float)
                if not np.isfinite(f).all():raise ValueError('Non-finite pixels')
                if mean is None:mean=np.zeros_like(f);m2=np.zeros_like(f)
                n+=1; delta=f-mean;mean+=delta/n;m2+=delta*(f-mean)
            for i in range(0,len(data),2):
                difference=data[i].astype(float)-data[i+1].astype(float)
                # Fixed spatial pattern cancels; subtracting the pair's spatial mean
                # rejects additive common-mode fluctuations, not source instability.
                pair_variances.append(float(difference.var(ddof=1)/2))
                pair_changes.append(float(abs(difference.mean())))
    if n<4:raise ValueError('At least four frames required')
    return {'mean':mean,'variance':m2/(n-1),'n':n,'pair_variance':float(np.mean(pair_variances)),
        'pair_se':float(np.std(pair_variances,ddof=1)/np.sqrt(len(pair_variances))),
        'pair_change_max':max(pair_changes),'max':maximum,'signature':signature}


def fit_ptc(x,y):
    if len(x)<3 or len(np.unique(x))<3:raise ValueError('Need >=3 distinct usable light levels')
    coef,cov=np.polyfit(x,y,1,cov=True)
    if coef[0]<=0:raise ValueError('PTC slope is not positive')
    return 1/coef[0],float(np.sqrt(cov[0,0])/coef[0]**2),coef


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,help='Optional new output directory')
    p.add_argument('--fit-min',type=float,default=FIT_MIN_ADU)
    p.add_argument('--fit-max',type=float,default=FIT_MAX_ADU)
    a=p.parse_args()
    if not 0<a.fit_min<a.fit_max:p.error('Require 0 < fit-min < fit-max')
    run=json.loads((a.run/'run.json').read_text())
    if run['status']!='complete':raise ValueError('Run is incomplete; finish/repeat acquisition before comparison')
    out=a.output or a.run/'analysis'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    results=[]
    for entry in run['modes']:
        mode=entry['name']; ref=run['reference']['modes'][mode]; folder=a.run/mode
        expected={'CAMID':run['camera'],'SCANMODE':mode,'EXPTIME':entry['exposure_s'],'READID':entry['readout_id']}
        dark=statistics(folder,entry['dark_files'],{**expected,'IMAGETYP':'DARK'})
        serial=str(dark['signature']['SERIAL']).removeprefix('S/N:').strip()
        if serial!=str(run['reference']['serial']).removeprefix('S/N:').strip():
            raise ValueError('Measured serial does not match reference camera')
        noise=np.sqrt(dark['variance']); rms=float(np.sqrt(dark['variance'].mean()))
        row={'camera':run['camera'],'serial':run['reference']['serial'],'mode':mode,
             'exposure_s':entry['exposure_s'],'dark_frames':dark['n'],
             'reference_noise_e_rms':ref['read_noise_e_rms'],'reference_gain_e_per_adu':ref['gain_e_per_adu'],
             'dark_noise_rms_adu':rms,'dark_noise_median_adu':float(np.median(noise)),
             'noise_e_using_reference_gain':rms*ref['gain_e_per_adu'],
             'noise_difference_percent_using_reference_gain':100*(rms*ref['gain_e_per_adu']/ref['read_noise_e_rms']-1),
             'measured_gain_e_per_adu':None,'gain_fit_se_e_per_adu':None,'gain_difference_percent':None,
             'noise_e_using_measured_gain':None,'noise_difference_percent_using_measured_gain':None,
             'comparison_status':'Reference comparison only; no acceptance tolerances supplied'}
        map_header=fits.Header({'BUNIT':'ADU','SERIAL':serial,'SCANMODE':mode,'EXPTIME':entry['exposure_s'],
            'NFRAMES':dark['n'],'REFGAIN':ref['gain_e_per_adu'],'REFNOISE':ref['read_noise_e_rms']})
        noise_header=fits.Header({'BUNIT':'ADU','DDOF':1})
        fits.HDUList([fits.PrimaryHDU(dark['mean'].astype('float32'),header=map_header),
            fits.ImageHDU(noise.astype('float32'),header=noise_header,name='NOISE_ADU')]).writeto(out/f'{mode}_dark_maps.fits',checksum=True)
        points=[]
        for level in entry['flat_levels']:
            flat=statistics(folder,level['files'],{**expected,'IMAGETYP':'FLAT'})
            if not signatures_match(flat['signature'],dark['signature']):raise ValueError('Flat/dark camera settings mismatch')
            signal=float((flat['mean']-dark['mean']).mean())
            variance=flat['pair_variance']-dark['pair_variance']
            drift=flat['pair_change_max']/max(abs(signal),1)
            reasons=[]
            if not a.fit_min<=signal<=a.fit_max:reasons.append('outside fit range')
            if variance<=0:reasons.append('nonpositive variance')
            if flat['max']>=level['ceiling_adu']:reasons.append('raw ceiling reached')
            if drift>MAX_PAIR_MEAN_CHANGE:reasons.append('pair mean instability')
            points.append({'level':level['level'],'signal_adu':signal,'variance_adu2':variance,
                'variance_se_adu2':float(np.hypot(flat['pair_se'],dark['pair_se'])),
                'pair_mean_change_fraction':drift,'raw_max_adu':flat['max'],
                'used':not reasons,'exclusion':'; '.join(reasons)})
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        axes[0].hist(noise.ravel(),bins=100,log=True)
        axes[0].axvline(ref['read_noise_e_rms']/ref['gain_e_per_adu'],color='tab:red',label='Sheet noise / sheet gain')
        axes[0].set(xlabel='Pixel temporal noise [ADU RMS]',ylabel='Pixel count (log)',title=f'{mode}: dark noise')
        axes[0].legend()
        if points:
            selected=[v for v in points if v['used']]
            x=np.array([v['signal_adu'] for v in selected]);y=np.array([v['variance_adu2'] for v in selected])
            axes[1].scatter([v['signal_adu'] for v in points],[v['variance_adu2'] for v in points],color='gray',label='All levels')
            axes[1].scatter(x,y,label='Fit levels')
            try:
                gain,se,coef=fit_ptc(x,y)
                row.update(measured_gain_e_per_adu=float(gain),gain_fit_se_e_per_adu=se,
                    gain_difference_percent=float(100*(gain/ref['gain_e_per_adu']-1)),
                    noise_e_using_measured_gain=float(rms*gain),
                    noise_difference_percent_using_measured_gain=float(100*(rms*gain/ref['read_noise_e_rms']-1)))
                residual=y-np.polyval(coef,x)
                row['ptc_rms_residual_adu2']=float(np.sqrt(np.mean(residual**2)))
                axes[1].plot(x,np.polyval(coef,x),label=f'Gain = {gain:.4f} e/ADU')
            except ValueError as exc:
                row['comparison_status']='Gain not measured: '+str(exc)
            with (out/f'{mode}_ptc_points.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
            axes[1].legend()
        else:axes[1].text(.5,.5,'Dark-only run: gain not measured',ha='center',transform=axes[1].transAxes)
        axes[1].set(xlabel='Mean flat minus dark [ADU]',ylabel='Pair variance minus dark [ADU²]',title='Photon transfer')
        fig.savefig(out/f'{mode}_comparison.png',dpi=150);plt.close(fig)
        results.append(row)
        print(mode,': RMS noise',round(rms,3),'ADU; using reference gain:',round(row['noise_e_using_reference_gain'],3),'e')
        if row['measured_gain_e_per_adu'] is not None:print('  Measured gain:',round(row['measured_gain_e_per_adu'],4),'e/ADU')
    keys=list(dict.fromkeys(k for r in results for k in r))
    with (out/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(results)
    (out/'comparison.json').write_text(json.dumps({'results':results,'fit_range_adu':[a.fit_min,a.fit_max],
        'caveats':['No formal acceptance limits supplied.',
        'Short-dark temporal noise includes dark shot noise and drift. RMS across pixels is compared, with no pixel rejection.',
        'Measured PTC gain is an effective gain for this ROI/mode. Fit SE excludes systematic errors and shared-dark covariance.',
        'Inspect PTC curvature, source stability, uniformity and clipping. A successful fit does not establish linearity.',
        'Fixed exposure, manually varied illumination; this does not itself measure response linearity versus calibrated flux.',
        'Exposure readback may be quantized. Centre position and 16-bit output are test choices.',
        'No EMVA certification is claimed.']},indent=2,allow_nan=False)+'\n')
    print('Saved comparison:',out)


if __name__=='__main__':main()
