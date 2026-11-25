#!/usr/bin/env python3
"""Michelson/SLM characterisation with a simple live GUI and conditional phase fit.

Run --help for options. Offline --analyse never imports hardware drivers.
No analysis/acquisition has been executed by the author. Read the accompanying README.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from astropy.io import fits


class Cancelled(Exception):
    pass


class FitLimit(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='michelson_v2_run01')
    p.add_argument('--analyse',type=Path)
    p.add_argument('--fx',type=float,default=0); p.add_argument('--fy',type=float,default=9)
    p.add_argument('--baseline',type=int,default=127)
    p.add_argument('--max-amplitude',type=float,default=80)
    p.add_argument('--step',type=float,default=8)
    p.add_argument('--preview-amplitude',type=float,default=40)
    p.add_argument('--nframes',type=int,default=3)
    p.add_argument('--width',type=int,default=9)
    p.add_argument('--screen',default='1'); p.add_argument('--shm',default='dcam0_data')
    p.add_argument('--settle',type=float,default=.3); p.add_argument('--timeout',type=float,default=5.)
    p.add_argument('--saturation',type=float,default=65535.)
    p.add_argument('--bias',type=float,default=None,help='Measured camera offset; offline defaults to saved setting')
    p.add_argument('--wavelength-nm',type=float,default=None,help='Offline defaults to saved wavelength; acquisition default 633')
    p.add_argument('--camera-period',type=float,help='Known projected command period along camera cut, fixed during fit')
    p.add_argument('--smooth-px',type=float,default=3.,help='Gaussian sigma in camera pixels; model and data filtered identically')
    p.add_argument('--min-illumination',type=float,default=.2)
    p.add_argument('--opd-max',type=float,default=8.,help='Upper search bound nm OPD/grey; positive magnitude only')
    p.add_argument('--profile-points',type=int,default=9,help='Fixed-gain diagnostic grid including zero')
    p.add_argument('--fit-samples',type=int,default=72)
    p.add_argument('--max-nfev',type=int,default=80,help='Evaluation limit per optimizer run')
    p.add_argument('--fit-seconds',type=float,default=180.,help='Total numerical fitting time budget; partial diagnostics saved')
    a=p.parse_args()
    numeric=[a.fx,a.fy,a.max_amplitude,a.step,a.preview_amplitude,a.settle,a.timeout,a.saturation,
             a.smooth_px,a.min_illumination,a.opd_max,a.fit_seconds]
    numeric += [v for v in (a.bias,a.wavelength_nm,a.camera_period) if v is not None]
    if not np.all(np.isfinite(numeric)): p.error('Numeric options must be finite')
    if not 0<=a.baseline<=255: p.error('Baseline must be in 0..255')
    if not 0<a.max_amplitude<=min(a.baseline,255-a.baseline): p.error('Sweep amplitude would clip command range 0..255')
    if not 0<a.preview_amplitude<=min(a.baseline,255-a.baseline): p.error('Preview amplitude would clip command range')
    if min(a.step,a.nframes,a.width,a.timeout,a.opd_max,a.fit_seconds,a.max_nfev)<=0: p.error('Counts/steps/limits must be positive')
    if a.width%2!=1: p.error('Strip width must be odd')
    if a.settle<0 or a.smooth_px<0: p.error('Settle and smoothing must be nonnegative')
    if not 0<a.min_illumination<1: p.error('Minimum illumination must be between 0 and 1')
    if not 32<=a.fit_samples<=128 or not 5<=a.profile_points<=21: p.error('Fit samples: 32..128; profile points: 5..21')
    if not np.hypot(a.fx,a.fy): p.error('Nonzero spatial frequency required')
    if a.camera_period is not None and a.camera_period<=0: p.error('Period must be positive')
    if a.wavelength_nm is not None and a.wavelength_nm<=0: p.error('Wavelength must be positive')
    if a.bias is not None and a.saturation<=a.bias: p.error('Saturation must be above camera bias')
    if not a.analyse and 2*(int(np.ceil(a.max_amplitude/a.step))+1)*a.nframes>96:
        p.error('Limit acquisition to 96 exposures; increase --step or reduce --nframes')
    return a


def geometry(points,width,shape):
    p0,p1=np.asarray(points,float); length=np.linalg.norm(p1-p0)
    if length<40: raise ValueError('Draw a cut at least 40 pixels long across several fringes')
    direction=(p1-p0)/length; x=np.linspace(0,length,min(1024,int(length)+1))
    centre=p0[:,None]+direction[:,None]*x
    normal=np.array([-direction[1],direction[0]])
    xy=centre[:,None,:]+normal[:,None,None]*np.arange(-(width//2),width//2+1)[None,:,None]
    if np.any(xy[0]<1) or np.any(xy[1]<1) or np.any(xy[0]>shape[1]-2) or np.any(xy[1]>shape[0]-2):
        raise ValueError('Entire strip must lie inside the image')
    return x,xy


def extract(frame,xy,saturation):
    xx,yy=np.floor(xy[0]).astype(int),np.floor(xy[1]).astype(int)
    bad=np.zeros(xx.shape,bool)
    for dy in (0,1):
        for dx in (0,1):
            value=frame[yy+dy,xx+dx]
            bad |= ~np.isfinite(value)|(value>=saturation)
    strip=map_coordinates(frame,[xy[1],xy[0]],order=1,output=np.float64,prefilter=False)
    valid=~bad.any(axis=0); cut=strip.mean(axis=0); cut[~valid]=np.nan
    return cut,valid


def next_frame(cam,previous,timeout,pump):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        pump()
        count=int(cam.md.cnt0)
        if count<previous: raise RuntimeError('Camera counter reset')
        if count!=previous and not cam.md.write:
            frame=np.array(cam,copy=True)
            if int(cam.md.cnt0)==count and not cam.md.write: return count,frame,utc()
        time.sleep(.002)
    raise TimeoutError('No new stable camera frame received')


def smooth_blocks(values,mask,sigma):
    """Filter the last axis in separate valid segments; leave invalid values NaN."""
    result=np.full(values.shape,np.nan,dtype=float)
    edges=np.diff(np.r_[False,mask,False].astype(int))
    for start,stop in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        result[...,start:stop]=(gaussian_filter1d(values[...,start:stop],sigma,axis=-1)
                               if sigma>0 else values[...,start:stop])
    return result


PARAMS=['log_C0','log_C_linear','log_C_curvature','contrast_logit','contrast_curvature',
        'contrast_centre','static_phase','static_tilt','static_curvature','command_cycles',
        'command_phase','opd_nm_per_grey']


class InterferenceFit:
    """I = bias + C(x) + B(x)*cos(phi_static(x)+piston_f+2pi*s*g*M(x)/lambda).

    Shared Gaussian-shaped C,B; B<=C. Each exposure retains its own piston.
    Analytic sparse Jacobian. Spatial smoothing is applied to prediction AND data.
    """
    def __init__(self,x,cuts,valid,amps,a,pump):
        self.a=a; self.x=x; self.amps=amps; self.pump=pump
        self.u=(x-x[0])/(x[-1]-x[0])-.5; self.t=self.u+.5
        self.factor=2*np.pi/a.wavelength_nm; self.nf=len(amps)
        self.sigma=a.smooth_px/np.median(np.diff(x))
        common=valid.all(axis=0)&np.isfinite(cuts).all(axis=0)
        centre=np.median(cuts[:,common]-a.bias,axis=0)
        positive=centre>0
        if positive.sum()<40: raise ValueError('Need at least 40 unsaturated positions above measured camera bias')
        pos=np.flatnonzero(common)[positive]
        # Only a seed/edge mask, not treated as a measured independent envelope.
        guess=np.polyfit(self.u[pos],np.log(centre[positive]),2)
        if guess[0]>0: guess=np.r_[0.,np.polyfit(self.u[pos],np.log(centre[positive]),1)]
        broad=np.exp(np.polyval(guess,self.u))
        common &= broad>=a.min_illumination*np.max(broad[pos])
        edges=np.diff(np.r_[False,common,False].astype(int)); eligible=np.zeros(len(x),bool)
        trim=int(np.ceil(4*self.sigma))
        for start,stop in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if stop-start>2*trim+8: eligible[start+trim:stop-trim if trim else stop]=True
        positions=np.flatnonzero(eligible)
        if len(positions)<32: raise ValueError('Too few clean samples after trimming filter edges; reduce smoothing or reselect')
        self.idx=positions[np.unique(np.linspace(0,len(positions)-1,min(a.fit_samples,len(positions))).astype(int))]
        self.mask=common; self.eligible=eligible
        self.scale=float(np.median(cuts[:,self.idx]-a.bias))
        if not np.isfinite(self.scale) or self.scale<=0: raise ValueError('Invalid intensity scale')
        self.raw=(cuts-a.bias)/self.scale
        self.filtered=smooth_blocks(self.raw,common,self.sigma)
        self.data=self.filtered[:,self.idx]
        guess=np.polyfit(self.u[pos],np.log(np.maximum(broad[pos]/self.scale,.001)),2)
        self.seedC=[np.clip(guess[2],-4,2),np.clip(guess[1],-7,7),np.clip(-guess[0],.01,19)]
        self.deadline=time.monotonic()+a.fit_seconds
        self.cache=None

    def check(self):
        self.pump()
        if time.monotonic()>self.deadline: raise FitLimit('Fit time budget reached')

    def compute(self,p,derivative=False):
        u=self.u; t=self.t
        c0,c1,c2,v,q,mu,phase,tilt,curve,k,theta,s=p[:12]
        C=np.exp(c0+c1*u-c2*u*u)
        vis=1/(1+np.exp(-v)); B=C*vis*np.exp(-q*(u-mu)**2)
        arg=2*np.pi*k*t+theta; mode=np.sin(arg)
        phi=phase+tilt*u+curve*u*u+np.r_[0.,p[12:]][:,None]+self.factor*s*self.amps[:,None]*mode
        osc=B*np.cos(phi); predicted=C+osc
        if not derivative: return predicted,C,B
        D=-B*np.sin(phi)
        common=D*self.factor*s*self.amps[:,None]*np.cos(arg)
        derivatives=[predicted,predicted*u,-predicted*u*u,osc*(1-vis),
                     -osc*(u-mu)**2,osc*2*q*(u-mu),D,D*u,D*u*u,
                     common*2*np.pi*t,common,D*self.factor*self.amps[:,None]*mode]
        filtered=smooth_blocks(np.asarray(derivatives),self.mask,self.sigma)[...,self.idx]
        n=self.nf*len(self.idx)
        rows=np.tile(np.arange(n),12)
        cols=np.repeat(np.arange(12),n)
        vals=filtered.reshape(-1)
        # Frame piston contributes only to that exposure's rows. Frame 0 fixes gauge.
        if self.nf>1:
            dp=smooth_blocks(D,self.mask,self.sigma)[:,self.idx]
            rows=np.r_[rows,np.arange(len(self.idx),n)]
            cols=np.r_[cols,np.repeat(np.arange(12,12+self.nf-1),len(self.idx))]
            vals=np.r_[vals,dp[1:].ravel()]
        return csr_matrix((vals,(rows,cols)),shape=(n,12+self.nf-1))

    def residual(self,p):
        self.check()
        prediction=self.compute(p)[0]
        return (smooth_blocks(prediction,self.mask,self.sigma)[:,self.idx]-self.data).ravel()

    def initial(self,cycles,slope,phase):
        return np.r_[self.seedC,-1.7,.5,0.,phase,0.,0.,cycles,0.,slope,np.zeros(self.nf-1)]

    def solve(self,p,period_fixed=None,slope_fixed=None):
        lo=np.r_[-5,-8,0,-7,0,-1,-10,-20,-40,.5,-10,0,np.full(self.nf-1,-10)]
        hi=np.r_[3,8,20,7,20,1,10,20,40,min(20,len(self.idx)/4),10,self.a.opd_max,np.full(self.nf-1,10)]
        free=np.ones(len(p),bool); p=p.copy()
        if period_fixed is not None: p[9]=period_fixed; free[9]=False
        if slope_fixed is not None: p[11]=slope_fixed; free[11]=False
        p[free]=np.clip(p[free],lo[free]+1e-8,hi[free]-1e-8)
        def expand(z):
            full=p.copy(); full[free]=z; return full
        def jac(z):
            self.check(); return self.compute(expand(z),True)[:,free]
        result=least_squares(lambda z:self.residual(expand(z)),p[free],jac=jac,
                             bounds=(lo[free],hi[free]),tr_solver='lsmr',x_scale='jac',
                             max_nfev=self.a.max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        fitted=expand(result.x)
        active=np.flatnonzero(free)[np.flatnonzero(result.active_mask)]
        active=[int(i) for i in active if not (i in (2,4) and fitted[i]<1e-6)]
        return dict(p=fitted,cost=float(np.mean(result.fun**2)),success=bool(result.success),
                    evaluations=result.nfev,active=active)


def analyse(out,a,pump=lambda:None):
    cfg=json.loads((out/'settings.json').read_text())
    if a.bias is None: a.bias=float(cfg.get('bias',0) or 0)
    if a.wavelength_nm is None: a.wavelength_nm=float(cfg.get('wavelength_nm',633) or 633)
    if a.wavelength_nm<=0: raise ValueError('Wavelength must be positive')
    with np.load(out/'cuts.npz',allow_pickle=False) as z:
        x=z['x']; cuts=z['cuts']; valid=z['valid'].astype(bool); amps=z['amplitude']
        direction=z['direction'] if 'direction' in z else np.full(len(amps),'unknown')
    if cuts.ndim!=2 or cuts.shape!=valid.shape or cuts.shape!=(len(amps),len(x)):
        raise ValueError('Inconsistent saved array shapes')
    if len(amps)>96 or len(x)>1024: raise ValueError('This bounded fitter accepts at most 96 exposures and 1024 cut samples')
    if len(amps)<8 or len(np.unique(amps))<5: raise ValueError('Need at least five distinct command amplitudes and eight exposures')
    if not np.isfinite(x).all() or not np.isfinite(amps).all() or len(x)<40 or np.any(np.diff(x)<=0):
        raise ValueError('Invalid coordinates/commands')
    results=out/('physical_fit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')); results.mkdir()
    problem=InterferenceFit(x,cuts,valid,amps,a,pump)
    fixed=(x[-1]-x[0])/a.camera_period if a.camera_period else None
    if fixed is not None and not .5<=fixed<=len(problem.idx)/4:
        raise ValueError('Known period outside resolved range; use a shorter cut or more --fit-samples')
    if fixed is None:
        zero=np.isclose(amps,0); high=amps>=.8*np.max(amps)
        signal=np.mean(problem.raw[high],axis=0)-np.mean(problem.raw[zero if zero.any() else np.ones(len(amps),bool)],axis=0)
        good=problem.mask & np.isfinite(signal)
        signal=np.interp(x,x[good],signal[good]); signal-=gaussian_filter1d(signal,max(2,len(x)/12))
        spectrum=np.abs(np.fft.rfft(signal*np.hanning(len(x)))); spectrum[:2]=0
        guess=float(np.argmax(spectrum))
        frequencies=sorted(set(np.clip([guess/2,guess,guess*2],1,min(16,len(problem.idx)/4)-.01)))
    else: frequencies=[fixed]
    runs=[]; profile=[]; branch_results=[]; stopped=None
    def record(run,kind):
        runs.append(run)
        print(f'{kind}: {run["p"][11]:.4g} nm/grey; RMS {np.sqrt(run["cost"])*problem.scale:.1f} ADU; converged={run["success"]}',flush=True)
        # Checkpoint numerical results so a later cancellation does not erase work.
        (results/'progress.json').write_text(json.dumps([dict(opd_nm_per_grey=float(r['p'][11]),
              rms_adu=float(np.sqrt(r['cost'])*problem.scale),converged=r['success']) for r in runs],indent=2))
    try:
        for k in frequencies:
            for fraction,phase in [(.2,.2),(.55,np.pi/2)]:
                run=problem.solve(problem.initial(k,a.opd_max*fraction,phase),fixed)
                record(run,'Initial fit')
        best=min(runs,key=lambda r:r['cost'])
        # Refit all nuisance parameters at fixed slopes. These are diagnostic
        # costs, not likelihood confidence intervals for correlated spatial samples.
        grid=np.unique(np.r_[np.linspace(0,a.opd_max,a.profile_points),best['p'][11]])
        for slope in grid:
            run=problem.solve(best['p'],fixed,float(slope))
            profile.append(dict(slope=float(slope),cost=run['cost'],success=run['success']))
            record(run,'Fixed-slope check')
        best=min(runs,key=lambda r:r['cost'])
        record(problem.solve(best['p'],fixed),'Refined fit')
        # Separate-direction sensitivity checks, not independent uncertainty
        # measurements: seed from the joint solution and refit each branch.
        best=min(runs,key=lambda r:r['cost'])
        for branch in ('up','down'):
            chosen=direction==branch
            if chosen.sum()<8 or len(np.unique(amps[chosen]))<5: continue
            problem.check()
            sub=InterferenceFit(x,cuts[chosen],valid[chosen],amps[chosen],a,pump)
            sub.deadline=problem.deadline
            seed=best['p'][:12].copy()
            seed[0]+=np.log(problem.scale/sub.scale)
            delta=np.r_[0.,best['p'][12:]][chosen]
            seed[6]+=delta[0]
            seed=np.r_[seed,delta[1:]-delta[0]]
            fit=sub.solve(seed,fixed)
            branch_results.append(dict(direction=branch,slope=float(fit['p'][11]),
                converged=fit['success'],active=fit['active'],rms_adu=float(np.sqrt(fit['cost'])*sub.scale)))
            print(f'{branch} sensitivity: {fit["p"][11]:.4g} nm/grey; converged={fit["success"]}',flush=True)
    except (Cancelled,FitLimit) as exc:
        stopped=str(exc) or 'User stopped analysis'
    if not runs:
        report={'status':'no_completed_fit','reason':stopped or 'No completed optimizer run'}
        (results/'summary.json').write_text(json.dumps(report,indent=2))
        return report,results
    best=min(runs,key=lambda r:r['cost']); p=best['p']
    predicted,C,B=problem.compute(p)
    filtered_prediction=smooth_blocks(predicted,problem.mask,problem.sigma)
    residual=(problem.filtered-filtered_prediction)*problem.scale
    reasons=[]
    if stopped: reasons.append(stopped)
    if not cfg.get('complete',False): reasons.append('Acquisition incomplete')
    if fixed is None: reasons.append('Command-to-camera spatial mapping not independently supplied')
    if not best['success']: reasons.append('Best optimizer did not converge')
    if any(not r['success'] for r in runs): reasons.append('Some optimizer runs did not converge; comparisons remain provisional')
    if best['active']: reasons.append('Parameters reached bounds: '+str(best['active']))
    if len(profile)<a.profile_points: reasons.append('Slope profile incomplete')
    close=[r['p'][11] for r in runs if r['cost']<=1.05*best['cost']+1e-12]
    if max(close)-min(close)>max(.1,.2*p[11]): reasons.append('Broad or competing slopes within 5% of best cost')
    if p[11]<.01*a.opd_max or p[11]>.99*a.opd_max: reasons.append('Slope close to search boundary')
    zero_cost=next((r['cost'] for r in profile if r['slope']==0),None)
    if zero_cost is None or zero_cost<=1.05*best['cost']: reasons.append('No clear cost improvement over zero SLM response')
    # Compare residual with the fitted command-induced signal at the same piston.
    pzero=p.copy(); pzero[11]=0
    without=problem.compute(pzero)[0]
    slm_signal=smooth_blocks(predicted-without,problem.mask,problem.sigma)[:,problem.idx]
    signal_rms=float(np.sqrt(np.mean(slm_signal**2))*problem.scale)
    rms=float(np.sqrt(best['cost'])*problem.scale)
    if signal_rms<=0 or rms>.5*signal_rms: reasons.append('Residual exceeds half the modelled SLM signal')
    gaps=np.diff(x[problem.idx]); local_phase_gradient=np.max(np.abs(p[7]+2*p[8]*problem.u))
    local_phase_gradient+=2*np.pi*p[9]*problem.factor*p[11]*np.max(np.abs(amps))
    if np.max(gaps)/(x[-1]-x[0])*local_phase_gradient>np.pi/2:
        reasons.append('Sparse fit samples may not resolve the maximum model phase gradient')
    if not np.any(np.isclose(amps,0)): reasons.append('No zero-command reference exposures')
    if len(branch_results)==2:
        if any(not r['converged'] or r['active'] for r in branch_results):
            reasons.append('Separate up/down sensitivity fits failed convergence or reached bounds')
        if abs(branch_results[0]['slope']-branch_results[1]['slope'])>max(.1,.2*p[11]):
            reasons.append('Separate up/down slopes differ by more than 20% (or 0.1 nm/grey)')
    report=dict(status='not_validated' if reasons else 'conditional_fit_passed_checks',
        opd_nm_per_grey=float(p[11]),phase_rad_per_grey=float(problem.factor*p[11]),
        equivalent_mirror_stroke_nm_per_grey=float(p[11]/2),period_camera_pixels=float((x[-1]-x[0])/p[9]),
        residual_rms_adu=rms,model_slm_rms_adu=signal_rms,flags=reasons,
        interpretation='Conditional positive phase-gain magnitude; not an independently validated absolute calibration.',
        uncertainty='No confidence interval: smoothed samples are correlated; profile costs are diagnostic only.',
        wavelength_nm=a.wavelength_nm,bias_adu=a.bias,smooth_sigma_pixels=a.smooth_px,
        fit_sample_count=len(problem.idx),common_valid_fraction=float(problem.mask.mean()),
        parameters=dict(zip(PARAMS+[f'piston_{i}' for i in range(1,len(amps))],p.tolist())),
        profile=profile,direction_sensitivity=branch_results,
        candidates=[dict(slope=float(r['p'][11]),cost=r['cost'],converged=r['success']) for r in runs])
    (results/'summary.json').write_text(json.dumps(report,indent=2))
    np.savez(results/'fit_arrays.npz',x=x,data=cuts,prediction=predicted*problem.scale+a.bias,
             illumination=C*problem.scale,contrast=B*problem.scale,mask=problem.mask,fit_indices=problem.idx,
             smoothed_data=problem.filtered*problem.scale+a.bias,smoothed_prediction=filtered_prediction*problem.scale+a.bias,
             parameters=p,amplitude=amps,direction=direction,residual=residual)
    # Original sampled-pixel residuals remain available alongside smoothed fits.
    raw_resid=(problem.raw-predicted)*problem.scale
    rms_frame=np.sqrt(np.mean(residual[:,problem.idx]**2,axis=1))
    raw_rms=np.sqrt(np.mean(raw_resid[:,problem.idx]**2,axis=1))
    np.savetxt(results/'frame_diagnostics.csv',np.c_[np.arange(len(amps)),amps,direction=='down',rms_frame,raw_rms,np.r_[0.,p[12:]]],
               delimiter=',',comments='',header='frame,command,is_down,smoothed_rms_adu,raw_rms_adu,piston_rad')
    plot_results(results,x,cuts,amps,direction,problem,p,predicted,C,B,filtered_prediction,report)
    # Plot-only additions must not prevent a completed physical-fit report from being returned.
    try:
        plot_command_response(results,amps,direction,problem,p,C,B,filtered_prediction,report)
    except Exception as exc:
        print(f"Command-response diagnostic plot failed: {exc}",file=sys.stderr,flush=True)
    print(json.dumps({k:report[k] for k in ('status','opd_nm_per_grey','residual_rms_adu','flags')},indent=2))
    print('Saved analysis:',results.resolve(),flush=True)
    return report,results


def plot_results(results,x,cuts,amps,direction,problem,p,predicted,C,B,filtered_prediction,report):
    # Noninteractive figure objects: offline mode needs no Qt/display.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    def figure(*args,**kwargs):
        f=Figure(*args,**kwargs); FigureCanvasAgg(f); return f
    examples=np.unique(np.linspace(0,len(amps)-1,6).astype(int))
    fig=figure(figsize=(12,2.5*len(examples)),layout='constrained'); axes=fig.subplots(len(examples),2,squeeze=False)
    valid=problem.eligible; bias=problem.a.bias; scale=problem.scale
    for (left,right),i in zip(axes,examples):
        left.plot(x,cuts[i],'.',ms=2,alpha=.35,label='raw')
        left.plot(x,np.where(valid,problem.filtered[i]*scale+bias,np.nan),label='smoothed data')
        left.plot(x,np.where(valid,filtered_prediction[i]*scale+bias,np.nan),label='filtered physical model')
        left.plot(x,C*scale+bias,'--',label='illumination + bias')
        left.set(title=f'{direction[i]} command {amps[i]:g}',ylabel='ADU')
        right.plot(x,np.where(valid,(problem.raw[i]-predicted[i])*scale,np.nan),alpha=.3,label='raw residual')
        right.plot(x,np.where(valid,(problem.filtered[i]-filtered_prediction[i])*scale,np.nan),label='smoothed residual')
        right.axhline(0,color='grey',lw=.7); right.set(ylabel='Residual ADU')
    for ax in axes[0]: ax.legend(fontsize=7)
    for ax in axes[-1]: ax.set_xlabel('Camera pixels along cut')
    fig.suptitle(f'{report["status"]}: candidate {p[11]:.3g} nm OPD/grey; see summary.json')
    fig.savefig(results/'crosscuts.png',dpi=140)
    fig=figure(figsize=(12,8),layout='constrained'); axes=fig.subplots(2,2)
    if report['profile']:
        grid=sorted(report['profile'],key=lambda r:r['slope'])
        axes[0,0].plot([r['slope'] for r in grid],[r['cost'] for r in grid],'o-')
        bad=[r for r in grid if not r['success']]
        axes[0,0].plot([r['slope'] for r in bad],[r['cost'] for r in bad],'rx',label='unconverged')
    axes[0,0].axvline(p[11],color='red'); axes[0,0].set(xlabel='Fixed OPD slope (nm/grey)',ylabel='Mean squared normalised residual',title='Nuisance-refitted costs; not a confidence interval')
    axes[0,1].plot(np.r_[0.,p[12:]],'.-'); axes[0,1].set(xlabel='Stored exposure order',ylabel='Frame piston (rad)')
    residual=problem.filtered-filtered_prediction
    for branch in np.unique(direction):
        m=direction==branch
        axes[1,0].plot(amps[m],np.sqrt(np.mean(residual[m][:,problem.idx]**2,axis=1))*scale,'.',label=branch)
    axes[1,0].set(xlabel='Command amplitude',ylabel='Residual RMS (ADU)'); axes[1,0].legend()
    axes[1,1].plot(x,C*scale,label='illumination'); axes[1,1].plot(x,B*scale,label='contrast')
    axes[1,1].set(xlabel='Camera pixels',ylabel='ADU'); axes[1,1].legend()
    fig.savefig(results/'diagnostics.png',dpi=140)



def plot_command_response(results,amps,direction,problem,p,C,B,filtered_prediction,report):
    """Additional diagnostics only; does not modify or rerun the global fit.

    The OPD markers are *conditional* per-command local refits. All spatial
    envelope, static-phase and exposure-piston terms are HELD FIXED at the
    global solution. They are not independent phase/OPD measurements or CIs.
    """
    import csv
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig=Figure(figsize=(13,9),layout='constrained')
    FigureCanvasAgg(fig)
    ax_signal,ax_opd=fig.subplots(2,1)

    idx=problem.idx
    scale=problem.scale
    # Per-exposure mean over the SAME accepted fit samples, after the SAME
    # smoothing used by the global fit. Bias is removed for plotting only.
    measured=np.mean(problem.filtered[:,idx],axis=1)*scale
    model=np.mean(filtered_prediction[:,idx],axis=1)*scale

    u=problem.u; t=problem.t
    phase_static=p[6]+p[7]*u+p[8]*u*u
    command_mode=np.sin(2*np.pi*p[9]*t+p[10])
    pistons=np.r_[0.,p[12:]]
    all_rows=[]
    colors={'up':'tab:blue','down':'tab:orange'}
    symbols={'up':'o','down':'s'}

    for branch in np.unique(direction):
        mask=direction==branch
        bamps=np.unique(amps[mask])
        raw_means=[]; raw_stds=[]; model_means=[]
        opd_amps=[]; opd_values=[]
        for amp in bamps:
            sel=np.flatnonzero(mask & np.isclose(amps,amp,rtol=1e-9,atol=1e-10))
            observed=measured[sel]
            raw_means.append(float(np.mean(observed)))
            raw_stds.append(float(np.std(observed,ddof=1)) if len(sel)>1 else 0.)
            model_means.append(float(np.mean(model[sel])))

            opd=np.nan; bound=False; success=False
            # At zero command OPD=0 is imposed by the model, rather than
            # inferred from intensity. Avoid plotting it as an OPD measurement.
            if not np.isclose(amp,0.):
                # Only the command-specific OPD is free. Piston per frame and
                # all other model parameters remain at the GLOBAL fit values.
                phi0=phase_static[None,:]+pistons[sel,None]
                target=problem.filtered[sel][:,idx]
                def calc(delta,with_jac=False):
                    phi=phi0+problem.factor*delta*command_mode[None,:]
                    values=C[None,:]+B[None,:]*np.cos(phi)
                    if with_jac:
                        values=-B[None,:]*np.sin(phi)*problem.factor*command_mode[None,:]
                    return smooth_blocks(values,problem.mask,problem.sigma)[:,idx]
                def residual(z):
                    return (calc(z[0])-target).ravel()
                def jac(z):
                    return calc(z[0],True).reshape(-1,1)
                edge=problem.a.opd_max*amp
                lo,hi=sorted((0.,edge))
                seed=float(np.clip(p[11]*amp,lo+1e-9,hi-1e-9))
                try:
                    fit=least_squares(residual,[seed],jac=jac,bounds=([lo],[hi]),
                                      max_nfev=50,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                    success=bool(fit.success)
                    if success:
                        opd=float(fit.x[0])
                        bound=bool(np.isclose(opd,lo,atol=max(1e-5,1e-3*(hi-lo))) or
                                   np.isclose(opd,hi,atol=max(1e-5,1e-3*(hi-lo))))
                        opd_amps.append(float(amp))
                        opd_values.append(opd)
                except (ValueError, FloatingPointError):
                    pass   # A failed diagnostic must not affect the global fit.
            all_rows.append(dict(direction=str(branch),command_grey=float(amp),
                n_exposures=len(sel),signal_mean_adu=raw_means[-1],
                signal_frame_std_adu=raw_stds[-1],model_mean_adu=model_means[-1],
                conditional_opd_nm=opd,global_fit_opd_nm=float(p[11]*amp),
                conditional_refit_success=success,conditional_refit_at_bound=bound))

        col=colors.get(str(branch),None)
        marker=symbols.get(str(branch),'D')
        ax_signal.errorbar(bamps,raw_means,yerr=raw_stds,fmt=marker+'-',
                           capsize=3,color=col,label=f'{branch}: mean ± frame SD')
        ax_signal.plot(bamps,model_means,'--',color=col,alpha=.85,
                       label=f'{branch}: fitted model')
        if opd_amps:
            ax_opd.plot(opd_amps,opd_values,marker,ms=6,linestyle='none',color=col,
                        label=f'{branch}: conditional OPD refits')

    cmd_grid=np.linspace(min(0.,float(np.min(amps))),max(0.,float(np.max(amps))),300)
    ax_opd.plot(cmd_grid,p[11]*cmd_grid,'k-',lw=2,
                label=f'Global model: {p[11]:.4g} nm OPD/grey × command')
    ax_opd.axhline(0,color='0.6',lw=.7)
    ax_signal.set(title='Aggregated signal per command (accepted cut samples)',
                  xlabel='SLM command amplitude (grey levels)',
                  ylabel='Mean intensity above camera bias (ADU)')
    ax_opd.set(title='Command-induced OPD amplitude (model-dependent)',
               xlabel='SLM command amplitude (grey levels)',ylabel='OPD amplitude (nm)')
    for ax in (ax_signal,ax_opd):
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle('SLM command-response diagnostics — '+report['status']+'\n'
                 'OPD markers: conditional local refits (fixed background/pistons); NOT independent measurements or CIs',
                 fontsize=11)
    fig.savefig(results/'command_response.png',dpi=160)

    with (results/'command_response.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

def live_gui(cam,slm,app,a):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from matplotlib.patches import Polygon
    fig,(ax,profile)=plt.subplots(1,2,figsize=(14,7),gridspec_kw={'width_ratios':[1.5,1]})
    fig.subplots_adjust(bottom=.22)
    state=dict(mode='live',points=[],press=None,drag=False,count=-1,stop=False,closing=False,out=None)
    shape=(slm.height,slm.width); yy,xx=np.indices(shape,dtype=float)
    pattern=np.sin(2*np.pi*(a.fx*xx/shape[1]+a.fy*yy/shape[0])); del yy,xx
    flat=np.full(shape,a.baseline,np.uint8)
    def command(amp):
        arr=np.rint(a.baseline+amp*pattern)
        if arr.min()<0 or arr.max()>255: raise ValueError('Command clipping refused')
        return arr.astype(np.uint8)
    def pump():
        app.processEvents()
        if state['stop'] or state['closing']: raise Cancelled('Stopped by user')
    _,raw,_=next_frame(cam,int(cam.md.cnt0),a.timeout,pump)
    camera_shape=raw.shape
    if raw.ndim!=2: raise ValueError('Camera stream must be a 2D image')
    cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('magenta')
    shown=np.ma.masked_where(~np.isfinite(raw)|(raw>=a.saturation),raw)
    good=shown.compressed()
    if not good.size: raise ValueError('No valid unsaturated camera pixels')
    lo,hi=np.percentile(good,[2,99])
    image=ax.imshow(shown,cmap=cmap,vmin=lo,vmax=max(lo+1,hi))
    fig.colorbar(image,ax=ax,label='Raw ADU',fraction=.04)
    line,=ax.plot([],[],'r.-',markersize=10)
    band=Polygon(np.zeros((4,2)),color='red',alpha=.2,visible=False); ax.add_patch(band)
    cutline,=profile.plot([],[],alpha=.35,label='raw')
    smoothedline,=profile.plot([],[],label='smoothed')
    profile.legend(); profile.set(xlabel='Camera pixels along cut',ylabel='ADU')
    ax.set_title('Drag across fringes, or click two endpoints')
    status=fig.text(.025,.12,'Select a clean strip across several fringes; magenta = invalid/saturated.',fontsize=11)
    slm.show(command(a.preview_amplitude)); app.processEvents()
    def draw():
        points=state['points']; line.set_data(*np.asarray(points).T) if points else line.set_data([],[])
        band.set_visible(len(points)==2)
        if len(points)==2:
            p0,p1=np.asarray(points); v=p1-p0; length=np.linalg.norm(v)
            if length:
                normal=np.array([-v[1],v[0]])/length*a.width/2
                band.set_xy([p0-normal,p1-normal,p1+normal,p0+normal])
        fig.canvas.draw_idle()
    def refresh():
        if state['mode']!='live': return
        count=int(cam.md.cnt0)
        if cam.md.write or count==state['count']: return
        frame=np.array(cam,copy=True)
        if cam.md.write or int(cam.md.cnt0)!=count: return
        if frame.shape!=camera_shape:
            status.set_text('Camera shape changed: close and restart.'); return
        state['count']=count
        image.set_data(np.ma.masked_where(~np.isfinite(frame)|(frame>=a.saturation),frame))
        if len(state['points'])==2:
            try:
                x,xy=geometry(state['points'],a.width,camera_shape)
                cut,valid=extract(frame,xy,a.saturation)
                cutline.set_data(x,cut)
                smoothedline.set_data(x,smooth_blocks(cut,valid,a.smooth_px/np.median(np.diff(x))))
                profile.relim(); profile.autoscale_view()
                if time.monotonic()>state.get('sweep_notice_until',0):
                    status.set_text(f'Strip width {a.width} px; {(~valid).mean():.1%} invalid positions. Start sweep when ready.')
            except ValueError as exc: status.set_text(str(exc))
        fig.canvas.draw_idle()
    def press(event):
        toolbar=getattr(fig.canvas,'toolbar',None)
        if state['mode']!='live' or event.inaxes is not ax or event.button!=1 or (toolbar and toolbar.mode): return
        state['press']=(event.xdata,event.ydata); state['drag']=False
    def motion(event):
        if state['press'] is None or event.inaxes is not ax: return
        end=(event.xdata,event.ydata)
        if np.linalg.norm(np.asarray(end)-state['press'])>5:
            state['drag']=True; state['points']=[state['press'],end]; draw()
    def release(event):
        if state['press'] is None: return
        if not state['drag'] and event.inaxes is ax:
            if len(state['points'])==2: state['points']=[]
            state['points'].append((event.xdata,event.ydata))
        state['press']=None; draw()
    def reset(event):
        if state['mode']!='live': return
        state['points']=[]; state['press']=None; draw()
        cutline.set_data([],[]); smoothedline.set_data([],[])
    def stop(event):
        if state['mode'] in ('sweep','analysis'):
            state['stop']=True; status.set_text('Stopping at the next safe checkpoint...'); fig.canvas.draw_idle()
    def close(event):
        state['closing']=True; state['stop']=True
    def start(event):
        # Start-sweep diagnostics: keep acquisition order and saved data unchanged.
        # In particular, make validation failures and camera waits visible immediately.
        print(f'[SLM sweep] Start sweep pressed; state={state["mode"]}, selected_points={len(state["points"])}',
              flush=True)
        if state['mode']!='live':
            msg=(f'Start sweep unavailable in state {state["mode"]!r}. '
                 'After a failed/finished sweep, restart with a NEW --output directory.')
            print('[SLM sweep] '+msg,flush=True)
            status.set_text(msg); fig.canvas.draw_idle()
            return
        try:
            if len(state['points'])!=2: raise ValueError('Select two endpoints or drag a cross-cut first')
            distance,xy=geometry(state['points'],a.width,camera_shape)
            out=Path(a.output)
            if out.exists(): raise FileExistsError(
                f'Output directory already exists: {out}. Restart with a new --output name; existing data are not overwritten.')
            out.mkdir(parents=True)
        except (ValueError,FileExistsError,OSError) as exc:
            msg=f'Cannot start sweep: {exc}'
            print('[SLM sweep] '+msg,file=sys.stderr,flush=True)
            status.set_text(msg)
            # The live preview otherwise immediately overwrites this warning.
            state['sweep_notice_until']=time.monotonic()+10.
            fig.canvas.draw_idle()
            return
        state.update(mode='sweep',stop=False,out=out)
        config=vars(a).copy(); config['analyse']=None
        config.update(points=state['points'],camera_shape=camera_shape,slm_shape=shape,complete=False,
                      version=2,command_formula='round(baseline + amplitude*sin(2*pi*(fx*x/SLMW+fy*y/SLMH)))',
                      timestamp_source='host-copy UTC, not exposure timestamp')
        cuts=[]; masks=[]; amps=[]; counters=[]; times=[]; directions=[]
        def show_progress(message):
            print('[SLM sweep] '+message,flush=True)
            status.set_text(message)
            fig.canvas.draw_idle()
            app.processEvents()
        def checkpoint():
            temp=out/'cuts.pending.npz'
            np.savez(temp,x=distance,cuts=cuts,valid=masks,amplitude=amps,counters=counters,host_utc=times,direction=directions)
            temp.replace(out/'cuts.npz')
        try:
            levels=np.unique(np.r_[np.arange(0,a.max_amplitude,a.step),a.max_amplitude])
            nsteps=2*len(levels)
            show_progress(f'Beginning {nsteps} SLM commands ({a.nframes} camera frames each); output={out}')
            (out/'settings.json').write_text(json.dumps(config,indent=2))
            fits.PrimaryHDU(pattern.astype(np.float32)).writeto(out/'slm_pattern.fits',overwrite=False)
            step_no=0
            for branch,values in [('up',levels),('down',levels[::-1])]:
                for amplitude in values:
                    step_no+=1
                    show_progress(f'[{step_no}/{nsteps}] {branch} amplitude={amplitude:g}: sending SLM pattern, settling...')
                    pump(); slm.show(command(amplitude)); app.processEvents()
                    deadline=time.monotonic()+a.settle
                    while time.monotonic()<deadline: pump(); time.sleep(.01)
                    count=int(cam.md.cnt0)
                    show_progress(f'[{step_no}/{nsteps}] amplitude={amplitude:g}: waiting for first fresh camera frame...')
                    count,_,_=next_frame(cam,count,a.timeout,pump)
                    for frame_no in range(1,a.nframes+1):
                        count,frame,stamp=next_frame(cam,count,a.timeout,pump)
                        if frame.shape!=camera_shape: raise RuntimeError('Camera shape changed')
                        cut,valid=extract(frame,xy,a.saturation)
                        cuts.append(cut); masks.append(valid); amps.append(amplitude)
                        counters.append(count); times.append(stamp); directions.append(branch)
                        print(f'[SLM sweep] [{step_no}/{nsteps}] frame {frame_no}/{a.nframes} '
                              f'counter={count}; invalid cut samples={(~valid).sum()}',flush=True)
                    # Preview the last captured frame without changing frame selection.
                    image.set_data(np.ma.masked_where(~np.isfinite(frame)|(frame>=a.saturation),frame))
                    show_progress(f'[{step_no}/{nsteps}] amplitude={amplitude:g}: saving checkpoint...')
                    checkpoint()
                    show_progress(f'[{step_no}/{nsteps}] {branch}: amplitude={amplitude:g}, exposures={len(cuts)} saved')
            config['complete']=True
            show_progress(f'Sweep completed: {len(cuts)} exposures saved under {out}. Press Analyse.')
        except (Cancelled,KeyboardInterrupt):
            show_progress('Stopped. Captured exposures saved; incomplete acquisition flagged.')
        except Exception as exc:
            msg=f'Acquisition error: {type(exc).__name__}: {exc}'
            print('[SLM sweep] '+msg,file=sys.stderr,flush=True)
            if isinstance(exc,TimeoutError):
                print('[SLM sweep] No fresh camera frame arrived before --timeout. '
                      'Check the camera stream/counter (dcam0_data) is running.',file=sys.stderr,flush=True)
            status.set_text(msg); fig.canvas.draw_idle()
        finally:
            try:
                if cuts: checkpoint()
                (out/'settings.json').write_text(json.dumps(config,indent=2))
            finally:
                try: slm.show(flat); app.processEvents()
                except Exception as exc: print(f'Could not restore flat: {exc}',file=sys.stderr,flush=True)
                state['mode']='ready' if cuts else 'failed'; state['stop']=False
            if config['complete']: status.set_text(f'Acquisition saved. Press Analyse. SLM returned to baseline.')
            fig.canvas.draw_idle()
    def do_analysis(event):
        if state['mode'] not in ('ready','analysis_done'): return
        state.update(mode='analysis',stop=False)
        status.set_text('Fitting physical model; Stop is available. Progress is printed in terminal.')
        fig.canvas.draw_idle()
        try:
            report,results=analyse(state['out'],a,pump)
            if not state['closing']:
                profile.clear()
                if 'profile' in report:
                    rows=sorted(report['profile'],key=lambda r:r['slope'])
                    profile.plot([r['slope'] for r in rows],[r['cost'] for r in rows],'o-')
                    profile.axvline(report['opd_nm_per_grey'],color='red')
                    profile.set(xlabel='Fixed OPD slope (nm/grey)',ylabel='Refitted residual cost',
                                title=f'{report["status"]}\nCandidate {report["opd_nm_per_grey"]:.3g} nm/grey')
                status.set_text(f'{report["status"]}. See {results.name}/summary.json and crosscuts.png')
        except (Cancelled,KeyboardInterrupt): status.set_text('Analysis stopped. Acquisition is unchanged.')
        except Exception as exc:
            status.set_text(f'Analysis error: {exc}'); print(f'Analysis error: {exc}',file=sys.stderr)
        finally:
            state.update(mode='analysis_done',stop=False); fig.canvas.draw_idle()
    buttons=[]
    for left,label,callback in [(.025,'Reset cut',reset),(.22,'Start sweep',start),(.415,'Analyse',do_analysis),
                                (.61,'Stop',stop),(.805,'Quit',lambda e:plt.close(fig))]:
        b=Button(fig.add_axes([left,.025,.17,.065]),label); b.label.set_fontsize(13)
        b.on_clicked(callback); buttons.append(b)
    for event,callback in [('button_press_event',press),('motion_notify_event',motion),
                           ('button_release_event',release),('close_event',close)]:
        fig.canvas.mpl_connect(event,callback)
    timer=fig.canvas.new_timer(interval=250); timer.add_callback(refresh); timer.start()
    try: plt.show(block=True)
    finally: timer.stop()


def main():
    a=arguments()
    if a.analyse:
        analyse(a.analyse,a)
        return
    if a.bias is None: a.bias=0.
    if a.wavelength_nm is None: a.wavelength_nm=633.
    import matplotlib
    matplotlib.use('Qt5Agg')
    from PyQt5 import QtWidgets
    import ImageStreamIOWrap
    from slmdisplay import SLM
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    cam=None; slm=None; entered=False; flat=None
    try:
        cam=ImageStreamIOWrap.Image(); cam.open(a.shm)
        screen=int(a.screen) if a.screen.lstrip('-').isdigit() else a.screen
        slm=SLM(screen=screen); slm.__enter__(); entered=True
        flat=np.full((slm.height,slm.width),a.baseline,np.uint8)
        live_gui(cam,slm,app,a)
    finally:
        try:
            if entered:
                try:
                    if flat is not None: slm.show(flat); app.processEvents()
                finally:
                    try: slm.__exit__(None,None,None)
                    except FileNotFoundError as exc:
                        if not (exc.filename and str(exc.filename).startswith('/psm_')): raise
                        print('Display shared memory already removed during shutdown; saved data retained.',file=sys.stderr)
        finally:
            if cam is not None and hasattr(cam,'close'): cam.close()


if __name__=='__main__':
    try: main()
    except (Cancelled,KeyboardInterrupt): print('Stopped. Existing saved acquisition files were not changed by analysis.')
    except (ValueError,RuntimeError,TimeoutError,FileExistsError) as exc: raise SystemExit(str(exc))
