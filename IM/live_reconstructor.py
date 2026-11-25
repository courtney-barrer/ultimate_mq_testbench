#!/usr/bin/env python3
"""Bounded-memory live worker for calibration_gui.py; stdout is newline JSON.
Uses adjacent build_IM.py for the exact camera/centroid interface. No closed loop.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse, base64, hashlib, importlib.util, json, signal, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from astropy.io import fits


def emit(**data):print(json.dumps(data,allow_nan=False,separators=(',',':')),flush=True)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def thumbnail(z,side=192,symmetric=True,limit=None):
    step=max(1,int(np.ceil(max(z.shape)/side)));z=np.asarray(z[::step,::step],float)
    if symmetric:
        hi=limit or max(float(np.nanmax(abs(z))),1e-6);lo=-hi
    else:lo,hi=np.nanpercentile(z,[1,99.8]);hi=max(hi,lo+1)
    pixels=np.nan_to_num((z-lo)*255/(hi-lo),nan=127).clip(0,255).astype('uint8')
    return dict(shape=list(pixels.shape),data=base64.b64encode(pixels.tobytes()).decode(),lo=float(lo),hi=float(hi))


class Controls:
    def __init__(self):
        self.lock=threading.Lock();self.coefficients=None;self.reference=False;self.preview=False;self.stop=threading.Event()
    def read(self):
        try:
            for line in sys.stdin:
                try:
                    msg=json.loads(line)
                    with self.lock:
                        if msg.get('action')=='stop':self.stop.set()
                        elif msg.get('action')=='reference':self.reference=True;self.coefficients=None
                        elif msg.get('action')=='set':self.coefficients=msg['coefficients']
                        elif msg.get('action')=='preview':self.preview=bool(msg['enabled'])
                except Exception as e:emit(event='warning',message=f'Invalid control message: {e}')
        finally:self.stop.set()
    def get(self):
        with self.lock:
            c,ref,preview=self.coefficients,self.reference,self.preview
            self.coefficients=None;self.reference=False
            return c,ref,preview


class Model:
    def __init__(self,a):
        self.files=[]
        try:
            self.rfile=fits.open(a.reconstructor,memmap=True);self.files.append(self.rfile)
            self.bfile=fits.open(a.basis,memmap=False);self.files.append(self.bfile)
            self.imfile=fits.open(a.im,memmap=True);self.files.append(self.imfile)
            rh=self.rfile[0].header;bh=self.bfile[0].header;ih=self.imfile[0].header
            if rh.get('SCHEMA')!='SLMRECON' or not rh.get('COMPLETE') or bh.get('SCHEMA')!='SLMBASIS' or not bh.get('COMPLETE') or not ih.get('COMPLETE'):raise ValueError('Need complete basis, IM and reconstructor FITS')
            if not rh['REGVALID'] and not a.allow_unvalidated:raise ValueError('Registration unvalidated; explicitly enable exploratory mode')
            if digest(a.basis)!=rh['BASISHA'] or digest(a.im)!=rh['IMSHA']:raise ValueError('Basis/IM hash does not match reconstructor')
            self.rh=rh;self.bh=bh
            self.r=np.array(self.rfile[0].data,float);self.ids=np.array(self.rfile['IM_MODES'].data['MODE_ID'],int)
            ci=np.flatnonzero(np.isclose(self.rfile['AMPLITUDES'].data['AMPLITUDE'],rh['CALAMP']))
            if len(ci)!=1:raise ValueError('Ambiguous calibration amplitude')
            self.j=np.array(self.rfile['IM'].data[int(ci[0])],float)
            self.centres=np.column_stack([self.rfile['SPOTS'].data[k] for k in ('X','Y')]).astype(int)
            rows=self.rfile['SLOPE_ROWS'].data;self.spots=np.array(rows['SPOT_ID'],int)
            self.axes=np.array([0 if str(v).strip()=='x' else 1 for v in rows['AXIS']]);self.used=np.unique(self.spots)
            self.dark=np.array(self.imfile['IMAGE00000'].data,float)
            self.darkvar=np.array(self.imfile['VAR00000'].data,float)
            self.darksat=np.array(self.imfile['SAT00000'].data,bool)
            self.mask=np.array(self.bfile['PUPIL'].data,bool)
            # Cache preview samples only: <=32 MiB even for large mode counts.
            side=min(192,max(16,int(np.sqrt(8_000_000/max(1,len(self.ids))))))
            self.step=max(1,int(np.ceil(max(self.mask.shape)/side)))
            self.smallmask=self.mask[::self.step,::self.step]
            self.small=np.array([self.bfile[0].section[i,::self.step,::self.step] for i in self.ids],dtype='float32')
            self.ncontrol=min(6,len(self.ids));self.coeff=np.zeros(len(self.ids))
            if not len(self.ids) or self.r.shape!=(len(self.ids),len(rows)):raise ValueError('Invalid reconstructor dimensions')
            # Native modes for six controls only; never load full basis cube.
            self.native=np.array([self.bfile[0].section[i,:,:] for i in self.ids[:self.ncontrol]],dtype='float32')
            peak=np.max(abs(self.native),axis=(1,2));headroom=min(bh['BASELINE'],255-bh['BASELINE'])
            self.limit=min(5.,float(np.min(headroom/np.maximum(peak,1e-12))))
            self.preview_limit=max(1.,float(rh['CALAMP']))
            self.options=SimpleNamespace(screen=a.screen,nx=bh['SLMNX'],ny=bh['SLMNY'],baseline=bh['BASELINE'],settle=.3,flush=3)
            self.roi=[rh[k] for k in ('CAMX','CAMY','CAMW','CAMH')]
            self.saturation=ih['SATADU'];self.xyref=None
        except BaseException:self.close();raise
    def close(self):
        for f in self.files:f.close()
    def pattern(self,c):
        c=np.asarray(c,float)
        if c.shape!=(self.ncontrol,) or not np.isfinite(c).all():raise ValueError('Invalid slider coefficients')
        z=np.tensordot(c,self.native,axes=(0,0));z[~self.mask]=0
        crop=self.bh['BASELINE']+np.rint(z)
        if crop.min()<0 or crop.max()>255:raise ValueError('Combined sliders exceed 8-bit headroom; reduce amplitudes')
        out=np.full((self.bh['SLMNY'],self.bh['SLMNX']),self.bh['BASELINE'],np.uint8)
        out[self.bh['Y0']:self.bh['Y1'],self.bh['X0']:self.bh['X1']]=crop.astype('uint8')
        return out
    def crop(self,image):
        x,y=self.rh['CROPX'],self.rh['CROPY'];h,w=self.dark.shape
        z=image[y:y+h,x:x+w]
        if z.shape!=self.dark.shape:raise ValueError('Current image cannot contain calibrated crop')
        return z


def average(bench,model,n,controls):
    mean=m2=sat=None
    for i in range(n):
        if controls.stop.is_set():raise InterruptedError('Stopping')
        im,_,_=bench.cam.frame();im=model.crop(im).astype(float)
        if mean is None:mean=np.zeros_like(im);m2=np.zeros_like(im);sat=np.zeros(im.shape,bool)
        delta=im-mean;mean+=delta/(i+1);m2+=delta*(im-mean);sat|=im>=model.saturation
    return mean,m2/(n-1)/n,sat


def live(a,controls):
    mod=Model(a);bench=None
    try:
        spec=importlib.util.spec_from_file_location('bench_interface',Path(a.scripts)/'build_IM.py')
        api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
        bench=api.Bench(mod.options)
        if bench.cam.original['roi']!=mod.roi or abs(bench.cam.exposure-mod.rh['EXPTIME'])>1.1e-5:raise ValueError('Camera ROI/exposure differs: restore camera first')
        emit(event='ready',ids=mod.ids.tolist(),ncontrol=mod.ncontrol,limit=mod.limit,message='Capturing fresh flat reference. Laser must be ON; saved matched dark is used.')
        needref=True;lastcheck=0
        while not controls.stop.is_set():
            started=time.monotonic();requested,newref,preview=controls.get()
            if newref:needref=True
            if needref:
                bench.show(bench.flat);mod.coeff[:]=0
                mean,var,sat=average(bench,mod,16,controls)
                xy,_,good,_,_,_=api.measure(mean-mod.dark,var+mod.darkvar,sat|mod.darksat,mod.centres,mod.rh['HALF'])
                if not good[mod.used].all():raise ValueError(f'{(~good[mod.used]).sum()} required lenslets invalid in fresh flat reference')
                mod.xyref=xy;needref=False;requested=None
                emit(event='reference',message='Fresh flat reference captured; sliders reset.')
            if requested is not None:
                try:
                    pattern=mod.pattern(requested);bench.show(pattern)
                    mod.coeff[:mod.ncontrol]=requested
                    emit(event='applied',coefficients=mod.coeff[:mod.ncontrol].tolist())
                except ValueError as e:emit(event='rejected',message=str(e),coefficients=mod.coeff[:mod.ncontrol].tolist())
            if time.monotonic()-lastcheck>5:bench.cam.check();lastcheck=time.monotonic()
            mean,var,sat=average(bench,mod,a.frames,controls)
            xy,_,good,_,_,_=api.measure(mean-mod.dark,var+mod.darkvar,sat|mod.darksat,mod.centres,mod.rh['HALF'])
            invalid=int((~good[mod.used]).sum())
            if invalid:
                emit(event='invalid',message=f'{invalid} required lenslets invalid: reconstruction withheld; no rows removed')
            else:
                slopes=(xy-mod.xyref)[mod.spots,mod.axes];coeff=mod.r@slopes;residual=slopes-mod.j@coeff
                z=np.tensordot(coeff,mod.small,axes=(0,0));z[~mod.smallmask]=np.nan
                probe=np.tensordot(mod.coeff,mod.small,axes=(0,0));probe[~mod.smallmask]=np.nan
                limit=max(mod.preview_limit,float(np.nanmax(abs(z))),float(np.nanmax(abs(probe))))
                payload=dict(event='frame',coefficients=coeff.tolist(),reconstruction=thumbnail(z,limit=limit),
                    slope_rms=float(np.sqrt(np.mean(slopes**2))),residual_rms=float(np.sqrt(np.mean(residual**2))),
                    message='Live — open loop; ~1 Hz target',elapsed=time.monotonic()-started)
                if preview:
                    payload.update(camera=thumbnail(mean,symmetric=False),probe=thumbnail(probe,limit=limit))
                emit(**payload)
            controls.stop.wait(max(0,1/a.hz-(time.monotonic()-started)))
    finally:
        if bench is not None:bench.close()
        mod.close()


def simulate(a,controls):
    ids=list(range(6));c=np.zeros(6);y,x=np.mgrid[-1:1:128j,-1:1:128j];mask=x*x+y*y<1
    modes=np.array([x,y,x*y,x*x-y*y,2*(x*x+y*y)-1,np.sin(3*x)])
    rng=np.random.default_rng(4)
    emit(event='ready',ids=ids,ncontrol=6,limit=5.,message='SIMULATION: no camera or SLM access')
    while not controls.stop.is_set():
        requested,ref,preview=controls.get()
        if requested is not None:c=np.array(requested);emit(event='applied',coefficients=c.tolist())
        if ref:c[:]=0;emit(event='reference',message='Simulated reference')
        recovered=c+rng.normal(0,.03,6);z=np.tensordot(recovered,modes,axes=(0,0));z[~mask]=np.nan
        payload=dict(event='frame',coefficients=recovered.tolist(),reconstruction=thumbnail(z,limit=5),slope_rms=.02,residual_rms=.003,message='SIMULATION — no hardware',elapsed=.01)
        if preview:payload.update(camera=thumbnail(np.exp(-8*(x*x+y*y)),symmetric=False),probe=thumbnail(np.where(mask,np.tensordot(c,modes,axes=(0,0)),np.nan),limit=5))
        emit(**payload);controls.stop.wait(1/a.hz)


def main():
    p=argparse.ArgumentParser();p.add_argument('--basis',type=Path);p.add_argument('--im',type=Path);p.add_argument('--reconstructor',type=Path)
    p.add_argument('--scripts',type=Path,default=Path(__file__).parent);p.add_argument('--screen',default='1');p.add_argument('--hz',type=float,default=1.)
    p.add_argument('--frames',type=int,default=4);p.add_argument('--simulate',action='store_true');p.add_argument('--allow-unvalidated',action='store_true')
    a=p.parse_args()
    if not 0<a.hz<=2 or a.frames<2:p.error('Need 0<hz<=2 and frames>=2')
    if not a.simulate and any(v is None for v in (a.basis,a.im,a.reconstructor)):p.error('Need basis, IM and reconstructor')
    control=Controls();signal.signal(signal.SIGINT,lambda *_:control.stop.set());signal.signal(signal.SIGTERM,lambda *_:control.stop.set())
    threading.Thread(target=control.read,daemon=True).start()
    try:
        if a.simulate:simulate(a,control)
        else:live(a,control)
    except InterruptedError:pass
    except Exception as e:emit(event='error',message=str(e));return 1
    finally:emit(event='stopped',message='Worker finished; cleanup attempted')
    return 0

if __name__=='__main__':sys.exit(main())
