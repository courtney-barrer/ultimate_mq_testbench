#!/usr/bin/env python3
"""Build a SHWFS interaction matrix from SLMBASIS v1 FITS. No camera settings change.
Dependencies: numpy, scipy, matplotlib, astropy; bench: ImageStreamIOWrap, slmdisplay.
Outputs one FITS plus diagnostic PNGs. --demo uses synthetic camera data only.
"""
from pathlib import Path
import argparse, hashlib, re, socket, time
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOST, PORT, DEVICE, STREAM = '127.0.0.1', 18010, 'dcam0', 'dcam0_data'
SENSOR_WH = (2304,2304)
SCHEMA = 'slm-fourier-images-1'
SATURATION_ADU = 65000

def command(cmd, timeout=30.):
    deadline = time.monotonic() + timeout
    data = bytearray()
    with socket.create_connection((HOST, PORT), timeout=min(5.,timeout)) as sock:
        sock.sendall(f'{DEVICE} {cmd}\r'.encode('ascii'))
        while b'\r' not in data:
            left = deadline - time.monotonic()
            if left <= 0: raise TimeoutError(f'{cmd}: missing reply terminator')
            sock.settimeout(left)
            part = sock.recv(4096)
            if not part: raise RuntimeError(f'{cmd}: connection closed before reply terminator')
            data.extend(part)
            if len(data) > 65536: raise RuntimeError('Oversized reply')
    reply = bytes(data).split(b'\r', 1)[0].decode(errors='replace').strip()
    if any(line.lstrip().startswith('Error:') for line in reply.splitlines()):
        raise RuntimeError(f'{cmd}: {reply}')
    return reply


def status(text):
    f = dict(re.findall(r'^\s*([\w ]+?)\s*:\s*\[\s*(.*?)\s*\]', text, re.M))
    w, h = map(int, re.fullmatch(r'(\d+)x(\d+)', f['Image size']).groups())
    ox, oy = map(int, re.findall(r'-?\d+', f['Offset']))
    return dict(roi=[(SENSOR_WH[0]-w)//2+ox, (SENSOR_WH[1]-h)//2+oy, w, h],
                exposure=float(f['Exptime']), busy=f['Capture'].startswith('BUSY'), raw=text)


class Camera:
    """Attach to the running stream; never start, stop, crop or change exposure."""
    def __init__(self):
        import ImageStreamIOWrap
        self.original = status(command('st'))
        if not self.original['busy']:
            raise RuntimeError('Start the camera in the normal GUI before running this script.')
        self.image = ImageStreamIOWrap.Image()
        rc = self.image.open(STREAM)
        if rc not in (None, 0):
            self.image.close()
            raise RuntimeError(f'Cannot open {STREAM}: {rc}')
        self.last = self.state()[0]
        self.exposure = self.original['exposure']

    def state(self):
        md = self.image.md
        if not hasattr(md, 'cnt0'): md = md[0]
        return int(md.cnt0), bool(md.write)

    def frame(self):
        end = time.monotonic() + max(10., 3*self.exposure+2)
        while time.monotonic() < end:
            before, writing = self.state()
            if before < self.last: raise RuntimeError('Camera frame counter reset')
            if before > self.last and not writing:
                im = np.array(self.image, copy=True).T
                after, writing = self.state()
                if after == before and not writing:
                    if im.shape != tuple(self.original['roi'][2:][::-1]):
                        raise RuntimeError('Unexpected camera shape after XY to YX transpose')
                    self.last = after
                    return im, after, time.time_ns()
            time.sleep(.001)
        raise TimeoutError(f'No fresh stable camera frame: last accepted={self.last}, '
                           f'last observed={before}, write={writing}; stream={STREAM}')

    def check(self):
        s = status(command('st'))
        if not s['busy'] or s['roi'] != self.original['roi'] or abs(s['exposure']-self.exposure)>1.1e-5:
            raise RuntimeError('Camera exposure/ROI/capture changed during the test')

    def close(self):
        self.image.close()


class Bench:
    def __init__(self, args):
        self.cam = self.slm = None
        self.args = args
        try:
            from slmdisplay import SLM
            self.cam = Camera()
            screen = int(args.screen) if args.screen.lstrip('-').isdigit() else args.screen
            self.slm = SLM(screen=screen)
            self.shape = (self.slm.height, self.slm.width)
            if self.shape != (args.ny,args.nx):
                raise ValueError(f'SLM shape {self.shape} differs from requested Ny,Nx={(args.ny,args.nx)}')
            self.flat = np.full(self.shape, args.baseline, np.uint8)
            self.show(self.flat)
        except BaseException:
            self.close(); raise

    def show(self, pattern):
        seq = self.slm.show(pattern, wait=True)
        # Vsync acknowledgement is not exposure synchronisation or LC settling.
        time.sleep(self.args.settle + 2*self.cam.exposure)
        count, _ = self.cam.state()
        if count < self.cam.last: raise RuntimeError('Camera counter reset during settling')
        self.cam.last = count
        for _ in range(self.args.flush): self.cam.frame()
        return seq

    def restore_flat(self):
        # Restoring the SLM is independent of whether the camera is still alive.
        return self.slm.show(self.flat, wait=True)

    def close(self):
        # Report cleanup failures without replacing the acquisition traceback.
        if self.slm is not None:
            slm = self.slm
            try:
                if hasattr(self, 'flat'): self.restore_flat()
            except Exception as exc: print(f'Cleanup warning: flat display: {exc}', flush=True)
            try: slm.close()
            except FileNotFoundError as exc:
                if str(exc.filename or '').startswith('/psm_'):
                    print('Cleanup warning: SLM buffer already unlinked; apply the included slmdisplay patch.',flush=True)
                else: print(f'Cleanup warning: SLM close: {exc}',flush=True)
            except Exception as exc: print(f'Cleanup warning: SLM close: {exc}',flush=True)
            self.slm = None
        if self.cam is not None:
            try: self.cam.close()
            except Exception as exc: print(f'Cleanup warning: camera detach: {exc}',flush=True)
            self.cam = None
        # No attempt to restart the camera or change its settings.


def detect_spots(image,min_distance=16,fraction=.03):
    smooth=gaussian_filter(image.astype(float),1.2)
    floor=np.median(smooth);noise=1.4826*np.median(abs(smooth-floor))
    threshold=floor+max(8*noise,fraction*(smooth.max()-floor))
    y,x=np.nonzero((smooth==maximum_filter(smooth,size=2*min_distance+1))&(smooth>threshold))
    if len(x)<3:raise ValueError('Too few spots: inspect image and change detection threshold/distance')
    return np.column_stack([x,y]).astype(int)


def measure(image,variance,sat,centres,half):
    """Fixed-window signed centre of gravity, local border-median background.
    Negative background-subtracted values are retained. No frame-dependent mask.
    SEM is approximate propagated variance of the averaged image, not jitter.
    """
    yy,xx=np.mgrid[-half:half+1,-half:half+1]
    border=np.maximum(abs(xx),abs(yy))>=half-1
    patches=image[centres[:,1,None,None]+yy,centres[:,0,None,None]+xx].astype(float)
    v=variance[centres[:,1,None,None]+yy,centres[:,0,None,None]+xx].astype(float)
    bad=sat[centres[:,1,None,None]+yy,centres[:,0,None,None]+xx].any(axis=(1,2))
    bg=np.median(patches[:,border],axis=1);z=patches-bg[:,None,None];flux=z.sum(axis=(1,2))
    # Median-background variance approximation for locally Gaussian independent noise.
    vbg=np.pi/2*np.median(v[:,border],axis=1)/border.sum()
    fvar=v.sum(axis=(1,2))+z.shape[1]**4*vbg
    with np.errstate(divide='ignore',invalid='ignore'):
        cx=(z*xx).sum(axis=(1,2))/flux;cy=(z*yy).sum(axis=(1,2))/flux
        errors=[]
        for coord,c in ((xx,cx),(yy,cy)):
            weight=coord[None]-c[:,None,None]
            errors.append(np.sqrt((v*weight**2).sum(axis=(1,2))+vbg*weight.sum(axis=(1,2))**2)/flux)
        snr=flux/np.sqrt(np.maximum(fvar,1e-12))
    good=(~bad)&np.isfinite(cx)&np.isfinite(cy)&(flux>0)&(snr>=10)&(np.maximum(abs(cx),abs(cy))<.6*half)
    xy=np.column_stack((cx,cy))+centres
    sem=np.column_stack(errors)
    return xy,sem,good,flux,snr,bg


class DemoBench:
    def __init__(self,args):
        self.args=args;self.cam=self;self.shape=(args.ny,args.nx);self.flat=np.full(self.shape,args.baseline,np.uint8)
        self.exposure=.01;self.original=dict(roi=[0,0,64,48],exposure=.01,busy=True)
        self.rng=np.random.default_rng(17);self.counter=0;self.clock=1.;self.show(self.flat)
    def check(self):pass
    def show(self,cmd):
        # Deterministic local small-displacement response; no optical claim.
        yy,xx=np.indices((48,64));self.expected=np.full((48,64),100.)
        for y in (12,24,36):
            for x in (16,32,48):
                u=min(self.args.nx-2,max(1,int(x/64*self.args.nx)))
                v=min(self.args.ny-2,max(1,int(y/48*self.args.ny)))
                dx=.004*(float(cmd[v,u+1])-float(cmd[v,u-1]))
                dy=.004*(float(cmd[v+1,u])-float(cmd[v-1,u]))
                self.expected+=2500*np.exp(-((xx-x-dx)**2+(yy-y-dy)**2)/8)
        self.clock+=1;return self.counter
    def frame(self):
        self.counter+=1;self.clock+=.01
        im=np.clip(np.rint(self.expected+self.rng.normal(0,2,self.expected.shape)),0,65535).astype('uint16')
        return im,self.counter,int(self.clock*1e9)
    def restore_flat(self):self.show(self.flat)
    def close(self):self.restore_flat()


def data_root():
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for parent in (start, *start.parents):
            if (parent/'.git').exists(): return parent/'data'
    raise ValueError('No Git repository found; provide --output /path/IM.fits')


def table(name, columns):
    return fits.BinTableHDU.from_columns([fits.Column(name=k,format=fmt,array=v)
        for k,fmt,v in columns],name=name)


def append(path, hdus):
    # Each completed acquisition survives a later timeout. COMPLETE stays false.
    with fits.open(path,mode='append',memmap=False) as f:
        for h in hdus:
            h.add_checksum();f.append(h)
        f.flush(output_verify='exception')


def capture(bench,a,pattern,path,index,kind,mode=-1,amp=0.):
    bench.cam.check();seq=bench.show(pattern)
    stamps=[];counts=[]
    for i in range(a.frames_per_probe):
        im,c,t=bench.cam.frame()
        if a.demo and kind=='dark':im=100+bench.rng.normal(0,2,im.shape)
        if a.roi:
            x,y,w,h=a.roi
            if x<0 or y<0 or x+w>im.shape[1] or y+h>im.shape[0]:raise ValueError('Software ROI outside camera frame')
            im=im[y:y+h,x:x+w]
        im=im.astype(float)
        if not np.isfinite(im).all():raise ValueError('Nonfinite image')
        if i==0:mean=np.zeros_like(im);m2=mean.copy();sat=np.zeros(im.shape,bool)
        delta=im-mean;mean+=delta/(i+1);m2+=delta*(im-mean);sat|=im>=a.saturation
        stamps.append(t);counts.append(c)
    if np.any(np.diff(counts)<=0):raise RuntimeError('Repeated frame counters')
    var=m2/(a.frames_per_probe-1)/a.frames_per_probe
    h=fits.Header(dict(STATE=index,KIND=kind,MODEID=mode,AMPL=amp,NFRAMES=a.frames_per_probe,
        EXPTIME=bench.cam.exposure,HOSTNS=int(np.mean(stamps)),SLMSEQ=int(seq or 0),BUNIT='ADU',
        CMDHASH=hashlib.sha256(pattern.tobytes()).hexdigest()))
    append(path,[fits.ImageHDU(mean.astype('float32'),h,name=f'IMAGE{index:05d}'),
        fits.ImageHDU(var.astype('float32'),fits.Header({'BUNIT':'ADU^2'}),name=f'VAR{index:05d}'),
        fits.ImageHDU(sat.astype('uint8'),name=f'SAT{index:05d}'),
        table(f'TIME{index:05d}',[('COUNTER','K',counts),('HOST_NS','K',stamps)])])
    print(f'{index:05d} {kind}: mode {mode}, amplitude {amp:g}; max mean {mean.max():.0f} ADU',flush=True)
    return dict(image=mean,var=var,sat=sat,time=float(np.mean(stamps)),index=index)


def render(basis,mode,amp):
    h=basis[0].header;mask=basis['PUPIL'].data.astype(bool)
    z=np.asarray(basis[0].data[mode],float)
    if not np.isfinite(z).all():raise ValueError('Nonfinite basis')
    if np.any(z[~mask]!=0):raise ValueError('Basis has nonzero modulation outside pupil')
    crop=h['BASELINE']+np.rint(amp*z)
    if crop.min()<0 or crop.max()>255:raise ValueError(f'Mode {mode}, amplitude {amp}: exceeds 8-bit headroom; lower --amplitudes')
    out=np.full((h['SLMNY'],h['SLMNX']),h['BASELINE'],np.uint8)
    out[h['Y0']:h['Y1'],h['X0']:h['X1']]=crop.astype(np.uint8)
    return out


def run(a):
    with fits.open(a.basis,memmap=True) as b:
        h=b[0].header
        if h.get('SCHEMA')!='SLMBASIS' or h.get('SCHEMAV')!=1 or not h.get('COMPLETE'):raise ValueError('Need completed SLMBASIS v1 FITS')
        if not h['REGVALID'] and not (a.allow_unvalidated or a.demo):raise ValueError('Basis registration is unvalidated; use --allow-unvalidated only for exploratory runs')
        a.nx,a.ny,a.baseline=h['SLMNX'],h['SLMNY'],h['BASELINE']
        modes=a.modes if a.modes is not None else list(range(h['NMODES']))
        if len(set(modes))!=len(modes) or min(modes)<0 or max(modes)>=h['NMODES']:raise ValueError('Invalid mode indices')
        if b[0].data.shape!=(h['NMODES'],h['Y1']-h['Y0'],h['X1']-h['X0']):raise ValueError('Basis shape mismatch')
        # Preflight every requested command before opening hardware.
        for m in modes:
            for sign in (-1,1):render(b,m,sign*max(a.amplitudes))
        name=re.sub('[^A-Za-z0-9_-]','_',h['BASISTYP'])
        path=a.output or data_root()/f'IM_{name}_{time.strftime("%Y%m%d_%H%M%S")}.fits'
        path=path.resolve();path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(path)
        digest=hashlib.sha256()
        with a.basis.open('rb') as src:
            for block in iter(lambda:src.read(1024*1024),b''):digest.update(block)
        header=fits.Header(dict(SCHEMA='SLMIM',SCHEMAV=1,COMPLETE=False,BASISTYP=h['BASISTYP'],
            BASISHA=digest.hexdigest(),BASISFIL=a.basis.name,REGVALID=h['REGVALID'],DEMO=a.demo,
            BASELINE=a.baseline,NMODES=len(modes),NAMP=len(a.amplitudes),AMPUNIT='grey RMS',
            BUNIT='pixel/grey',SLOPORD='all x, then all y',NFRAMES=a.frames_per_probe,
            HALF=a.half_window,SETTLE=a.settle,NFLUSH=a.flush,SATADU=a.saturation,
            DRIFTCOR=True,FLUXNORM=False))
        header['COMMENT']='IM[a,slope,mode]=(drift-corrected centroid+ - centroid-)/(2*amplitude).'
        header['COMMENT']='Slopes are pixel displacement, not calibrated wavefront angle. No phase LUT.'
        fits.HDUList([fits.PrimaryHDU(header=header),b['MODES'].copy()]).writeto(path,checksum=True)
        bench=None;idx=0;products=[];allvalid=[];allcent=[];allflux=[];allsem=[];state_rows=[]
        try:
            bench=DemoBench(a) if a.demo else Bench(a)
            with fits.open(path,mode='update') as f:
                for k,v in zip(('CAMX','CAMY','CAMW','CAMH'),bench.cam.original['roi']):f[0].header[k]=v
                for k,v in zip(('CROPX','CROPY'),a.roi[:2] if a.roi else (0,0)):f[0].header[k]=v
                f[0].header['EXPTIME']=bench.cam.exposure
            print(f'Output: {path}\n{len(modes)*len(a.amplitudes)*5+2} states × {a.frames_per_probe} frames. Camera settings remain fixed.')
            if not a.demo:input('Block laser for matched dark; camera running. Enter: ')
            dark=capture(bench,a,bench.flat,path,idx,'dark');idx+=1
            if not a.demo:input('Unblock laser; keep power stable and unsaturated. Enter: ')
            ref=capture(bench,a,bench.flat,path,idx,'reference');idx+=1
            image=ref['image']-dark['image'];centres=detect_spots(image,a.min_distance,a.threshold)
            r=a.half_window;H,W=image.shape
            centres=centres[(centres[:,0]>=r)&(centres[:,0]<W-r)&(centres[:,1]>=r)&(centres[:,1]<H-r)]
            if not len(centres):raise ValueError('No contained spots')
            xy,sem,good,flux,snr,bg=measure(image,ref['var']+dark['var'],ref['sat']|dark['sat'],centres,r)
            centres=centres[good];refxy=xy[good];refflux=flux[good]
            if len(centres)<3:raise ValueError('Fewer than three valid reference spots')
            fig,ax=plt.subplots(figsize=(9,8));ax.imshow(image,cmap='gray',vmax=np.percentile(image,99.5))
            ax.scatter(*centres.T,s=30,facecolors='none',edgecolors='r');ax.set_title(f'{len(centres)} reference lenslets; fixed windows ±{r}px')
            fig.savefig(path.with_name(path.stem+'_spots.png'));plt.close(fig)
            append(path,[table('SPOTS',[('ID','J',np.arange(len(centres))),('X','D',centres[:,0]),('Y','D',centres[:,1]),
                ('REF_X','D',refxy[:,0]),('REF_Y','D',refxy[:,1]),('REF_FLUX','D',refflux)])])
            if not a.demo and input(f'Inspect {path.stem}_spots.png. Proceed with {len(centres)} spots? [y/N]: ').lower()!='y':raise RuntimeError('Stopped before probes by user')
            for ai,amp in enumerate(a.amplitudes):
                for mi,m in enumerate(modes):
                    states=[];measurements=[]
                    for kind,signed in [('flat_before',0),('plus',amp),('flat_middle',0),('minus',-amp),('flat_after',0)]:
                        st=capture(bench,a,render(b,m,signed),path,idx,kind,m,signed);idx+=1;states.append(st)
                        result=measure(st['image']-dark['image'],st['var']+dark['var'],st['sat']|dark['sat'],centres,r)
                        measurements.append(result);allcent.append(result[0]);allsem.append(result[1]);allvalid.append(result[2]);allflux.append(result[3])
                        state_rows.append((st['index'],m,ai,signed,kind))
                    def residual(k,left,right,field):
                        w=(states[k]['time']-states[left]['time'])/(states[right]['time']-states[left]['time'])
                        vals=[v['image'] for v in states] if field=='image' else [v[0] for v in measurements]
                        return vals[k]-((1-w)*vals[left]+w*vals[right])
                    response=(residual(1,0,2,'image')-residual(3,2,4,'image'))/2
                    bad=np.logical_or.reduce([v['sat'] for v in states])|dark['sat'];response[bad]=np.nan
                    d=(residual(1,0,2,'xy')-residual(3,2,4,'xy'))/2
                    valid=np.logical_and.reduce([v[2] for v in measurements]);d[~valid]=np.nan
                    products.append((ai,mi,d))
                    append(path,[fits.ImageHDU(response.astype('float32'),fits.Header(dict(MODEID=m,AMPIDX=ai,AMPL=amp,BUNIT='ADU',HALFDIFF=True)),name=f'RESP{ai:03d}_{m:05d}')])
            bench.cam.check()
        finally:
            if bench is not None:bench.close()
        # Common lenslet selection keeps row ordering identical across all columns/amplitudes.
        common=np.logical_and.reduce(allvalid)
        if common.sum()<3:raise ValueError('Fewer than three common valid lenslets; acquisitions retained, COMPLETE=false')
        half=np.full((len(a.amplitudes),2*common.sum(),len(modes)),np.nan)
        for ai,mi,d in products:half[ai,:,mi]=d[common].T.reshape(-1)
        im=half/np.asarray(a.amplitudes)[:,None,None]
        singular=np.array([np.linalg.svd(v,compute_uv=False) for v in im])
        columns=list(zip(*state_rows))
        append(path,[fits.ImageHDU(im.astype('float32'),fits.Header({'BUNIT':'pixel/grey','AXORDER':'amplitude,slope,mode'}),name='IM'),fits.ImageHDU(half.astype('float32'),fits.Header({'BUNIT':'pixel'}),name='HALFDIFF'),
            fits.ImageHDU(singular,name='SINGULAR'),fits.ImageHDU(common.astype('uint8'),name='VALID'),
            fits.ImageHDU(np.asarray(allcent)-refxy,name='SLOPES'),fits.ImageHDU(np.asarray(allcent),name='CENTROIDS'),
            fits.ImageHDU(np.asarray(allsem),name='CENT_SEM'),fits.ImageHDU(np.asarray(allflux),name='FLUX'),
            fits.ImageHDU(np.asarray(allvalid,dtype='uint8'),name='STATE_VALID'),
            table('STATES',[(k,fmt,v) for k,fmt,v in zip(('STATE','MODE_ID','AMP_IDX','AMPLITUDE','KIND'),('J','J','J','D','16A'),columns)]),
            table('IM_MODES',[('MODE_ID','J',modes)]),table('AMPLITUDES',[('AMPLITUDE','D',a.amplitudes)]),
            table('SLOPE_ROWS',[('SPOT_ID','J',np.tile(np.flatnonzero(common),2)),('AXIS','1A',['x']*common.sum()+['y']*common.sum())])])
        with fits.open(path,mode='update',memmap=True) as f:
            f[0].header['COMPLETE']=True;f[0].header['NSPOTS']=len(centres);f[0].header['NGOOD']=int(common.sum())
            for ext in f:ext.add_checksum()
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ai,amp in enumerate(a.amplitudes):axes[0].semilogy(singular[ai],label=f'{amp:g} grey RMS')
        axes[0].legend();axes[0].set(title='IM singular values',xlabel='Index',ylabel='pixel / grey RMS')
        axes[1].plot(a.amplitudes,np.linalg.norm(im,axis=1),'o-');axes[1].set(title='Per-mode response vs amplitude',xlabel='grey RMS',ylabel='Slope-vector norm / grey RMS')
        axes[2].plot(np.median(np.asarray(allflux)[:,common]/refflux[common],axis=1));axes[2].set(title='Median lenslet intensity / reference',xlabel='Measurement state',ylabel='Flux ratio')
        fig.tight_layout();fig.savefig(path.with_name(path.stem+'_diagnostics.png'));plt.close(fig)
        print(f'Saved {path}: {len(modes)} modes, {common.sum()}/{len(centres)} common lenslets. No phase calibration.')
        return path


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('basis',type=Path);p.add_argument('--output',type=Path,help='New output FITS path')
    p.add_argument('--modes',nargs='+',type=int,help='Zero-based basis mode IDs; default all')
    p.add_argument('--amplitudes',nargs='+',type=float,default=[5.],help='Positive RMS grey coefficients; both signs acquired')
    p.add_argument('--frames-per-probe',type=int,default=16)
    p.add_argument('--half-window',type=int,default=10);p.add_argument('--min-distance',type=int,default=24)
    p.add_argument('--threshold',type=float,default=.03,help='Detection fraction of smoothed peak')
    p.add_argument('--roi',nargs=4,type=int,metavar=('X','Y','W','H'),help='Software crop relative to current camera frame')
    p.add_argument('--settle',type=float,default=.3);p.add_argument('--flush',type=int,default=3);p.add_argument('--screen',default='1')
    p.add_argument('--saturation',type=float,default=65000)
    p.add_argument('--allow-unvalidated',action='store_true');p.add_argument('--demo',action='store_true')
    a=p.parse_args()
    if not np.isfinite(a.amplitudes+[a.settle,a.threshold,a.saturation]).all():p.error('Numeric values must be finite')
    if min(a.amplitudes)<=0 or len(set(a.amplitudes))!=len(a.amplitudes):p.error('Distinct positive amplitudes required')
    if a.frames_per_probe<2 or a.flush<2 or a.settle<0 or a.half_window<3 or a.min_distance<2*a.half_window+1:p.error('Need frames>=2, flush>=2, settle>=0, half-window>=3, min-distance>=2*half-window+1')
    if not 0<a.threshold<1 or a.saturation<=0:p.error('Invalid threshold/saturation')
    if a.roi and (min(a.roi[:2])<0 or min(a.roi[2:])<=0):p.error('Invalid ROI')
    return a

if __name__=='__main__':run(arguments())
