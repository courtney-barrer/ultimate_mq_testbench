#!/usr/bin/env python3
"""Fast strip SLM registration (v2), standalone. No camera settings are changed.
Dependencies: numpy, scipy, matplotlib; live acquisition also needs
ImageStreamIOWrap and the installed slmdisplay package.
python IM/register_slm.py --output registration_01
python IM/register_slm.py --analyse registration_01
python IM/register_slm.py --demo --output registration_demo
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree, ConvexHull, Voronoi
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import socket
import re
from scipy.ndimage import label

# Inlined response-check support: acquisition, diagnostics and demo unchanged.
# camera_command is the socket helper; command below creates SLM patterns.
HOST, PORT, DEVICE, STREAM = '127.0.0.1', 18010, 'dcam0', 'dcam0_data'


SENSOR_WH = (2304, 2304)


SATURATION_ADU = 65000


RESPONSE_THRESHOLD = 5.0


MIN_RESPONSE_ADU = 1.0


MIN_CONNECTED_PIXELS = 3


def camera_command(cmd, timeout=30.):
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
        self.original = status(camera_command('st'))
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
        s = status(camera_command('st'))
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
            self.flat = np.full(self.shape, 128, np.uint8)
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


def regions(shape, grid):
    ys = np.linspace(0, shape[0], grid+1, dtype=int)
    xs = np.linspace(0, shape[1], grid+1, dtype=int)
    return [(int(xs[c]), int(ys[r]), int(xs[c+1]), int(ys[r+1]))
            for r in range(grid) for c in range(grid)]


def save_json(path, data):
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False))
    pending.replace(path)


def capture(bench, cmd, n, stem, save_data):
    """Streaming mean/variance: only one raw frame in RAM unless saving a memmap."""
    bench.cam.check()
    seq = bench.show(cmd)
    times, counters = [], []
    raw = None
    meta = dict(complete=False, valid_frames=0, slm_sequence=int(seq or 0))
    if save_data:
        np.save(str(stem)+'_command.npy', cmd)
        save_json(stem.with_suffix('.json'), meta)
    try:
        for i in range(n):
            im, count, stamp = bench.cam.frame()
            if i == 0:
                mean = np.zeros(im.shape, np.float64); m2 = np.zeros_like(mean)
                saturated = np.zeros(im.shape, bool)
                if save_data:
                    raw = np.lib.format.open_memmap(str(stem)+'_frames.npy', mode='w+',
                                                    dtype=im.dtype, shape=(n,*im.shape))
            if not np.isfinite(im).all(): raise RuntimeError('Nonfinite camera values')
            if raw is not None: raw[i] = im
            delta = im-mean; mean += delta/(i+1); m2 += delta*(im-mean)
            saturated |= im >= SATURATION_ADU
            times.append(stamp); counters.append(count)
            meta['valid_frames'] = i+1
        meta['complete'] = True
        return dict(mean=mean.astype('float32'), varmean=(m2/(n-1)/n).astype('float32'),
                    saturated=saturated, time=float(np.mean(np.array(times)/1e9)))
    finally:
        if raw is not None: raw.flush(); del raw
        if save_data:
            meta.update(counters=counters, host_time_ns=times)
            save_json(stem.with_suffix('.json'), meta)


def interpolate_static(left, right, target):
    span = right['time']-left['time']
    w = np.clip((target['time']-left['time'])/span, 0, 1) if span>0 else .5
    mean = (1-w)*left['mean']+w*right['mean']
    var = (1-w)**2*left['varmean']+w*w*right['varmean']
    return mean, var


def response(states):
    """Both sign-vs-static and sign-vs-sign contrasts, repeated in reverse order."""
    contrasts, variances = [], []
    for k in (0,1):
        b0,b1 = states[f'b{k}'],states[f'b{k+1}']
        p,m = states[f'p{k}'],states[f'm{k}']
        bp,vp = interpolate_static(b0,b1,p); bm,vm = interpolate_static(b0,b1,m)
        contrasts.append(np.array([p['mean']-bp, m['mean']-bm, p['mean']-m['mean']]))
        variances.append(np.array([p['varmean']+vp,m['varmean']+vm,p['varmean']+m['varmean']]))
    # Empirical static motion/drift term conservatively guards the noise-only SEM.
    drift = np.maximum(abs(states['b1']['mean']-states['b0']['mean']),
                       abs(states['b2']['mean']-states['b1']['mean']))
    bad = np.logical_or.reduce([s['saturated'] for s in states.values()])
    scores = [abs(d)/np.sqrt(v+drift[None]**2+MIN_RESPONSE_ADU**2)
              for d,v in zip(contrasts,variances)]
    repeated = (contrasts[0]*contrasts[1]>0)
    score = np.max(np.minimum(scores[0],scores[1])*repeated,axis=0)
    score[bad] = 0
    components,n = label((score>=RESPONSE_THRESHOLD)&~bad)
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    mask = sizes[components]>=MIN_CONNECTED_PIXELS
    delta = (contrasts[0]+contrasts[1])/2
    return dict(delta=delta, score=score, mask=mask, drift=drift, bad=bad,
                static=(states['b0']['mean']+states['b1']['mean']+states['b2']['mean'])/3)


def finish_plot(fig, path, show):
    import matplotlib.pyplot as plt
    fig.tight_layout(); fig.savefig(path, dpi=125)
    print(f'Plot: {path}',flush=True)
    if show: plt.show(block=True)
    plt.close(fig)


def detail_plot(result, cmd, path, title, show):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(2,4,figsize=(16,8))
    panels = [cmd,result['static'],*result['delta'],result['drift'],result['score'],result['mask']]
    names = ['Positive SLM command [grey]','Static camera [ADU]',
             'Positive − interpolated static [ADU]','Negative − interpolated static [ADU]',
             'Positive − negative [ADU]','Max static change [ADU]',
             'Repeat response / noise + drift','Repeatable response mask']
    for j,(a,im,name) in enumerate(zip(ax.flat,panels,names)):
        kw = dict(origin='upper',cmap='viridis')
        if j in (2,3,4):
            v = max(1.,float(np.percentile(abs(im),99.9)))
            kw.update(cmap='RdBu_r',vmin=-v,vmax=v)
        elif j==1: kw.update(vmin=float(np.percentile(im,1)),vmax=float(np.percentile(im,99.9)))
        elif j==6: kw.update(vmin=0,vmax=max(10.,float(np.percentile(im,99.9))))
        art = a.imshow(im,**kw); fig.colorbar(art,ax=a,shrink=.7)
        a.set_title(name,fontsize=10)
        a.set_xlabel('SLM x' if j==0 else 'camera x'); a.set_ylabel('SLM y' if j==0 else 'camera y')
    fig.suptitle(title+f"; {result['mask'].sum()} response pixels; {result['bad'].sum()} saturated pixels excluded")
    finish_plot(fig,path,show)


class SyntheticBench:
    """Hardware-free spatially local image response with a partially lit SLM."""
    def __init__(self,args):
        self.args=args; self.shape=(128,160); self.flat=np.full(self.shape,128,np.uint8)
        self.cam=self; self.exposure=.01; self.counter=0; self.clock=0.; self.rng=np.random.default_rng(7)
        self.original=dict(roi=[0,0,96,80],exposure=.01,busy=True)
        yy,xx=np.indices((80,96)); self.base=np.full((80,96),100.)
        self.spots=[]
        for y in (18,40,62):
            for x in (20,48,76):
                u,v=(x-20)*1.1+40,(y-18)*1.1+35
                spot=2500*np.exp(-((xx-x)**2+(yy-y)**2)/8)
                self.base+=spot; self.spots.append((u,v,spot))
        self.show(self.flat)
    def check(self): pass
    def show(self,cmd):
        from scipy.ndimage import sobel
        gx=sobel(cmd.astype(float),axis=1)/8; gy=sobel(cmd.astype(float),axis=0)/8
        self.expected=self.base.copy()
        for u,v,spot in self.spots:
            # Fixed local mapping; signed shape perturbation, not an optical model.
            dx,dy=gx[int(v),int(u)],gy[int(v),int(u)]
            sy,sx=np.gradient(spot)
            self.expected+=.018*(dx*sx+dy*sy)
        self.clock+=1
        return self.counter
    def frame(self):
        self.counter+=1;self.clock+=.01
        a=np.clip(np.rint(self.expected+self.rng.normal(0,3,self.expected.shape)),0,65535).astype(np.uint16)
        return a,self.counter,int(self.clock*1e9)
    def restore_flat(self): return self.show(self.flat)
    def close(self): self.restore_flat()

# Registration algorithm (unchanged apart from local helper references).


def detect(image, distance, fraction, half):
    im=gaussian_filter(image.astype(float),1.2)
    floor=np.median(im);threshold=floor+max(8*1.4826*np.median(abs(im-floor)),fraction*(im.max()-floor))
    y,x=np.nonzero((im==maximum_filter(im,size=2*distance+1))&(im>threshold))
    ok=(x>=half)&(y>=half)&(x<im.shape[1]-half)&(y<im.shape[0]-half)
    centres=np.column_stack((x[ok],y[ok]))
    if len(centres)<6:raise ValueError('Fewer than six spots; inspect detection settings')
    if cKDTree(centres).query(centres,k=2)[0][:,1].min()<=2*half:
        raise ValueError('Spot windows overlap: reduce --half-window or increase --min-distance')
    return centres


def command(shape,box,axis,phase,sign,amplitude,cycles):
    # Same carrier pitch at every hierarchy level; quadrature covers gradient nodes.
    x0,y0,x1,y1=box
    u=np.arange(x0,x1)[None,:] if axis=='x' else np.arange(y0,y1)[:,None]
    length=shape[1] if axis=='x' else shape[0]
    wave=np.sin(2*np.pi*cycles*u/length+np.deg2rad(phase))
    result=np.full(shape,128,np.uint8)
    result[y0:y1,x0:x1]=(128+sign*np.rint(amplitude*wave)).astype(np.uint8)
    return result


def spot_scores(result,centres,half):
    yy,xx=np.mgrid[-half:half+1,-half:half+1]
    score=result['score'][centres[:,1,None,None]+yy,centres[:,0,None,None]+xx]
    bad=result['bad'][centres[:,1,None,None]+yy,centres[:,0,None,None]+xx].any(axis=(1,2))
    # Third strongest repeatable pixel: no reliance on a single noisy pixel.
    values=np.partition(score.reshape(len(centres),-1),-3,axis=1)[:,-3]
    values[bad]=0
    return values,bad


def split(box):
    x0,y0,x1,y1=box;xm=(x0+x1)//2;ym=(y0+y1)//2
    return [(x0,y0,xm,ym),(xm,y0,x1,ym),(x0,ym,xm,y1),(xm,ym,x1,y1)]


def plot_windows(image,centres,half,path):
    fig,ax=plt.subplots(figsize=(9,8));ax.imshow(image,origin='upper',vmax=np.percentile(image,99.8))
    for x,y in centres:ax.add_patch(Rectangle((x-half,y-half),2*half+1,2*half+1,fill=False,color='r',lw=.4))
    ax.set(title=f'{len(centres)} fixed lenslet windows',xlabel='Cropped camera x [pixel]',ylabel='Cropped camera y [pixel]')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def fast_response(states):
    """Four-state diagnostic, not a repeatability claim. Static drift is penalised."""
    b0,b1=states['b0'],states['b1'];p,m=states['p0'],states['m0']
    bp,vp=interpolate_static(b0,b1,p);bm,vm=interpolate_static(b0,b1,m)
    delta=np.array([p['mean']-bp,m['mean']-bm,p['mean']-m['mean']])
    var=np.array([p['varmean']+vp,m['varmean']+vm,p['varmean']+m['varmean']])
    drift=abs(b1['mean']-b0['mean'])
    bad=np.logical_or.reduce([s['saturated'] for s in states.values()])
    score=np.max(abs(delta)/np.sqrt(var+drift[None]**2+MIN_RESPONSE_ADU**2),axis=0)
    score[bad]=0
    return dict(delta=delta,score=score,mask=score>=RESPONSE_THRESHOLD,
                drift=drift,bad=bad,static=(b0['mean']+b1['mean'])/2)


def trial(bench,args,root,centres,tag,box,axes,repeat=False,diagnostic=False):
    values=np.zeros(len(centres));bad=np.zeros(len(centres),bool);started=time.monotonic()
    names=('b0','p0','m0','b1','m1','p1','b2') if repeat else ('b0','p0','m0','b1')
    empty=box[0]>=box[2] or box[1]>=box[3]
    if empty:return values,bad,0.,0
    for axis in axes:
        for phase in (0.,90.):
            states={}
            cmds={'b':bench.flat,'p':command(bench.shape,box,axis,phase,1,args.amplitude,args.cycles),
                  'm':command(bench.shape,box,axis,phase,-1,args.amplitude,args.cycles)}
            for name in names:
                states[name]=capture(bench,cmds[name[0]],args.frames_per_probe,
                                         root/f'{tag}_{axis}_{phase:g}_{name}',args.save_data)
            result=response(states) if repeat else fast_response(states)
            v,b=spot_scores(result,centres,args.half_window);values=np.maximum(values,v);bad|=b
            if diagnostic:detail_plot(result,cmds['p'],root/f'{tag}_{axis}_{phase:g}.png',tag,False)
    values[bad]=0
    return values,bad,time.monotonic()-started,len(axes)*2*len(names)


def illuminated_bounds(shape,quadrants,scores,full,threshold):
    ny,nx=shape;active=(scores>=threshold).any(axis=1)
    # If quadrants fail to cover full-SLM detections, do not silently crop those lenslets away.
    uncovered=(full>=threshold)&~(scores>=threshold).any(axis=0)
    if not active.any() or uncovered.any():return (0,0,nx,ny),True
    b=np.array(quadrants)[active];lo=b[:,:2].min(0);hi=b[:,2:].max(0)
    pad=.1*(hi-lo)
    return tuple(np.r_[np.maximum(0,np.floor(lo-pad)),np.minimum([nx,ny],np.ceil(hi+pad))].astype(int)),False


def strip_plan(shape,bounds,positions,width_steps):
    """Two overlapping 1D scans plus zero-support guards, with full transverse extent.
    The carrier is transverse to the strip: moving its window does not move its sine nodes.
    """
    ny,nx=shape;records=[]
    for axis in ('x','y'):
        dim=0 if axis=='x' else 1;lo,hi=bounds[dim],bounds[dim+2]
        step=(hi-lo)/(positions-1);width=width_steps*step
        count=int(np.ceil(width_steps/2))+1
        for c in lo+np.arange(-count,positions+count)*step:
            start=max(0,int(np.floor(c-width/2)));end=min(shape[1-dim],int(np.ceil(c+width/2)))
            start=min(shape[1-dim],start);end=max(0,end)
            box=(start,0,end,ny) if axis=='x' else (0,start,nx,end)
            records.append(dict(stage='strip',axis=axis,centre=float(c),step=float(step),width=float(width),
                                box=list(box),carrier_axes=['y' if axis=='x' else 'x']))
    return records


def scan(args):
    root=args.output;root.mkdir(parents=True,exist_ok=False);bench=None
    settings=dict(schema='slm-strip-scan-2',complete=False,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},records=[])
    scores=[];bad=[];durations=[]
    def checkpoint():
        with (root/'responses.pending.npz').open('wb') as f:
            np.savez_compressed(f,scores=np.array(scores),saturated=np.array(bad))
        (root/'responses.pending.npz').replace(root/'responses.npz')
        save_json(root/'scan.json',settings)
    def record(spec,repeat=False,diagnostic=False):
        tag=f"{spec['stage']}_{len(scores):04d}"
        v,b,elapsed,nstates=trial(bench,args,root,centres,tag,spec['box'],spec['carrier_axes'],repeat,diagnostic)
        row=dict(spec,repeat=repeat,elapsed_s=elapsed,display_states=nstates,responding_lenslets=int((v>=args.threshold).sum()))
        settings['records'].append(row);scores.append(v);bad.append(b)
        if nstates:durations.append(elapsed/nstates)
        checkpoint()
        print(f'{tag}: {row["responding_lenslets"]} lenslets; {elapsed:.1f} s',flush=True)
        return len(scores)-1
    try:
        bench=SyntheticBench(args) if args.demo else Bench(args)
        settings.update(camera=bench.cam.original,slm_shape_yx=list(bench.shape),synthetic=args.demo)
        save_json(root/'scan.json',settings)
        if not args.demo:input('Laser ON, camera fixed and unsaturated. Enter to detect spots: ')
        ref=capture(bench,bench.flat,args.frames_per_probe,root/'initial',args.save_data)
        centres=detect(ref['mean'],args.min_distance,args.detection_fraction,args.half_window)
        np.savez_compressed(root/'reference.npz',mean=ref['mean'],centres_xy=centres)
        plot_windows(ref['mean'],centres,args.half_window,root/'lenslet_windows.png')
        print(f'Inspect {root}/lenslet_windows.png',flush=True)
        if not args.demo and input('Are the lenslet windows correct? [y/N]: ').strip().lower()!='y':return
        ny,nx=bench.shape
        full=record(dict(stage='full',box=[0,0,nx,ny],carrier_axes=['x','y']),True,True)
        if (scores[full]>=args.threshold).sum()<6:raise ValueError('Insufficient repeatable full-SLM response')
        if not args.demo and input('Inspect full_* plots. Continue with quadrants and strip scans? [y/N]: ').strip().lower()!='y':return
        quadrants=regions(bench.shape,2);qids=[]
        for box in quadrants:qids.append(record(dict(stage='coarse',box=list(box),carrier_axes=['x','y'])))
        bounds,fallback=illuminated_bounds(bench.shape,quadrants,np.array(scores)[qids],scores[full],args.threshold)
        settings.update(scan_bounds=list(map(int,bounds)),coarse_fallback_to_full_slm=bool(fallback))
        plan=strip_plan(bench.shape,bounds,args.positions,args.width_steps)
        nstates=sum(8 for r in plan if r['box'][0]<r['box'][2] and r['box'][1]<r['box'][3])+28*args.repeat_positions
        estimate=np.median(durations)*nstates
        settings['estimated_remaining_seconds']=float(estimate);checkpoint()
        print(f'{len(plan)} strip positions including guards; about {estimate/60:.1f} minutes remaining, based on measured acquisition speed. Bounds={bounds}; full-SLM fallback={fallback}.',flush=True)
        if not args.demo and input('Proceed with this strip scan? [y/N]: ').strip().lower()!='y':return
        stripids=[]
        for j,spec in enumerate(plan):
            stripids.append(record(spec))
            remaining=len(plan)-j-1
            print(f'  {remaining} strip positions remaining (~{remaining*8*np.median(durations)/60:.1f} min, plus repeat checks)',flush=True)
        # Repeat responsive positions distributed across each measured coordinate range.
        for axis in ('x','y'):
            ids=[i for i in stripids if settings['records'][i]['axis']==axis and (scores[i]>=args.threshold).any()]
            chosen=np.unique(np.linspace(0,len(ids)-1,min(args.repeat_positions,len(ids))).astype(int)) if ids else []
            for j in chosen:
                i=ids[j];spec=dict(settings['records'][i]);spec.update(stage='repeat',original_record=i)
                record(spec,True)
        settings['complete']=True;checkpoint()
    except BaseException as exc:
        settings['error']=repr(exc);save_json(root/'scan.json',settings);raise
    finally:
        if bench is not None:bench.close()
    analyse(root)


def locate_axis(records,scores,axis,threshold):
    ids=[i for i,r in enumerate(records) if r['stage']=='strip' and r['axis']==axis]
    ids=sorted(ids,key=lambda i:records[i]['centre'])
    c=np.array([records[i]['centre'] for i in ids]);v=scores[ids]
    n=scores.shape[1];position=np.full(n,np.nan);extent=np.full(n,np.nan);hits=[[] for _ in range(n)]
    if len(ids)<5:return position,extent,hits,ids
    step=float(np.median(np.diff(c)));width=records[ids[0]]['width']
    for k in range(n):
        active=np.flatnonzero(v[:,k]>=threshold)
        # Need a bracket on both sides, at least two overlapping hits and a single compact lobe.
        if len(active)<2 or active[0]==0 or active[-1]==len(ids)-1:continue
        if np.any(np.diff(active)>2) or c[active[-1]]-c[active[0]]>width+2*step:continue
        # Midpoint of threshold-crossing brackets; no signal-strength/distance assumption.
        lo=(c[active[0]-1]+c[active[0]])/2;hi=(c[active[-1]]+c[active[-1]+1])/2
        position[k]=(lo+hi)/2
        # Conservative localisation scale includes sampling and finite strip width.
        extent[k]=max(step,width/2)
        hits[k]=[ids[i] for i in active]
    return position,extent,hits,ids


def strip_correspondences(centres,records,scores,threshold):
    x,dx,hx,ix=locate_axis(records,scores,'x',threshold)
    y,dy,hy,iy=locate_axis(records,scores,'y',threshold)
    uv=np.column_stack((x,y));unc=np.column_stack((dx,dy));hits=[a+b for a,b in zip(hx,hy)]
    # Spatial bins group correlated strip estimates for region-held-out validation.
    valid=np.isfinite(uv).all(1);group=np.full(len(centres),-1,int)
    if valid.any():
        bins=np.floor((uv[valid]-np.nanmin(uv[valid],axis=0))/np.nanmedian(unc[valid],axis=0)/2).astype(int)
        _,group[valid]=np.unique(bins,axis=0,return_inverse=True)
    return uv,unc,group,hits


def repeat_quality(records,scores,threshold):
    values=[];counts={axis:0 for axis in ('x','y')}
    for i,r in enumerate(records):
        if r['stage']!='repeat':continue
        before=scores[r['original_record']]>=threshold;after=scores[i]>=threshold
        union=before|after
        agreement=float((before&after).sum()/union.sum()) if union.any() else 0.
        values.append(dict(record=i,axis=r['axis'],jaccard=agreement));counts[r['axis']]+=1
    good=all(v>=2 for v in counts.values()) and bool(values) and min(v['jaccard'] for v in values)>=.5
    return dict(passed=bool(good),checks=values,checks_per_axis=counts)


def fit_affine(xy,uv,unc):
    origin=xy.mean(0);scale=np.std(xy,axis=0)
    if np.any(scale<1):raise ValueError('Insufficient two-dimensional lenslet coverage')
    design=np.column_stack(((xy-origin)/scale,np.ones(len(xy))))
    if np.linalg.matrix_rank(design)<3:raise ValueError('Degenerate lenslet positions')
    initial=np.linalg.lstsq(design,uv,rcond=None)[0]
    fit=least_squares(lambda p:((design@p.reshape(3,2)-uv)/unc).ravel(),initial.ravel(),loss='soft_l1')
    b=fit.x.reshape(3,2);a=b[:2].T/scale[None,:];offset=b[2]-a@origin
    return np.column_stack((a,offset))


def transform(xy,t):return np.asarray(xy)@t[:,:2].T+t[:,2]


def plot_strip_responses(root,records,scores,threshold):
    fig,axs=plt.subplots(1,2,figsize=(13,6))
    for ax,axis in zip(axs,('x','y')):
        ids=sorted([i for i,r in enumerate(records) if r['stage']=='strip' and r['axis']==axis],key=lambda i:records[i]['centre'])
        if not ids:continue
        positions=np.array([records[i]['centre'] for i in ids]);v=scores[ids].T
        # Sort display only by strongest response location; lenslet IDs in data remain fixed.
        order=np.argsort(np.argmax(v,axis=1))
        im=ax.imshow(v[order],origin='lower',aspect='auto',extent=[positions[0],positions[-1],0,len(v)],vmin=0,vmax=2*threshold,cmap='magma')
        fig.colorbar(im,ax=ax,label='Response score')
        ax.set(title=f'{axis} strip scan',xlabel=f'SLM {axis} strip centre',ylabel='Lenslets (sorted for display)')
    fig.tight_layout();fig.savefig(root/'strip_responses.png',dpi=160);plt.close(fig)


def analyse(root):
    settings=json.loads((root/'scan.json').read_text());args=settings['args']
    if not settings['complete']:raise ValueError('Scan incomplete; retain compact data but do not certify registration')
    with np.load(root/'reference.npz') as f:centres=f['centres_xy'];image=f['mean']
    with np.load(root/'responses.npz') as f:scores=f['scores']
    records=settings['records'];boxes=np.array([r['box'] for r in records])
    plot_strip_responses(root,records,scores,args['threshold'])
    uv,unc,group,hits=strip_correspondences(centres,records,scores,args['threshold'])
    measured=np.isfinite(uv).all(1)
    repeat=repeat_quality(records,scores,args['threshold'])
    if measured.sum()<6:raise ValueError('Fewer than six lenslets with bracketed strip responses on both axes; inspect strip_responses.png, signal and scan bounds')
    t=fit_affine(centres[measured],uv[measured],unc[measured]);pred=transform(centres,t)
    residual=np.linalg.norm(pred-uv,axis=1)
    norm=np.max(abs(pred-uv)/unc,axis=1)
    inlier=measured&(norm<=1.5)
    if inlier.sum()>=6:t=fit_affine(centres[inlier],uv[inlier],unc[inlier]);pred=transform(centres,t)
    residual=np.linalg.norm(pred-uv,axis=1)
    inlier=measured&(np.max(abs(pred-uv)/unc,axis=1)<=1.5)
    if inlier.sum()<6:raise ValueError('Fewer than six spatially consistent correspondences')
    # Hold out whole response regions, not random lenslets sharing the same quantised coordinate.
    groups=np.unique(group[measured]);cv=[]
    for held in groups:
        train=measured&(group!=held);test=measured&(group==held)
        if train.sum()<6:continue
        try:q=fit_affine(centres[train],uv[train],unc[train])
        except ValueError:continue
        cv.extend(np.max(abs(transform(centres[test],q)-uv[test])/unc[test],axis=1).tolist())
    a=t[:,:2];sv=np.linalg.svd(a,compute_uv=False)
    if sv[-1]<=1e-9:raise ValueError('Singular registration; no usable inverse')
    inverse=np.column_stack((np.linalg.inv(a),-np.linalg.solve(a,t[:,2])))
    ny,nx=settings['slm_shape_yx']
    inside=(pred[:,0]>=0)&(pred[:,0]<nx)&(pred[:,1]>=0)&(pred[:,1]<ny)
    # Voronoi cells are geometrical estimates, NOT measured optical influence boundaries.
    vor=Voronoi(centres);polygons=[None]*len(centres);areas=[]
    for i,reg in enumerate(vor.point_region):
        v=vor.regions[reg]
        if not v or -1 in v:continue
        camera_poly=vor.vertices[v];poly=transform(camera_poly,t)
        area=ConvexHull(poly).volume
        polygons[i]=poly.tolist();areas.append(area)
    area_med=float(np.median(areas)) if areas else 0
    # Reject abnormally large edge cells formed by missing neighbours.
    for i,p in enumerate(polygons):
        if p is not None and not .5*area_med<=ConvexHull(p).volume<=1.5*area_med:polygons[i]=None
    pitch=float(np.median(cKDTree(centres).query(centres,k=2)[0][:,1]))
    # Outer spot centres underestimate the illuminated support. Pad their convex hull
    # by half a lenslet pitch in camera space, then transform; edge estimate explicit.
    outer=centres[inlier][ConvexHull(centres[inlier]).vertices]
    angles=np.linspace(0,2*np.pi,32,endpoint=False)
    offsets=.5*pitch*np.column_stack((np.cos(angles),np.sin(angles)))
    expanded=(outer[:,None,:]+offsets[None,:,:]).reshape(-1,2)
    camera_hull=expanded[ConvexHull(expanded).vertices]
    hull=transform(camera_hull,t)
    slm_pitch=pitch*np.sqrt(abs(np.linalg.det(a)))
    half_patch=float(np.median(np.linalg.norm(unc[measured],axis=1)))
    coverage=float(ConvexHull(centres[inlier]).volume/ConvexHull(centres).volume) if inlier.sum()>=3 else 0
    cv95=float(np.percentile(cv,95)) if cv else None
    valid=bool(inlier.sum()>=12 and len(groups)>=6 and coverage>=.6 and cv95 is not None and cv95<=1.5 and
               sv[0]/sv[-1]<10 and inside[inlier].all() and half_patch/slm_pitch<=2 and repeat['passed'])
    lenslets=[]
    for i,xy in enumerate(centres):
        lenslets.append(dict(id=i,camera_xy=xy.tolist(),slm_xy_estimated=pred[i].tolist(),
            measured_at_finest_grid=bool(measured[i]),measured_on_both_strip_axes=bool(measured[i]),fit_inlier=bool(inlier[i]),response_patch_ids=hits[i],
            measured_region_centre_slm=uv[i].tolist() if np.isfinite(uv[i]).all() else None,
            region_half_extent_slm=unc[i].tolist() if np.isfinite(unc[i]).all() else None,
            estimated_voronoi_group_slm=polygons[i]))
    config=dict(schema='slm-shwfs-patch-registration-1',method='overlapping-strips-v2',repeat_checks=repeat,valid=valid,synthetic=settings.get('synthetic',False),
        camera=settings['camera'],slm=dict(nx=nx,ny=ny,baseline=128,actuator_definition='one native SLM pixel'),
        coordinates='xy column vectors; camera coords relative to saved ROI; SLM origin upper left; both y down',
        camera_to_slm_affine=t.tolist(),slm_to_camera_affine=inverse.tolist(),
        pupil_slm_polygon_estimated=hull.tolist(),pupil_camera_polygon_estimated=camera_hull.tolist(),
        pupil_boundary_method='Convex hull of inlier lenslet centres padded by half median lenslet pitch; estimated edge uncertainty ~one lenslet',
        frequency_conversion=dict(formula='cycles_across_pupil = max(P @ k) - min(P @ k)',
            P='pupil_slm_polygon_estimated',k='[fx, fy] in cycles per native SLM pixel',
            x_extent_actuators=float(np.ptp(hull[:,0])),y_extent_actuators=float(np.ptp(hull[:,1])),
            camera_frequency_formula='k_camera = camera_to_slm_affine[:, :2].T @ k_slm'),
        quality=dict(measured_lenslets=int(measured.sum()),inliers=int(inlier.sum()),response_regions=len(groups),
            camera_hull_coverage=coverage,regional_residual_rms_actuators=float(np.sqrt(np.mean(residual[inlier]**2))),
            held_out_region_95percent_half_extent_units=cv95,median_region_half_diagonal_actuators=half_patch,
            approximate_actuators_per_lenslet_pitch=slm_pitch,region_half_diagonal_in_lenslet_pitches=half_patch/slm_pitch,
            affine_condition_number=float(sv[0]/sv[-1]),median_interior_group_area_actuators2=area_med,
            lenslet_scale_resolved=bool(half_patch/slm_pitch<.5)),lenslets=lenslets,
        limitations=['Affine assumption; not an individual actuator influence-function measurement.',
            'Hard patch edges and full-range commands can cause nonlocal diffraction; inspect response maps.',
            'Main strip scores are four-state noise/drift diagnostics; selected positions are independently repeated. Scores are not calibrated detection probabilities.',
            'Region extents quantify localisation resolution, not statistical confidence intervals.',
            'Pupil support and Voronoi groups are inferred; edge lenslets and unilluminated holes are uncertain.',
            'valid means geometric checks passed; inspect plots before using for an interaction matrix.'])
    save_json(root/'registration.json',config)
    fig,axs=plt.subplots(1,3,figsize=(17,5))
    axs[0].imshow(image,origin='upper',vmax=np.percentile(image,99.8));axs[0].scatter(*centres[inlier].T,s=5,c=pred[inlier,0],cmap='viridis')
    axs[0].set(title='Lenslets coloured by inferred SLM x',xlabel='Camera x',ylabel='Camera y')
    axs[1].scatter(*uv[measured].T,s=10,label='Measured regional centres');axs[1].scatter(*pred[inlier].T,s=5,label='Affine lenslet positions')
    for i in np.flatnonzero(inlier):axs[1].plot([uv[i,0],pred[i,0]],[uv[i,1],pred[i,1]],color='grey',lw=.3)
    axs[1].add_patch(Polygon(hull,fill=False,color='r',label='Estimated pupil support'))
    axs[1].set(xlim=(0,nx),ylim=(ny,0),aspect='equal',xlabel='SLM x',ylabel='SLM y',title='Regional correspondence and affine fit');axs[1].legend(fontsize=7)
    for i,p in enumerate(polygons):
        if p is not None and inlier[i]:axs[2].add_patch(Polygon(p,fill=False,color='tab:blue',lw=.5))
    axs[2].scatter(*pred[~inlier].T,s=5,c='red');axs[2].set(xlim=(0,nx),ylim=(ny,0),aspect='equal',xlabel='SLM x',ylabel='SLM y',title='Inferred contiguous lenslet groups; red = unverified')
    fig.suptitle(f'Geometric checks: {"PASS" if valid else "FAIL"}; region half-diagonal ≈ {half_patch/slm_pitch:.2f} lenslet pitches')
    fig.tight_layout();fig.savefig(root/'registration.png',dpi=170);plt.close(fig)
    print(f'Registration: {root}/registration.json; valid={valid}; {inlier.sum()} inliers; resolution ~{half_patch/slm_pitch:.2f} lenslet pitches')
    selected=np.unique([h for i in np.flatnonzero(inlier) for h in hits[i]]);selected=selected[np.linspace(0,len(selected)-1,min(6,len(selected))).astype(int)]
    fig,axs=plt.subplots(2,len(selected),figsize=(3.3*len(selected),6),squeeze=False)
    for j,k in enumerate(selected):
        x0,y0,x1,y1=boxes[k]
        axs[0,j].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='tab:blue'))
        axs[0,j].plot(*np.vstack((hull,hull[0])).T,color='r',lw=1)
        axs[0,j].set(xlim=(0,nx),ylim=(ny,0),aspect='equal',title=f'SLM strip {k}')
        axs[1,j].imshow(image,origin='upper',cmap='gray',vmax=np.percentile(image,99.8))
        active=scores[k]>=args['threshold']
        axs[1,j].scatter(*centres[active].T,s=12,facecolors='none',edgecolors='red')
        axs[1,j].set(title=f'{active.sum()} responding lenslets')
    fig.tight_layout();fig.savefig(root/'patch_associations.png',dpi=150);plt.close(fig)
    return config


def repository_data_root(script_path=None,cwd=None):
    """Prefer the script's repository; support .git files used by Git worktrees."""
    starts=[Path(script_path or __file__).resolve().parent,Path(cwd or Path.cwd()).resolve()]
    for start in starts:
        for parent in (start,*start.parents):
            if (parent/'.git').exists():return parent/'data'
    raise ValueError('Cannot locate a Git repository root. Run from your repository or supply --output explicitly.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,help='Override default: <Git repository>/data/slm_registration_v2_<timestamp>')
    p.add_argument('--frequency-config',type=Path,help='Convert desired cycles/pupil using an existing registration; no hardware')
    p.add_argument('--cycles-per-pupil',type=float,default=4)
    p.add_argument('--orientation',type=float,default=0,help='Frequency vector angle in SLM xy degrees, y down')
    p.add_argument('--analyse',type=Path,help='Reanalyse compact saved measurements without hardware')
    p.add_argument('--save-data','--save_data',action='store_true',help='Also save full raw stacks and commands; compact scan/config always saved')
    p.add_argument('--screen',default='1');p.add_argument('--frames-per-probe',type=int,default=8)
    p.add_argument('--positions',type=int,default=32,help='Strip centres per axis across coarse illuminated bounds, plus guard positions')
    p.add_argument('--width-steps',type=float,default=2.5,help='Strip width / position spacing; default 2.5 gives overlapping measurements')
    p.add_argument('--repeat-positions',type=int,default=3,help='Responsive strip positions to repeat per axis, at least 2')
    p.add_argument('--amplitude',type=float,default=127);p.add_argument('--cycles',type=float,default=8)
    p.add_argument('--threshold',type=float,default=5)
    p.add_argument('--settle',type=float,default=.3);p.add_argument('--flush',type=int,default=3)
    p.add_argument('--half-window',type=int,default=10);p.add_argument('--min-distance',type=int,default=16)
    p.add_argument('--detection-fraction',type=float,default=.03);p.add_argument('--demo',action='store_true')
    a=p.parse_args()
    if a.frequency_config:
        config=json.loads(a.frequency_config.read_text())
        if not config['valid']:p.error('Registration did not pass quality checks')
        if not np.isfinite([a.cycles_per_pupil,a.orientation]).all() or a.cycles_per_pupil<=0:p.error('Invalid frequency')
        direction=np.array([np.cos(np.deg2rad(a.orientation)),np.sin(np.deg2rad(a.orientation))])
        span=float(np.ptp(np.array(config['pupil_slm_polygon_estimated'])@direction))
        f=a.cycles_per_pupil/span
        if np.max(abs(f*direction))>=.5:p.error('Frequency exceeds SLM pixel Nyquist')
        print(f'{a.cycles_per_pupil:g} cycles/pupil at {a.orientation:g} degrees: --frequencies {f:.10g} --orientations {a.orientation:g}')
        print(f'Projected illuminated pupil width: {span:.3f} native SLM pixels; geometric conversion only, not a WFS bandwidth guarantee.')
        return
    if a.analyse:analyse(a.analyse);return
    if a.positions<8 or a.positions>256 or a.repeat_positions<2:p.error('Need 8..256 positions and >=2 repeated positions/axis')
    if not np.isfinite(a.width_steps) or not 2<=a.width_steps<=4:p.error('--width-steps must be 2..4')
    if not np.isfinite([a.amplitude,a.cycles,a.threshold,a.settle,a.detection_fraction]).all():p.error('Nonfinite option')
    if not 0<a.amplitude<=127 or a.cycles<=0 or a.threshold<=0 or a.settle<0 or not 0<a.detection_fraction<1:p.error('Invalid amplitudes or thresholds')
    if a.frames_per_probe<4 or a.flush<2 or a.half_window<2 or a.min_distance<2:p.error('Need >=4 frames, >=2 flushes, window/distance >=2')
    if a.output is None:
        try:a.output=repository_data_root()/('slm_registration_v2_'+time.strftime('%Y%m%d_%H%M%S'))
        except ValueError as exc:p.error(str(exc))
    print(f'Output directory: {a.output.resolve()}',flush=True)
    scan(a)

if __name__=='__main__':main()


# #!/usr/bin/env python3
# """Fast strip SLM registration (v2), using the working response-check interfaces.
# Keep slm_response_check.py beside this file. No camera settings are changed.
# python register_slm_v2.py --output registration_01
# python register_slm_v2.py --analyse registration_01
# python register_slm_v2.py --demo --output registration_demo
# """
# from pathlib import Path
# import argparse
# import json
# import time
# import numpy as np
# from scipy.ndimage import gaussian_filter, maximum_filter
# from scipy.spatial import cKDTree, ConvexHull, Voronoi
# from scipy.optimize import least_squares
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, Polygon
# import slm_response_check as base


# def detect(image, distance, fraction, half):
#     im=gaussian_filter(image.astype(float),1.2)
#     floor=np.median(im);threshold=floor+max(8*1.4826*np.median(abs(im-floor)),fraction*(im.max()-floor))
#     y,x=np.nonzero((im==maximum_filter(im,size=2*distance+1))&(im>threshold))
#     ok=(x>=half)&(y>=half)&(x<im.shape[1]-half)&(y<im.shape[0]-half)
#     centres=np.column_stack((x[ok],y[ok]))
#     if len(centres)<6:raise ValueError('Fewer than six spots; inspect detection settings')
#     if cKDTree(centres).query(centres,k=2)[0][:,1].min()<=2*half:
#         raise ValueError('Spot windows overlap: reduce --half-window or increase --min-distance')
#     return centres


# def command(shape,box,axis,phase,sign,amplitude,cycles):
#     # Same carrier pitch at every hierarchy level; quadrature covers gradient nodes.
#     x0,y0,x1,y1=box
#     u=np.arange(x0,x1)[None,:] if axis=='x' else np.arange(y0,y1)[:,None]
#     length=shape[1] if axis=='x' else shape[0]
#     wave=np.sin(2*np.pi*cycles*u/length+np.deg2rad(phase))
#     result=np.full(shape,128,np.uint8)
#     result[y0:y1,x0:x1]=(128+sign*np.rint(amplitude*wave)).astype(np.uint8)
#     return result


# def spot_scores(result,centres,half):
#     yy,xx=np.mgrid[-half:half+1,-half:half+1]
#     score=result['score'][centres[:,1,None,None]+yy,centres[:,0,None,None]+xx]
#     bad=result['bad'][centres[:,1,None,None]+yy,centres[:,0,None,None]+xx].any(axis=(1,2))
#     # Third strongest repeatable pixel: no reliance on a single noisy pixel.
#     values=np.partition(score.reshape(len(centres),-1),-3,axis=1)[:,-3]
#     values[bad]=0
#     return values,bad


# def split(box):
#     x0,y0,x1,y1=box;xm=(x0+x1)//2;ym=(y0+y1)//2
#     return [(x0,y0,xm,ym),(xm,y0,x1,ym),(x0,ym,xm,y1),(xm,ym,x1,y1)]


# def plot_windows(image,centres,half,path):
#     fig,ax=plt.subplots(figsize=(9,8));ax.imshow(image,origin='upper',vmax=np.percentile(image,99.8))
#     for x,y in centres:ax.add_patch(Rectangle((x-half,y-half),2*half+1,2*half+1,fill=False,color='r',lw=.4))
#     ax.set(title=f'{len(centres)} fixed lenslet windows',xlabel='Cropped camera x [pixel]',ylabel='Cropped camera y [pixel]')
#     fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


# def fast_response(states):
#     """Four-state diagnostic, not a repeatability claim. Static drift is penalised."""
#     b0,b1=states['b0'],states['b1'];p,m=states['p0'],states['m0']
#     bp,vp=base.interpolate_static(b0,b1,p);bm,vm=base.interpolate_static(b0,b1,m)
#     delta=np.array([p['mean']-bp,m['mean']-bm,p['mean']-m['mean']])
#     var=np.array([p['varmean']+vp,m['varmean']+vm,p['varmean']+m['varmean']])
#     drift=abs(b1['mean']-b0['mean'])
#     bad=np.logical_or.reduce([s['saturated'] for s in states.values()])
#     score=np.max(abs(delta)/np.sqrt(var+drift[None]**2+base.MIN_RESPONSE_ADU**2),axis=0)
#     score[bad]=0
#     return dict(delta=delta,score=score,mask=score>=base.RESPONSE_THRESHOLD,
#                 drift=drift,bad=bad,static=(b0['mean']+b1['mean'])/2)


# def trial(bench,args,root,centres,tag,box,axes,repeat=False,diagnostic=False):
#     values=np.zeros(len(centres));bad=np.zeros(len(centres),bool);started=time.monotonic()
#     names=('b0','p0','m0','b1','m1','p1','b2') if repeat else ('b0','p0','m0','b1')
#     empty=box[0]>=box[2] or box[1]>=box[3]
#     if empty:return values,bad,0.,0
#     for axis in axes:
#         for phase in (0.,90.):
#             states={}
#             cmds={'b':bench.flat,'p':command(bench.shape,box,axis,phase,1,args.amplitude,args.cycles),
#                   'm':command(bench.shape,box,axis,phase,-1,args.amplitude,args.cycles)}
#             for name in names:
#                 states[name]=base.capture(bench,cmds[name[0]],args.frames_per_probe,
#                                          root/f'{tag}_{axis}_{phase:g}_{name}',args.save_data)
#             result=base.response(states) if repeat else fast_response(states)
#             v,b=spot_scores(result,centres,args.half_window);values=np.maximum(values,v);bad|=b
#             if diagnostic:base.detail_plot(result,cmds['p'],root/f'{tag}_{axis}_{phase:g}.png',tag,False)
#     values[bad]=0
#     return values,bad,time.monotonic()-started,len(axes)*2*len(names)


# def illuminated_bounds(shape,quadrants,scores,full,threshold):
#     ny,nx=shape;active=(scores>=threshold).any(axis=1)
#     # If quadrants fail to cover full-SLM detections, do not silently crop those lenslets away.
#     uncovered=(full>=threshold)&~(scores>=threshold).any(axis=0)
#     if not active.any() or uncovered.any():return (0,0,nx,ny),True
#     b=np.array(quadrants)[active];lo=b[:,:2].min(0);hi=b[:,2:].max(0)
#     pad=.1*(hi-lo)
#     return tuple(np.r_[np.maximum(0,np.floor(lo-pad)),np.minimum([nx,ny],np.ceil(hi+pad))].astype(int)),False


# def strip_plan(shape,bounds,positions,width_steps):
#     """Two overlapping 1D scans plus zero-support guards, with full transverse extent.
#     The carrier is transverse to the strip: moving its window does not move its sine nodes.
#     """
#     ny,nx=shape;records=[]
#     for axis in ('x','y'):
#         dim=0 if axis=='x' else 1;lo,hi=bounds[dim],bounds[dim+2]
#         step=(hi-lo)/(positions-1);width=width_steps*step
#         count=int(np.ceil(width_steps/2))+1
#         for c in lo+np.arange(-count,positions+count)*step:
#             start=max(0,int(np.floor(c-width/2)));end=min(shape[1-dim],int(np.ceil(c+width/2)))
#             start=min(shape[1-dim],start);end=max(0,end)
#             box=(start,0,end,ny) if axis=='x' else (0,start,nx,end)
#             records.append(dict(stage='strip',axis=axis,centre=float(c),step=float(step),width=float(width),
#                                 box=list(box),carrier_axes=['y' if axis=='x' else 'x']))
#     return records


# def scan(args):
#     root=args.output;root.mkdir(parents=True,exist_ok=False);bench=None
#     settings=dict(schema='slm-strip-scan-2',complete=False,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},records=[])
#     scores=[];bad=[];durations=[]
#     def checkpoint():
#         with (root/'responses.pending.npz').open('wb') as f:
#             np.savez_compressed(f,scores=np.array(scores),saturated=np.array(bad))
#         (root/'responses.pending.npz').replace(root/'responses.npz')
#         base.save_json(root/'scan.json',settings)
#     def record(spec,repeat=False,diagnostic=False):
#         tag=f"{spec['stage']}_{len(scores):04d}"
#         v,b,elapsed,nstates=trial(bench,args,root,centres,tag,spec['box'],spec['carrier_axes'],repeat,diagnostic)
#         row=dict(spec,repeat=repeat,elapsed_s=elapsed,display_states=nstates,responding_lenslets=int((v>=args.threshold).sum()))
#         settings['records'].append(row);scores.append(v);bad.append(b)
#         if nstates:durations.append(elapsed/nstates)
#         checkpoint()
#         print(f'{tag}: {row["responding_lenslets"]} lenslets; {elapsed:.1f} s',flush=True)
#         return len(scores)-1
#     try:
#         bench=base.SyntheticBench(args) if args.demo else base.Bench(args)
#         settings.update(camera=bench.cam.original,slm_shape_yx=list(bench.shape),synthetic=args.demo)
#         base.save_json(root/'scan.json',settings)
#         if not args.demo:input('Laser ON, camera fixed and unsaturated. Enter to detect spots: ')
#         ref=base.capture(bench,bench.flat,args.frames_per_probe,root/'initial',args.save_data)
#         centres=detect(ref['mean'],args.min_distance,args.detection_fraction,args.half_window)
#         np.savez_compressed(root/'reference.npz',mean=ref['mean'],centres_xy=centres)
#         plot_windows(ref['mean'],centres,args.half_window,root/'lenslet_windows.png')
#         print(f'Inspect {root}/lenslet_windows.png',flush=True)
#         if not args.demo and input('Are the lenslet windows correct? [y/N]: ').strip().lower()!='y':return
#         ny,nx=bench.shape
#         full=record(dict(stage='full',box=[0,0,nx,ny],carrier_axes=['x','y']),True,True)
#         if (scores[full]>=args.threshold).sum()<6:raise ValueError('Insufficient repeatable full-SLM response')
#         if not args.demo and input('Inspect full_* plots. Continue with quadrants and strip scans? [y/N]: ').strip().lower()!='y':return
#         quadrants=base.regions(bench.shape,2);qids=[]
#         for box in quadrants:qids.append(record(dict(stage='coarse',box=list(box),carrier_axes=['x','y'])))
#         bounds,fallback=illuminated_bounds(bench.shape,quadrants,np.array(scores)[qids],scores[full],args.threshold)
#         settings.update(scan_bounds=list(map(int,bounds)),coarse_fallback_to_full_slm=bool(fallback))
#         plan=strip_plan(bench.shape,bounds,args.positions,args.width_steps)
#         nstates=sum(8 for r in plan if r['box'][0]<r['box'][2] and r['box'][1]<r['box'][3])+28*args.repeat_positions
#         estimate=np.median(durations)*nstates
#         settings['estimated_remaining_seconds']=float(estimate);checkpoint()
#         print(f'{len(plan)} strip positions including guards; about {estimate/60:.1f} minutes remaining, based on measured acquisition speed. Bounds={bounds}; full-SLM fallback={fallback}.',flush=True)
#         if not args.demo and input('Proceed with this strip scan? [y/N]: ').strip().lower()!='y':return
#         stripids=[]
#         for j,spec in enumerate(plan):
#             stripids.append(record(spec))
#             remaining=len(plan)-j-1
#             print(f'  {remaining} strip positions remaining (~{remaining*8*np.median(durations)/60:.1f} min, plus repeat checks)',flush=True)
#         # Repeat responsive positions distributed across each measured coordinate range.
#         for axis in ('x','y'):
#             ids=[i for i in stripids if settings['records'][i]['axis']==axis and (scores[i]>=args.threshold).any()]
#             chosen=np.unique(np.linspace(0,len(ids)-1,min(args.repeat_positions,len(ids))).astype(int)) if ids else []
#             for j in chosen:
#                 i=ids[j];spec=dict(settings['records'][i]);spec.update(stage='repeat',original_record=i)
#                 record(spec,True)
#         settings['complete']=True;checkpoint()
#     except BaseException as exc:
#         settings['error']=repr(exc);base.save_json(root/'scan.json',settings);raise
#     finally:
#         if bench is not None:bench.close()
#     analyse(root)


# def locate_axis(records,scores,axis,threshold):
#     ids=[i for i,r in enumerate(records) if r['stage']=='strip' and r['axis']==axis]
#     ids=sorted(ids,key=lambda i:records[i]['centre'])
#     c=np.array([records[i]['centre'] for i in ids]);v=scores[ids]
#     n=scores.shape[1];position=np.full(n,np.nan);extent=np.full(n,np.nan);hits=[[] for _ in range(n)]
#     if len(ids)<5:return position,extent,hits,ids
#     step=float(np.median(np.diff(c)));width=records[ids[0]]['width']
#     for k in range(n):
#         active=np.flatnonzero(v[:,k]>=threshold)
#         # Need a bracket on both sides, at least two overlapping hits and a single compact lobe.
#         if len(active)<2 or active[0]==0 or active[-1]==len(ids)-1:continue
#         if np.any(np.diff(active)>2) or c[active[-1]]-c[active[0]]>width+2*step:continue
#         # Midpoint of threshold-crossing brackets; no signal-strength/distance assumption.
#         lo=(c[active[0]-1]+c[active[0]])/2;hi=(c[active[-1]]+c[active[-1]+1])/2
#         position[k]=(lo+hi)/2
#         # Conservative localisation scale includes sampling and finite strip width.
#         extent[k]=max(step,width/2)
#         hits[k]=[ids[i] for i in active]
#     return position,extent,hits,ids


# def strip_correspondences(centres,records,scores,threshold):
#     x,dx,hx,ix=locate_axis(records,scores,'x',threshold)
#     y,dy,hy,iy=locate_axis(records,scores,'y',threshold)
#     uv=np.column_stack((x,y));unc=np.column_stack((dx,dy));hits=[a+b for a,b in zip(hx,hy)]
#     # Spatial bins group correlated strip estimates for region-held-out validation.
#     valid=np.isfinite(uv).all(1);group=np.full(len(centres),-1,int)
#     if valid.any():
#         bins=np.floor((uv[valid]-np.nanmin(uv[valid],axis=0))/np.nanmedian(unc[valid],axis=0)/2).astype(int)
#         _,group[valid]=np.unique(bins,axis=0,return_inverse=True)
#     return uv,unc,group,hits


# def repeat_quality(records,scores,threshold):
#     values=[];counts={axis:0 for axis in ('x','y')}
#     for i,r in enumerate(records):
#         if r['stage']!='repeat':continue
#         before=scores[r['original_record']]>=threshold;after=scores[i]>=threshold
#         union=before|after
#         agreement=float((before&after).sum()/union.sum()) if union.any() else 0.
#         values.append(dict(record=i,axis=r['axis'],jaccard=agreement));counts[r['axis']]+=1
#     good=all(v>=2 for v in counts.values()) and bool(values) and min(v['jaccard'] for v in values)>=.5
#     return dict(passed=bool(good),checks=values,checks_per_axis=counts)


# def fit_affine(xy,uv,unc):
#     origin=xy.mean(0);scale=np.std(xy,axis=0)
#     if np.any(scale<1):raise ValueError('Insufficient two-dimensional lenslet coverage')
#     design=np.column_stack(((xy-origin)/scale,np.ones(len(xy))))
#     if np.linalg.matrix_rank(design)<3:raise ValueError('Degenerate lenslet positions')
#     initial=np.linalg.lstsq(design,uv,rcond=None)[0]
#     fit=least_squares(lambda p:((design@p.reshape(3,2)-uv)/unc).ravel(),initial.ravel(),loss='soft_l1')
#     b=fit.x.reshape(3,2);a=b[:2].T/scale[None,:];offset=b[2]-a@origin
#     return np.column_stack((a,offset))


# def transform(xy,t):return np.asarray(xy)@t[:,:2].T+t[:,2]


# def plot_strip_responses(root,records,scores,threshold):
#     fig,axs=plt.subplots(1,2,figsize=(13,6))
#     for ax,axis in zip(axs,('x','y')):
#         ids=sorted([i for i,r in enumerate(records) if r['stage']=='strip' and r['axis']==axis],key=lambda i:records[i]['centre'])
#         if not ids:continue
#         positions=np.array([records[i]['centre'] for i in ids]);v=scores[ids].T
#         # Sort display only by strongest response location; lenslet IDs in data remain fixed.
#         order=np.argsort(np.argmax(v,axis=1))
#         im=ax.imshow(v[order],origin='lower',aspect='auto',extent=[positions[0],positions[-1],0,len(v)],vmin=0,vmax=2*threshold,cmap='magma')
#         fig.colorbar(im,ax=ax,label='Response score')
#         ax.set(title=f'{axis} strip scan',xlabel=f'SLM {axis} strip centre',ylabel='Lenslets (sorted for display)')
#     fig.tight_layout();fig.savefig(root/'strip_responses.png',dpi=160);plt.close(fig)


# def analyse(root):
#     settings=json.loads((root/'scan.json').read_text());args=settings['args']
#     if not settings['complete']:raise ValueError('Scan incomplete; retain compact data but do not certify registration')
#     with np.load(root/'reference.npz') as f:centres=f['centres_xy'];image=f['mean']
#     with np.load(root/'responses.npz') as f:scores=f['scores']
#     records=settings['records'];boxes=np.array([r['box'] for r in records])
#     plot_strip_responses(root,records,scores,args['threshold'])
#     uv,unc,group,hits=strip_correspondences(centres,records,scores,args['threshold'])
#     measured=np.isfinite(uv).all(1)
#     repeat=repeat_quality(records,scores,args['threshold'])
#     if measured.sum()<6:raise ValueError('Fewer than six lenslets with bracketed strip responses on both axes; inspect strip_responses.png, signal and scan bounds')
#     t=fit_affine(centres[measured],uv[measured],unc[measured]);pred=transform(centres,t)
#     residual=np.linalg.norm(pred-uv,axis=1)
#     norm=np.max(abs(pred-uv)/unc,axis=1)
#     inlier=measured&(norm<=1.5)
#     if inlier.sum()>=6:t=fit_affine(centres[inlier],uv[inlier],unc[inlier]);pred=transform(centres,t)
#     residual=np.linalg.norm(pred-uv,axis=1)
#     inlier=measured&(np.max(abs(pred-uv)/unc,axis=1)<=1.5)
#     if inlier.sum()<6:raise ValueError('Fewer than six spatially consistent correspondences')
#     # Hold out whole response regions, not random lenslets sharing the same quantised coordinate.
#     groups=np.unique(group[measured]);cv=[]
#     for held in groups:
#         train=measured&(group!=held);test=measured&(group==held)
#         if train.sum()<6:continue
#         try:q=fit_affine(centres[train],uv[train],unc[train])
#         except ValueError:continue
#         cv.extend(np.max(abs(transform(centres[test],q)-uv[test])/unc[test],axis=1).tolist())
#     a=t[:,:2];sv=np.linalg.svd(a,compute_uv=False)
#     if sv[-1]<=1e-9:raise ValueError('Singular registration; no usable inverse')
#     inverse=np.column_stack((np.linalg.inv(a),-np.linalg.solve(a,t[:,2])))
#     ny,nx=settings['slm_shape_yx']
#     inside=(pred[:,0]>=0)&(pred[:,0]<nx)&(pred[:,1]>=0)&(pred[:,1]<ny)
#     # Voronoi cells are geometrical estimates, NOT measured optical influence boundaries.
#     vor=Voronoi(centres);polygons=[None]*len(centres);areas=[]
#     for i,reg in enumerate(vor.point_region):
#         v=vor.regions[reg]
#         if not v or -1 in v:continue
#         camera_poly=vor.vertices[v];poly=transform(camera_poly,t)
#         area=ConvexHull(poly).volume
#         polygons[i]=poly.tolist();areas.append(area)
#     area_med=float(np.median(areas)) if areas else 0
#     # Reject abnormally large edge cells formed by missing neighbours.
#     for i,p in enumerate(polygons):
#         if p is not None and not .5*area_med<=ConvexHull(p).volume<=1.5*area_med:polygons[i]=None
#     pitch=float(np.median(cKDTree(centres).query(centres,k=2)[0][:,1]))
#     # Outer spot centres underestimate the illuminated support. Pad their convex hull
#     # by half a lenslet pitch in camera space, then transform; edge estimate explicit.
#     outer=centres[inlier][ConvexHull(centres[inlier]).vertices]
#     angles=np.linspace(0,2*np.pi,32,endpoint=False)
#     offsets=.5*pitch*np.column_stack((np.cos(angles),np.sin(angles)))
#     expanded=(outer[:,None,:]+offsets[None,:,:]).reshape(-1,2)
#     camera_hull=expanded[ConvexHull(expanded).vertices]
#     hull=transform(camera_hull,t)
#     slm_pitch=pitch*np.sqrt(abs(np.linalg.det(a)))
#     half_patch=float(np.median(np.linalg.norm(unc[measured],axis=1)))
#     coverage=float(ConvexHull(centres[inlier]).volume/ConvexHull(centres).volume) if inlier.sum()>=3 else 0
#     cv95=float(np.percentile(cv,95)) if cv else None
#     valid=bool(inlier.sum()>=12 and len(groups)>=6 and coverage>=.6 and cv95 is not None and cv95<=1.5 and
#                sv[0]/sv[-1]<10 and inside[inlier].all() and half_patch/slm_pitch<=2 and repeat['passed'])
#     lenslets=[]
#     for i,xy in enumerate(centres):
#         lenslets.append(dict(id=i,camera_xy=xy.tolist(),slm_xy_estimated=pred[i].tolist(),
#             measured_at_finest_grid=bool(measured[i]),measured_on_both_strip_axes=bool(measured[i]),fit_inlier=bool(inlier[i]),response_patch_ids=hits[i],
#             measured_region_centre_slm=uv[i].tolist() if np.isfinite(uv[i]).all() else None,
#             region_half_extent_slm=unc[i].tolist() if np.isfinite(unc[i]).all() else None,
#             estimated_voronoi_group_slm=polygons[i]))
#     config=dict(schema='slm-shwfs-patch-registration-1',method='overlapping-strips-v2',repeat_checks=repeat,valid=valid,synthetic=settings.get('synthetic',False),
#         camera=settings['camera'],slm=dict(nx=nx,ny=ny,baseline=128,actuator_definition='one native SLM pixel'),
#         coordinates='xy column vectors; camera coords relative to saved ROI; SLM origin upper left; both y down',
#         camera_to_slm_affine=t.tolist(),slm_to_camera_affine=inverse.tolist(),
#         pupil_slm_polygon_estimated=hull.tolist(),pupil_camera_polygon_estimated=camera_hull.tolist(),
#         pupil_boundary_method='Convex hull of inlier lenslet centres padded by half median lenslet pitch; estimated edge uncertainty ~one lenslet',
#         frequency_conversion=dict(formula='cycles_across_pupil = max(P @ k) - min(P @ k)',
#             P='pupil_slm_polygon_estimated',k='[fx, fy] in cycles per native SLM pixel',
#             x_extent_actuators=float(np.ptp(hull[:,0])),y_extent_actuators=float(np.ptp(hull[:,1])),
#             camera_frequency_formula='k_camera = camera_to_slm_affine[:, :2].T @ k_slm'),
#         quality=dict(measured_lenslets=int(measured.sum()),inliers=int(inlier.sum()),response_regions=len(groups),
#             camera_hull_coverage=coverage,regional_residual_rms_actuators=float(np.sqrt(np.mean(residual[inlier]**2))),
#             held_out_region_95percent_half_extent_units=cv95,median_region_half_diagonal_actuators=half_patch,
#             approximate_actuators_per_lenslet_pitch=slm_pitch,region_half_diagonal_in_lenslet_pitches=half_patch/slm_pitch,
#             affine_condition_number=float(sv[0]/sv[-1]),median_interior_group_area_actuators2=area_med,
#             lenslet_scale_resolved=bool(half_patch/slm_pitch<.5)),lenslets=lenslets,
#         limitations=['Affine assumption; not an individual actuator influence-function measurement.',
#             'Hard patch edges and full-range commands can cause nonlocal diffraction; inspect response maps.',
#             'Main strip scores are four-state noise/drift diagnostics; selected positions are independently repeated. Scores are not calibrated detection probabilities.',
#             'Region extents quantify localisation resolution, not statistical confidence intervals.',
#             'Pupil support and Voronoi groups are inferred; edge lenslets and unilluminated holes are uncertain.',
#             'valid means geometric checks passed; inspect plots before using for an interaction matrix.'])
#     base.save_json(root/'registration.json',config)
#     fig,axs=plt.subplots(1,3,figsize=(17,5))
#     axs[0].imshow(image,origin='upper',vmax=np.percentile(image,99.8));axs[0].scatter(*centres[inlier].T,s=5,c=pred[inlier,0],cmap='viridis')
#     axs[0].set(title='Lenslets coloured by inferred SLM x',xlabel='Camera x',ylabel='Camera y')
#     axs[1].scatter(*uv[measured].T,s=10,label='Measured regional centres');axs[1].scatter(*pred[inlier].T,s=5,label='Affine lenslet positions')
#     for i in np.flatnonzero(inlier):axs[1].plot([uv[i,0],pred[i,0]],[uv[i,1],pred[i,1]],color='grey',lw=.3)
#     axs[1].add_patch(Polygon(hull,fill=False,color='r',label='Estimated pupil support'))
#     axs[1].set(xlim=(0,nx),ylim=(ny,0),aspect='equal',xlabel='SLM x',ylabel='SLM y',title='Regional correspondence and affine fit');axs[1].legend(fontsize=7)
#     for i,p in enumerate(polygons):
#         if p is not None and inlier[i]:axs[2].add_patch(Polygon(p,fill=False,color='tab:blue',lw=.5))
#     axs[2].scatter(*pred[~inlier].T,s=5,c='red');axs[2].set(xlim=(0,nx),ylim=(ny,0),aspect='equal',xlabel='SLM x',ylabel='SLM y',title='Inferred contiguous lenslet groups; red = unverified')
#     fig.suptitle(f'Geometric checks: {"PASS" if valid else "FAIL"}; region half-diagonal ≈ {half_patch/slm_pitch:.2f} lenslet pitches')
#     fig.tight_layout();fig.savefig(root/'registration.png',dpi=170);plt.close(fig)
#     print(f'Registration: {root}/registration.json; valid={valid}; {inlier.sum()} inliers; resolution ~{half_patch/slm_pitch:.2f} lenslet pitches')
#     selected=np.unique([h for i in np.flatnonzero(inlier) for h in hits[i]]);selected=selected[np.linspace(0,len(selected)-1,min(6,len(selected))).astype(int)]
#     fig,axs=plt.subplots(2,len(selected),figsize=(3.3*len(selected),6),squeeze=False)
#     for j,k in enumerate(selected):
#         x0,y0,x1,y1=boxes[k]
#         axs[0,j].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='tab:blue'))
#         axs[0,j].plot(*np.vstack((hull,hull[0])).T,color='r',lw=1)
#         axs[0,j].set(xlim=(0,nx),ylim=(ny,0),aspect='equal',title=f'SLM strip {k}')
#         axs[1,j].imshow(image,origin='upper',cmap='gray',vmax=np.percentile(image,99.8))
#         active=scores[k]>=args['threshold']
#         axs[1,j].scatter(*centres[active].T,s=12,facecolors='none',edgecolors='red')
#         axs[1,j].set(title=f'{active.sum()} responding lenslets')
#     fig.tight_layout();fig.savefig(root/'patch_associations.png',dpi=150);plt.close(fig)
#     return config


# def repository_data_root(script_path=None,cwd=None):
#     """Prefer the script's repository; support .git files used by Git worktrees."""
#     starts=[Path(script_path or __file__).resolve().parent,Path(cwd or Path.cwd()).resolve()]
#     for start in starts:
#         for parent in (start,*start.parents):
#             if (parent/'.git').exists():return parent/'data'
#     raise ValueError('Cannot locate a Git repository root. Run from your repository or supply --output explicitly.')


# def main():
#     p=argparse.ArgumentParser(description=__doc__)
#     p.add_argument('--output',type=Path,help='Override default: <Git repository>/data/slm_registration_v2_<timestamp>')
#     p.add_argument('--frequency-config',type=Path,help='Convert desired cycles/pupil using an existing registration; no hardware')
#     p.add_argument('--cycles-per-pupil',type=float,default=4)
#     p.add_argument('--orientation',type=float,default=0,help='Frequency vector angle in SLM xy degrees, y down')
#     p.add_argument('--analyse',type=Path,help='Reanalyse compact saved measurements without hardware')
#     p.add_argument('--save-data','--save_data',action='store_true',help='Also save full raw stacks and commands; compact scan/config always saved')
#     p.add_argument('--screen',default='1');p.add_argument('--frames-per-probe',type=int,default=8)
#     p.add_argument('--positions',type=int,default=32,help='Strip centres per axis across coarse illuminated bounds, plus guard positions')
#     p.add_argument('--width-steps',type=float,default=2.5,help='Strip width / position spacing; default 2.5 gives overlapping measurements')
#     p.add_argument('--repeat-positions',type=int,default=3,help='Responsive strip positions to repeat per axis, at least 2')
#     p.add_argument('--amplitude',type=float,default=127);p.add_argument('--cycles',type=float,default=8)
#     p.add_argument('--threshold',type=float,default=5)
#     p.add_argument('--settle',type=float,default=.3);p.add_argument('--flush',type=int,default=3)
#     p.add_argument('--half-window',type=int,default=10);p.add_argument('--min-distance',type=int,default=16)
#     p.add_argument('--detection-fraction',type=float,default=.03);p.add_argument('--demo',action='store_true')
#     a=p.parse_args()
#     if a.frequency_config:
#         config=json.loads(a.frequency_config.read_text())
#         if not config['valid']:p.error('Registration did not pass quality checks')
#         if not np.isfinite([a.cycles_per_pupil,a.orientation]).all() or a.cycles_per_pupil<=0:p.error('Invalid frequency')
#         direction=np.array([np.cos(np.deg2rad(a.orientation)),np.sin(np.deg2rad(a.orientation))])
#         span=float(np.ptp(np.array(config['pupil_slm_polygon_estimated'])@direction))
#         f=a.cycles_per_pupil/span
#         if np.max(abs(f*direction))>=.5:p.error('Frequency exceeds SLM pixel Nyquist')
#         print(f'{a.cycles_per_pupil:g} cycles/pupil at {a.orientation:g} degrees: --frequencies {f:.10g} --orientations {a.orientation:g}')
#         print(f'Projected illuminated pupil width: {span:.3f} native SLM pixels; geometric conversion only, not a WFS bandwidth guarantee.')
#         return
#     if a.analyse:analyse(a.analyse);return
#     if a.positions<8 or a.positions>256 or a.repeat_positions<2:p.error('Need 8..256 positions and >=2 repeated positions/axis')
#     if not np.isfinite(a.width_steps) or not 2<=a.width_steps<=4:p.error('--width-steps must be 2..4')
#     if not np.isfinite([a.amplitude,a.cycles,a.threshold,a.settle,a.detection_fraction]).all():p.error('Nonfinite option')
#     if not 0<a.amplitude<=127 or a.cycles<=0 or a.threshold<=0 or a.settle<0 or not 0<a.detection_fraction<1:p.error('Invalid amplitudes or thresholds')
#     if a.frames_per_probe<4 or a.flush<2 or a.half_window<2 or a.min_distance<2:p.error('Need >=4 frames, >=2 flushes, window/distance >=2')
#     if a.output is None:
#         try:a.output=repository_data_root()/('slm_registration_v2_'+time.strftime('%Y%m%d_%H%M%S'))
#         except ValueError as exc:p.error(str(exc))
#     print(f'Output directory: {a.output.resolve()}',flush=True)
#     scan(a)

# if __name__=='__main__':main()
