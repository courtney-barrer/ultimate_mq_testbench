#!/usr/bin/env python3
"""Offline registered SLM Fourier/Zernike bases; no camera/SLM access.
Default outputs: <Git repository>/data/slm_basis_<timestamp>/.
Mode arrays are dimensionless, piston removed over the discrete registered mask,
orthonormal under mean(mode_i * mode_j) over the pupil. A command is 128 + round(amplitude * mode), flat outside.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import tempfile
from scipy.ndimage import distance_transform_edt
from scipy.linalg import cholesky, solve_triangular, eigh, qr
import numpy as np
from scipy.special import eval_jacobi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath

SCHEMA='registered-slm-basis-2'


def data_root():
    for start in (Path(__file__).resolve().parent,Path.cwd().resolve()):
        for parent in (start,*start.parents):
            if (parent/'.git').exists():return parent/'data'
    raise ValueError('No Git repository found; supply --output explicitly')


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False))


def polygon_centroid(poly):
    q=np.roll(poly,-1,axis=0);cross=poly[:,0]*q[:,1]-q[:,0]*poly[:,1]
    area2=cross.sum()
    if abs(area2)<1e-6:raise ValueError('Degenerate pupil polygon')
    return ((poly+q)*cross[:,None]).sum(0)/(3*area2)


def geometry(config,buffer_pixels=16,taper_pixels=0):
    if config.get('schema')!='slm-shwfs-patch-registration-1':raise ValueError('Unsupported registration schema')
    nx,ny=int(config['slm']['nx']),int(config['slm']['ny'])
    if nx<=0 or ny<=0:raise ValueError('Invalid SLM dimensions')
    a=np.asarray(config['camera_to_slm_affine'],float)
    poly=np.asarray(config['pupil_slm_polygon_estimated'],float)
    if a.shape!=(2,3) or poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3 or not np.isfinite(a).all() or not np.isfinite(poly).all():
        raise ValueError('Invalid registration geometry')
    if abs(np.linalg.det(a[:,:2]))<1e-10:raise ValueError('Singular affine transform')
    inv=np.linalg.inv(a[:,:2]);camera_poly=(poly-a[:,2])@inv.T
    centre=polygon_centroid(camera_poly)
    radius=float(np.max(np.linalg.norm(camera_poly-centre,axis=1)))
    lo=np.maximum(0,np.floor(poly.min(0)).astype(int)-buffer_pixels-1);hi=np.minimum([nx,ny],np.ceil(poly.max(0)).astype(int)+buffer_pixels+2)
    if np.any(hi<=lo):raise ValueError('Pupil does not intersect the SLM')
    yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
    points=np.column_stack((xx.ravel(),yy.ravel()))
    # Explicit closing vertex avoids Path(closed=True) discarding a polygon vertex.
    mask=PolygonPath(np.vstack((poly,poly[0]))).contains_points(points).reshape(xx.shape)
    if mask.sum()<16:raise ValueError('Too few illuminated SLM pixels')
    distance_out=distance_transform_edt(~mask)
    buffer=(~mask)&(distance_out<=buffer_pixels)
    taper=np.ones(mask.shape,float)
    if taper_pixels>0:
        distance_in=distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]-1
        taper=np.sin(.5*np.pi*np.clip(distance_in/taper_pixels,0,1))**2
    taper[~mask]=0
    cam=(points-a[:,2])@inv.T
    uv=((cam-centre)/radius).reshape(*xx.shape,2)
    return dict(nx=nx,ny=ny,bbox_xyxy=[*lo.tolist(),*hi.tolist()],mask=mask,buffer=buffer,taper=taper,uv=uv,
                camera_polygon=camera_poly,camera_centre=centre,camera_radius=radius,
                affine=a,xx=xx,yy=yy,slm_centre=a[:,:2]@centre+a[:,2])


def zernike_nm(count):
    # OSA/ANSI ordering: increasing radial degree n, m=-n,-n+2,...,+n. Omit n=0.
    result=[];n=1
    while len(result)<count:
        for m in range(-n,n+1,2):
            result.append(dict(kind='zernike',n=n,m=m,osa_index=(n*(n+2)+m)//2))
            if len(result)==count:break
        n+=1
    return result


def fourier_specs(max_cycles,layout,orientations,frequencies,geo):
    if layout=='lattice':
        limit=int(np.floor(max_cycles));pairs=[]
        for ky in range(-limit,limit+1):
            for kx in range(0,limit+1):
                if (kx>0 or ky>0) and 0<np.hypot(kx,ky)<=max_cycles+1e-12:pairs.append((kx,ky))
        pairs.sort(key=lambda k:(k[0]**2+k[1]**2,np.arctan2(k[1],k[0])))
        waves=[(float(np.hypot(x,y)),float(np.degrees(np.arctan2(y,x))),[x,y]) for x,y in pairs]
    else:
        freqs=frequencies if frequencies is not None else list(np.arange(1,int(np.floor(max_cycles))+1,dtype=float))
        angles=sorted(set(float(t)%180 for t in orientations))
        waves=[(float(f),theta,None) for f in sorted(set(freqs)) for theta in angles]
    specs=[]
    for f,theta,lattice in waves:
        d=np.array([np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))])
        span=float(np.ptp(geo['camera_polygon']@d))
        kc=f/span*d
        ks=np.linalg.solve(geo['affine'][:,:2].T,kc)
        if np.max(abs(ks))>=.5:raise ValueError('Mode exceeds native SLM pixel Nyquist')
        for phase in (0.,90.):
            specs.append(dict(kind='fourier',cycles_per_pupil=f,orientation_camera_deg=theta,
                lattice_direction=lattice,phase_at_pupil_centre_deg=phase,
                fx_cpa=float(ks[0]),fy_cpa=float(ks[1]),
                phase_at_slm_origin_deg=float((phase-360*np.dot(ks,geo['slm_centre']))%360)))
    return specs


def raw_mode(spec,geo):
    if spec['kind']=='fourier':
        return np.sin(2*np.pi*(spec['fx_cpa']*geo['xx']+spec['fy_cpa']*geo['yy'])+np.deg2rad(spec['phase_at_slm_origin_deg']))
    n,m=spec['n'],spec['m'];am=abs(m)
    if n<1 or am>n or (n-am)%2:raise ValueError('Invalid non-piston Zernike n,m')
    x,y=geo['uv'][...,0],geo['uv'][...,1];rho=np.minimum(1,np.hypot(x,y));theta=np.arctan2(y,x)
    radial=rho**am*eval_jacobi((n-am)//2,0,am,2*rho*rho-1)
    angular=np.cos(am*theta) if m>=0 else np.sin(am*theta)
    return np.sqrt((n+1)*(1 if m==0 else 2))*radial*angular


def normalise(raw,mask,normalization):
    mean=float(np.mean(raw[mask]));centred=raw[mask]-mean
    scale=float(np.max(abs(centred)) if normalization=='peak' else np.sqrt(np.mean(centred**2)))
    if not np.isfinite(scale) or scale<1e-10:raise ValueError('Mode is degenerate on registered pupil')
    mode=np.zeros(mask.shape,np.float32);mode[mask]=centred/scale
    return mode,mean,scale


def orthogonalizer(gram,tolerance):
    """Rank-reveal first, then ordered Cholesky: retain seed ordering where possible."""
    gram=(gram+gram.T)/2
    values,vectors=eigh(gram)
    keep=values>tolerance*values[-1]
    rank=int(keep.sum())
    if not rank:raise ValueError('No numerically independent modes remain')
    if rank<len(gram):
        # Pick independent seed columns in the retained eigenspace, then restore order.
        feature=np.sqrt(values[keep])[:,None]*vectors[:,keep].T
        _,_,pivots=qr(feature,mode='economic',pivoting=True)
        selected=np.sort(pivots[:rank])
    else:selected=np.arange(len(gram))
    sub=gram[np.ix_(selected,selected)]
    lower=cholesky(sub,lower=True)
    mixing=np.zeros((len(gram),rank))
    mixing[selected]=solve_triangular(lower.T,np.eye(rank),lower=False)
    error=float(np.max(abs(mixing.T@gram@mixing-np.eye(rank))))
    if error>1e-5:raise ValueError('Unstable orthogonalisation; increase --rank-tolerance or reduce mode count')
    return mixing,selected,values,error


class Basis:
    """Read recipes and mixing; reconstruct modes/commands without hardware access."""
    def __init__(self,manifest):
        self.path=Path(manifest);self.info=json.loads(self.path.read_text())
        if self.info.get('schema')!=SCHEMA or not self.info.get('complete'):raise ValueError('Unsupported or incomplete basis')
        snapshot=(self.path.parent/'registration_used.json').read_bytes()
        if hashlib.sha256(snapshot).hexdigest()!=self.info['registration_sha256']:raise ValueError('Registration snapshot changed')
        self.geo=geometry(json.loads(snapshot),self.info['buffer_pixels'],self.info['taper_pixels'])
        with np.load(self.path.parent/'pupil_mask.npz') as f:
            if not np.array_equal(self.geo['mask'],f['mask']):raise ValueError('Pupil mask differs from registration')
        mixing_path=self.path.parent/'mixing.npz'
        if self.info.get('mixing_sha256') and hashlib.sha256(mixing_path.read_bytes()).hexdigest()!=self.info['mixing_sha256']:raise ValueError('Mixing matrix changed')
        with np.load(mixing_path) as f:self.mixing=f['seed_to_orthonormal']

    def mode(self,index,full=False):
        if not 0<=index<len(self.info['modes']):raise IndexError('Mode index out of range')
        out=np.zeros(self.geo['mask'].shape,float)
        for j,coefficient in enumerate(self.mixing[:,index]):
            if coefficient==0:continue
            spec=self.info['seeds'][j]
            out+=coefficient*(raw_mode(spec,self.geo)-spec['removed_piston'])*self.geo['taper']/spec['normalisation_scale']
        out[~self.geo['mask']]=0;out=out.astype(np.float32)
        if not full:return out
        result=np.zeros((self.geo['ny'],self.geo['nx']),np.float32)
        x0,y0,x1,y1=self.geo['bbox_xyxy'];result[y0:y1,x0:x1]=out
        return result

    def command(self,index,amplitude,baseline=None,amplitude_units='rms'):
        baseline=self.info['baseline'] if baseline is None else baseline
        if not np.isfinite([amplitude,baseline]).all() or int(baseline)!=baseline or not 0<=baseline<=255:
            raise ValueError('Finite amplitude and integer baseline in 0..255 required')
        mode=self.mode(index)
        if amplitude_units=='peak':mode=mode/np.max(abs(mode))
        elif amplitude_units!='rms':raise ValueError('Amplitude units must be rms or peak')
        if np.max(abs(float(amplitude)*mode))>min(baseline,255-baseline)+1e-6:
            raise ValueError('Amplitude exceeds grey headroom; lower it or use --amplitude-units peak')
        delta=np.rint(float(amplitude)*mode)
        result=np.full((self.geo['ny'],self.geo['nx']),int(baseline),np.uint8)
        x0,y0,x1,y1=self.geo['bbox_xyxy'];result[y0:y1,x0:x1]=(baseline+delta).astype(np.uint8)
        return result


def preview(root,geo,images,labels):
    count=len(images);cols=min(4,count);rows=int(np.ceil(count/cols))
    fig,axs=plt.subplots(rows,cols,figsize=(3.5*cols,3.5*rows),squeeze=False)
    for i,ax in enumerate(axs.flat):
        if i>=count:ax.axis('off');continue
        im=np.ma.array(images[i],mask=~geo['mask']);v=max(1e-6,float(np.max(abs(images[i]))))
        ax.imshow(im,origin='upper',cmap='RdBu_r',vmin=-v,vmax=v,extent=[geo['bbox_xyxy'][0],geo['bbox_xyxy'][2],geo['bbox_xyxy'][3],geo['bbox_xyxy'][1]])
        ax.set(title=labels[i],xlabel='Native SLM x',ylabel='Native SLM y')
    fig.suptitle('Dimensionless pupil modes; outside pupil stays at baseline')
    fig.tight_layout();fig.savefig(root/'basis_preview.png',dpi=150);plt.close(fig)


def write_basis_fits(path,cube,geo,meta,registration,mixing):
    """Standard FITS HDUs with a documented, basis-independent application schema."""
    from astropy.io import fits
    x0,y0,x1,y1=geo['bbox_xyxy']
    h=fits.Header()
    for key,value in dict(SCHEMA='SLMBASIS',SCHEMAV=1,COMPLETE=True,BASISTYP=meta['basis'],
        NMODES=len(meta['modes']),NSEEDS=len(meta['seeds']),SLMNX=geo['nx'],SLMNY=geo['ny'],
        X0=x0,Y0=y0,X1=x1,Y1=y1,BASELINE=meta['baseline'],BUNIT='1',
        NORM='PUPIL_RMS',ORTHO=True,PISTFREE=True,REGVALID=meta['registration_valid'],
        BUFFPX=meta['buffer_pixels'],TAPERPX=meta['taper_pixels'],
        GRAMERR=meta['max_orthonormality_error'],PISTERR=meta['max_absolute_piston'],
        INPROD='MEAN_OVER_PUPIL',AMPUNIT='grey RMS',AXORDER='mode,y,x',
        OUTSIDE='BASELINE',ORIGIN='UPPER_LEFT',REGHASH=meta['registration_sha256'],
        CTYPE1='LINEAR',CTYPE2='LINEAR',CTYPE3='MODE',CRPIX1=1.,CRPIX2=1.,CRPIX3=1.,
        CRVAL1=float(x0),CRVAL2=float(y0),CRVAL3=0.,CDELT1=1.,CDELT2=1.,CDELT3=1.).items():h[key]=value
    h['COMMENT']='Primary data: float32 [mode,y,x] cropped dimensionless orthonormal modes.'
    h['COMMENT']='SLM xy are zero-based pixel centres. Crop endpoints X1,Y1 are exclusive.'
    h['COMMENT']='Command = BASELINE + rint(amplitude_grey_RMS * mode); reject overflow.'
    h['COMMENT']='Outside crop and PUPIL mask: zero modulation, command BASELINE.'
    primary=fits.PrimaryHDU(cube,header=h)
    modes=meta['modes']
    columns=[fits.Column(name='MODE_ID',format='J',array=[v['id'] for v in modes]),
        fits.Column(name='SEED_ID',format='J',array=[v['ordered_seed_index'] for v in modes])]
    for name,key in [('RMS','rms'),('MEAN','discrete_mean'),('PEAK_ABS','peak_abs'),('MAXAMP','max_abs_amplitude_grey_rms')]:
        columns.append(fits.Column(name=name,format='D',unit='grey' if name=='MAXAMP' else None,array=[v[key] for v in modes]))
    spatial=fits.Header(dict(X0=x0,Y0=y0,X1=x1,Y1=y1,SLMNX=geo['nx'],SLMNY=geo['ny']))
    # Provenance is optional for consumers: primary + PUPIL + MODES suffice to probe.
    payload=json.dumps(dict(basis=meta,registration=registration),ensure_ascii=True,allow_nan=False)
    provenance=fits.BinTableHDU.from_columns([fits.Column(name='JSON',format=f'{len(payload)}A',array=[payload])],name='PROVENANCE')
    hdus=fits.HDUList([primary,
        fits.ImageHDU(geo['mask'].astype(np.uint8),header=spatial,name='PUPIL'),
        fits.ImageHDU(geo['buffer'].astype(np.uint8),header=spatial,name='BUFFER'),
        fits.ImageHDU(geo['taper'].astype(np.float32),header=spatial,name='TAPER'),
        fits.BinTableHDU.from_columns(columns,name='MODES'),
        fits.ImageHDU(geo['affine'],name='CAM2SLM'),
        fits.ImageHDU(mixing,header=fits.Header({'AXORDER':'seed,mode'}),name='MIXING'),provenance])
    pending=path.with_name(path.stem+'.pending.fits')
    hdus.writeto(pending,overwrite=False,checksum=True,output_verify='exception')
    # Reopen and verify structure before publishing the completed filename.
    with fits.open(pending,memmap=True) as f:
        f.verify('exception')
        if f[0].data.shape!=(len(modes),*geo['mask'].shape):raise ValueError('FITS shape check failed')
    pending.replace(path)


def export_existing(manifest,output=None):
    """Export existing recipes without recomputing their orthogonalisation."""
    basis=Basis(manifest);geo=basis.geo;meta=basis.info;mask=geo['mask'];n=int(mask.sum())
    root=output or data_root()/('slm_basis_fits_'+time.strftime('%Y%m%d_%H%M%S'))
    root.mkdir(parents=True,exist_ok=False)
    registration=json.loads((basis.path.parent/'registration_used.json').read_text())
    with tempfile.TemporaryDirectory(prefix='.fits_work_',dir=root) as temp:
        samples=np.lib.format.open_memmap(Path(temp)/'seeds.npy',mode='w+',dtype=np.float64,shape=(len(meta['seeds']),n))
        for j,s in enumerate(meta['seeds']):
            samples[j]=((raw_mode(s,geo)[mask]-s['removed_piston'])*geo['taper'][mask]/s['normalisation_scale'])
        cube=np.lib.format.open_memmap(Path(temp)/'cube.npy',mode='w+',dtype=np.float32,shape=(len(meta['modes']),*mask.shape));cube[:]=0
        indices=np.flatnonzero(mask)
        for start in range(0,n,8192):
            stop=min(n,start+8192);q=np.array(samples[:,start:stop]).T@basis.mixing
            for j in range(len(meta['modes'])):cube[j].reshape(-1)[indices[start:stop]]=q[:,j]
        cube.flush();write_basis_fits(root/'basis.fits',cube,geo,meta,registration,basis.mixing)
        del samples,cube
    print(f'Exported self-contained FITS: {root}/basis.fits; no hardware accessed.')
    return root


def build(args):
    source=Path(args.config);source_bytes=source.read_bytes();cfg=json.loads(source_bytes)
    if not cfg.get('valid',False) and not args.allow_unvalidated:
        raise ValueError('Registration valid=false. Inspect diagnostics; --allow-unvalidated creates an exploratory basis only.')
    geo=geometry(cfg,args.buffer_pixels,args.taper_pixels)
    if args.basis=='fourier':
        if args.layout=='lattice' and args.frequencies is not None:raise ValueError('--frequencies requires --layout polar')
        specs=fourier_specs(args.max_cycles,args.layout,args.orientations,args.frequencies,geo)
        if not specs:raise ValueError('No Fourier modes requested')
        maxf=max(s['cycles_per_pupil'] for s in specs)
        if maxf>=args.lenslets_across/2 and not args.allow_aliasing:
            raise ValueError(f'{maxf:g} cpp reaches/exceeds nominal SHWFS Nyquist {args.lenslets_across/2:g}; use --allow-aliasing for deliberate experiments')
    else:specs=zernike_nm(args.modes);maxf=None
    root=args.output or data_root()/('slm_basis_'+args.basis+'_'+time.strftime('%Y%m%d_%H%M%S'))
    root.mkdir(parents=True,exist_ok=False);(root/'registration_used.json').write_bytes(source_bytes)
    mask=geo['mask'];n=int(mask.sum());m=len(specs)
    np.savez_compressed(root/'pupil_mask.npz',mask=mask,buffer=geo['buffer'],taper=geo['taper'],bbox_xyxy=geo['bbox_xyxy'],slm_shape_yx=[geo['ny'],geo['nx']])
    warnings=[]
    if not cfg.get('valid',False):warnings.append('EXPLORATORY: registration failed quality gates')
    if maxf is not None and maxf>=args.lenslets_across/2:warnings.append('Seeds reach/exceed nominal WFS Nyquist; aliasing expected')
    meta=dict(schema=SCHEMA,complete=False,basis=args.basis,registration_valid=bool(cfg.get('valid',False)),
        registration_source=str(source.resolve()),registration_sha256=hashlib.sha256(source_bytes).hexdigest(),
        baseline=int(cfg['slm'].get('baseline',128)),slm=dict(nx=geo['nx'],ny=geo['ny']),
        pupil_bbox_xyxy=geo['bbox_xyxy'],pupil_pixels=n,buffer_pixels=args.buffer_pixels,taper_pixels=args.taper_pixels,
        inner_product='mean(mode_i * mode_j) over all pixels of registered pupil mask; uniform area weighting',
        normalization='orthonormal, unit pupil RMS; NOT unit peak',piston_removed_on_discrete_mask=True,
        orthogonalization='rank-revealing eigenspace/QR selection then ordered Cholesky; output modes are mixtures of seeds',
        rank_tolerance=args.rank_tolerance,
        camera_centre_xy=geo['camera_centre'].tolist(),zernike_reference_radius_camera_px=geo['camera_radius'],
        coordinate_convention='x right, y down; seed orientations/angles in CAMERA xy; native SLM pixels are actuators',
        zernike_convention='OSA n,m order, omit n=0; negative m sine, positive m cosine; orthonormal outputs mix these seeds',
        fourier_convention='Sine phases 0/90 at pupil centre; seed cpp measured across projected registered polygon; outputs mix seed frequencies',
        taper_convention='sin^2 inward taper; subtract taper-weighted seed mean before multiplying taper to retain zero piston AND edge taper',
        command_convention='baseline+rint(amplitude*mode); default amplitude is grey RMS; buffer AND all other outside pixels pinned to baseline',
        nominal_lenslets_across=args.lenslets_across,nominal_wfs_nyquist_cpp=args.lenslets_across/2,
        warnings=warnings,seeds=[],modes=[])
    print(f'{m} seeds; {n} pupil pixels. Temporary workspace ~{m*n*8/1e9:.2f} GB; FITS payload ~{m*mask.size*4/1e9:.2f} GB. Output: {root}',flush=True)
    for w in warnings:print(w,flush=True)
    # Disk-backed samples avoid holding hundreds of full-SLM arrays in RAM.
    with tempfile.TemporaryDirectory(prefix='.basis_work_',dir=root) as temp:
        samples=np.lib.format.open_memmap(Path(temp)/'seeds.npy',mode='w+',dtype=np.float64,shape=(m,n))
        taper=geo['taper'][mask]
        if taper.sum()==0:raise ValueError('Taper eliminates pupil support')
        for i,spec in enumerate(specs):
            raw=raw_mode(spec,geo)[mask]
            mean=float(np.sum(taper*raw)/taper.sum());v=taper*(raw-mean)
            scale=float(np.sqrt(np.mean(v*v)))
            if scale<1e-12:raise ValueError('Degenerate seed after taper/piston removal')
            samples[i]=v/scale
            meta['seeds'].append(dict(spec,id=i,removed_piston=mean,normalisation_scale=scale))
            if (i+1)%100==0:print(f'Prepared {i+1}/{m} seeds',flush=True)
        samples.flush();gram=np.zeros((m,m));block=8192
        print('Computing discrete pupil Gram matrix...',flush=True)
        for start in range(0,n,block):
            x=np.array(samples[:,start:start+block]);gram+=x@x.T/n
        mixing,selected,eigenvalues,error=orthogonalizer(gram,args.rank_tolerance);rank=mixing.shape[1]
        print(f'Orthonormalising: retained {rank}/{m} independent seeds; Gram error {error:.2g}',flush=True)
        np.savez_compressed(root/'mixing.npz',seed_to_orthonormal=mixing,selected_seed_indices=selected,seed_gram=gram,eigenvalues=eigenvalues)
        meta['mixing_sha256']=hashlib.sha256((root/'mixing.npz').read_bytes()).hexdigest()
        # Evaluate exact output statistics and verify after float32 storage conversion.
        peaks=np.zeros(rank);means=np.zeros(rank);final_gram=np.zeros((rank,rank))
        previews=np.zeros((min(12,rank),mask.size),np.float32);indices=np.flatnonzero(mask)
        cube_path=root/('modes.npy' if args.save_modes else '.fits_cube.npy')
        cube=np.lib.format.open_memmap(cube_path,mode='w+',dtype=np.float32,shape=(rank,*mask.shape))
        if cube is not None:cube[:]=0
        for start in range(0,n,block):
            stop=min(start+block,n);q=(np.array(samples[:,start:stop]).T@mixing).astype(np.float32)
            q64=q.astype(float);final_gram+=q64.T@q64/n;means+=q64.sum(0)/n;peaks=np.maximum(peaks,np.max(abs(q64),axis=0))
            previews[:,indices[start:stop]]=q[:,:len(previews)].T
            if cube is not None:
                for j in range(rank):cube[j].reshape(-1)[indices[start:stop]]=q[:,j]
        if cube is not None:cube.flush();del cube
        del samples
    final_error=float(np.max(abs(final_gram-np.eye(rank))))
    if final_error>1e-5 or max(abs(means))>1e-5:raise ValueError('Numerical orthonormality/piston check failed; basis remains incomplete')
    for j,seed in enumerate(selected):
        meta['modes'].append(dict(id=j,ordered_seed_index=int(seed),peak_abs=float(peaks[j]),rms=float(np.sqrt(final_gram[j,j])),
            discrete_mean=float(means[j]),max_abs_amplitude_grey_rms=float(min(meta['baseline'],255-meta['baseline'])/peaks[j])))
    labels=[f'{j}: orthonormal mode (seed {seed})' for j,seed in enumerate(selected[:len(previews)])]
    preview(root,geo,[v.reshape(mask.shape) for v in previews],labels)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    axs[0].imshow(mask.astype(int)+2*geo['buffer'],origin='upper',vmin=0,vmax=2);axs[0].set_title('Pupil=1, pinned outer buffer=2')
    axs[1].imshow(geo['taper'],origin='upper',vmin=0,vmax=1);axs[1].set_title('Inward taper (before orthogonalising)')
    art=axs[2].imshow(final_gram-np.eye(rank),cmap='RdBu_r');fig.colorbar(art,ax=axs[2]);axs[2].set_title('Measured Gram minus identity')
    fig.tight_layout();fig.savefig(root/'basis_checks.png',dpi=150);plt.close(fig)
    meta.update(complete=True,retained_modes=rank,dropped_seed_count=m-rank,
                max_orthonormality_error=final_error,max_absolute_piston=float(max(abs(means))))
    lenslets=sum(bool(v.get('fit_inlier')) for v in cfg.get('lenslets',[]))
    if lenslets and rank>2*lenslets:meta['warnings'].append('More basis modes than lenslet slopes: full-rank reconstructor impossible')
    meta['fits_file']='basis.fits'
    cube=np.load(cube_path,mmap_mode='r')
    write_basis_fits(root/'basis.fits',cube,geo,meta,cfg,mixing)
    del cube
    if not args.save_modes:cube_path.unlink()
    print(f'Saved {root}/basis.fits; float32 Gram error {final_error:.2g}; no hardware accessed.',flush=True)
    return root


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path,nargs='?',help='Registration JSON')
    p.add_argument('--basis',choices=['fourier','zernike'],default='fourier')
    p.add_argument('--output',type=Path,help='Override repository-root/data timestamped directory')
    p.add_argument('--max-cycles',type=float,default=12,help='Fourier cutoff in cycles/pupil, default12 (<16 for32 lenslets)')
    p.add_argument('--layout',choices=['lattice','polar'],default='lattice',help='2D half-plane frequency grid, or selected radial frequencies/orientations')
    p.add_argument('--frequencies',nargs='+',type=float,help='Explicit cycles/pupil; polar layout only; overrides max-cycles')
    p.add_argument('--orientations',nargs='+',type=float,default=[0,45,90,135],help='Polar layout angles in CAMERA xy, degrees')
    p.add_argument('--lenslets-across',type=float,default=32,help='Nominal diameter sampling for Nyquist warning, not inferred from config')
    p.add_argument('--allow-aliasing',action='store_true')
    p.add_argument('--allow-unvalidated',action='store_true')
    p.add_argument('--modes',type=int,default=30,help='Number of non-piston Zernikes; default30')
    p.add_argument('--buffer-pixels',type=int,default=16,help='Pinned outer band width in native SLM pixels; all farther pixels also flat')
    p.add_argument('--taper-pixels',type=float,default=0,help='Optional inward sin^2 taper width; applied before orthonormalisation')
    p.add_argument('--rank-tolerance',type=float,default=1e-6,help='Relative eigenvalue threshold for independent seed selection')
    p.add_argument('--save-modes',action='store_true',help='Also write cropped float32 mode cube modes.npy; can be large')
    p.add_argument('--export-manifest',type=Path,help='Export an existing basis.json to self-contained basis.fits without recalibration')
    p.add_argument('--render-manifest',type=Path,help='Render a saved basis mode to command.npy offline')
    p.add_argument('--mode',type=int,default=0);p.add_argument('--amplitude',type=float,default=10)
    p.add_argument('--amplitude-units',choices=['rms','peak'],default='rms')
    a=p.parse_args()
    numbers=[a.max_cycles,a.lenslets_across,a.amplitude,a.taper_pixels,a.rank_tolerance,*(a.frequencies or []),*a.orientations]
    if not np.isfinite(numbers).all() or a.max_cycles<=0 or a.lenslets_across<=0 or a.modes<1 or a.mode<0:p.error('Invalid numeric options')
    if a.buffer_pixels<0 or a.taper_pixels<0 or not 0<a.rank_tolerance<1:p.error('Invalid buffer/taper/rank threshold')
    if a.frequencies and min(a.frequencies)<=0:p.error('Frequencies must be positive')
    if sum(v is not None for v in (a.config,a.render_manifest,a.export_manifest))!=1:p.error('Choose registration JSON, --render-manifest, or --export-manifest')
    return a


def main():
    args=arguments()
    if args.export_manifest:
        export_existing(args.export_manifest,args.output)
        return
    if args.render_manifest:
        basis=Basis(args.render_manifest)
        if not basis.info['complete']:raise ValueError('Basis generation was incomplete')
        command=basis.command(args.mode,args.amplitude,amplitude_units=args.amplitude_units)
        root=args.output or data_root()/('slm_command_'+time.strftime('%Y%m%d_%H%M%S'));root.mkdir(parents=True,exist_ok=False)
        np.save(root/'command.npy',command)
        x0,y0,x1,y1=basis.geo['bbox_xyxy']
        delta=command[y0:y1,x0:x1][basis.geo['mask']].astype(float)-basis.info['baseline']
        save_json(root/'command.json',dict(basis=str(args.render_manifest.resolve()),mode=args.mode,amplitude=args.amplitude,amplitude_units=args.amplitude_units,baseline=basis.info['baseline'],
            realised_pupil_rms_grey=float(np.sqrt(np.mean(delta*delta))),realised_piston_grey=float(delta.mean()),
            realised_peak_grey=float(np.max(abs(delta))),command_sha256=hashlib.sha256(command.tobytes()).hexdigest()))
        plt.imsave(root/'command.png',command,cmap='gray',vmin=0,vmax=255)
        print(f'Saved {root}/command.npy; NOT displayed on SLM.')
    else:build(args)

if __name__=='__main__':main()
