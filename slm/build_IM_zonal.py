#!/usr/bin/env python3
"""Acquire a split-region SLM phase-calibration scan using the v2 hardware APIs.

Reference pixels remain at --baseline. Probe pixels receive an ABSOLUTE grey
level, not a sinusoidal amplitude. Both illuminated regions must appear in the
SAME camera frame. The camera must image the SLM regions sufficiently well to
measure their separate Michelson fringe phases. This program acquires raw data;
it does not set the camera exposure/gain or infer phase/OPD.

Examples:
  # Show the command plan only; NO camera/SLM connection or file creation:
  python3 build_IM_v3.py --dry-run --probe-half left --grey-step 8

  # Full scan, up then down, with a simultaneous fixed right-half reference:
  python3 build_IM_v3.py --probe-half left --grey-step 8 --nframes 5 \
      --settle 0.3 --output slm_calibration.fits

  # Define a probe rectangle in SLM pixels; all pixels outside it stay fixed:
  python3 build_IM_v3.py --probe-roi 1000 1500 500 1100 --grey-step 8 \
      --nframes 5 --output slm_rectangle.fits

Coordinates: x is command-array column, y is command-array row; bounds are
zero-based and upper-exclusive. Hardware display orientation can differ from
camera orientation. Choose the region boundary through the illuminated footprint,
not merely through unused SLM pixels. Avoid the boundary in subsequent analysis.

Scan levels include both requested endpoints and the baseline if it lies within
the requested range. --directions both reverses immediately at the upper endpoint,
without an intervening reset/flat. A fresh endpoint acquisition begins the reverse
pass. Dedicated full-flat frames bracket each complete repeat; there is no flat
between the up/down passes, preserving scan history for hysteresis checks.

FITS: PRIMARY; FLAT_CMD; PROBE_MASK; repeated FLAT_FRAMES, SLM_CMD, CAM_FRAMES;
and FRAME_META tables containing each frame counter and HOST capture timestamps.
Individual exposures are never averaged. SLM_CMD stores the exact uint8 command.
CAM_FRAMES includes GREY (absolute), AMPL=GREY-BASE, MODEID=0, SCANID, REPEAT,
DIRECTION, LEVELIDX, SEQID, region/crop metadata and a pointer to FRAME_META.
This is BASIS='split_uniform', NOT a Fourier acquisition; the old Fourier/ADU
analysis scripts must not interpret these commands as sinusoidal modes.

Exposure, gain, QE, wavelength and LUT note are USER-SUPPLIED metadata, not read
back or verified hardware settings. Polling timestamps are not exposure times.
The server remains responsible for exposure, triggering and camera settings.
Memory holds one captured cube, one source frame, and a few SLM images at a time.
--camera-roi reduces stored camera data; it must contain BOTH probe and reference.
FITS is appended incrementally; COMPLETE remains false after interruption/failure.
SLM is returned to the baseline on normal exit and catchable errors/interruption.

Adapted from the supplied build_IM_v2.py. Written but not executed or hardware-tested.
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from astropy.io import fits


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')


def grey_levels(args):
    if args.grey_levels is not None:
        levels=sorted(set(args.grey_levels))
        if len(levels)<2:raise ValueError('--grey-levels must contain at least two distinct levels')
    else:
        levels=list(range(args.grey_min,args.grey_max+1,args.grey_step))
        levels.append(args.grey_max)
    if not all(0<=v<=255 for v in levels):raise ValueError('All grey levels must be in 0..255')
    if min(levels)<=args.baseline<=max(levels):levels.append(args.baseline)
    return sorted(set(levels))


def probe_bounds(args,shape):
    h,w=shape
    if args.probe_roi is not None:
        x0,x1,y0,y1=args.probe_roi
    else:
        half=args.probe_half or 'left'
        if half in ('left','right'):
            split=int(round(args.split_fraction*w))
            x0,x1=(0,split) if half=='left' else (split,w)
            y0,y1=0,h
        else:
            split=int(round(args.split_fraction*h))
            x0,x1=0,w
            # 'top' means lower array row indices, not guaranteed camera top.
            y0,y1=(0,split) if half=='top' else (split,h)
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Probe ROI outside the SLM or empty')
    if (x1-x0)*(y1-y0)==h*w:raise ValueError('Probe cannot cover the entire SLM: reference pixels are required')
    return x0,x1,y0,y1


def plan(args,levels):
    scan_id=0;sequence=0;command_id=0
    for repeat in range(args.repeats):
        yield dict(kind='flat',repeat=repeat,scan=-1,direction='flat_start',levelidx=-1,
                   grey=args.baseline,sequence=sequence,command_id=-1)
        sequence+=1
        directions=['up','down'] if args.directions=='both' else [args.directions]
        for direction in directions:
            ordered=levels if direction=='up' else levels[::-1]
            for levelidx,grey in enumerate(ordered):
                yield dict(kind='probe',repeat=repeat,scan=scan_id,direction=direction,
                           levelidx=levelidx,grey=grey,sequence=sequence,command_id=command_id)
                sequence+=1;command_id+=1
            scan_id+=1
        yield dict(kind='flat',repeat=repeat,scan=-1,direction='flat_end',levelidx=-1,
                   grey=args.baseline,sequence=sequence,command_id=-1)
        sequence+=1


def next_frame(cam,previous_count,timeout,app):
    """Same stable shared-memory copy check as v2. No hardware timestamp implied."""
    deadline=time.monotonic()+timeout;last_events=0.
    while time.monotonic()<deadline:
        count=int(cam.md.cnt0)
        if count!=previous_count and not cam.md.write:
            im=np.array(cam,copy=True)
            if int(cam.md.cnt0)==count and not cam.md.write:
                if count<previous_count:raise RuntimeError('Camera frame counter reset/wrapped during acquisition')
                return count,im,utc_now(),time.monotonic()
        now=time.monotonic()
        if now-last_events>.02:
            app.processEvents();last_events=now
        time.sleep(.001)
    raise TimeoutError('No distinct stable DCAM frame received')


def crop_bounds(args,shape):
    if len(shape)!=2:raise ValueError(f'Expected a 2D camera image, got {shape}')
    h,w=shape
    roi=tuple(args.camera_roi) if args.camera_roi is not None else (0,w,0,h)
    x0,x1,y0,y1=roi
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('--camera-roi outside camera image')
    return roi


def acquire(slm,app,command,cam,args,cam_shape,cam_dtype,crop):
    cmd_utc=utc_now();cmd_mono=time.monotonic()
    slm.show(command);app.processEvents()
    until=time.monotonic()+args.settle
    while time.monotonic()<until:
        app.processEvents();time.sleep(min(.01,max(0,until-time.monotonic())))
    # Discard one complete new frame after settling, as in v2.
    count=int(cam.md.cnt0)
    count,discard,_,_=next_frame(cam,count,args.timeout,app)
    del discard
    x0,x1,y0,y1=crop
    cube=np.empty((args.nframes,y1-y0,x1-x0),dtype=cam_dtype)
    counters=[];times=[];monotonic=[];gaps=[]
    for k in range(args.nframes):
        old=count
        count,im,t,tm=next_frame(cam,old,args.timeout,app)
        if im.shape!=cam_shape or im.dtype!=cam_dtype:
            raise RuntimeError('Camera frame shape/dtype changed during acquisition')
        cube[k]=im[y0:y1,x0:x1]
        counters.append(count);times.append(t);monotonic.append(tm);gaps.append(max(0,count-old-1))
        del im
    return cube,dict(counters=counters,times=times,monotonic=monotonic,gaps=gaps,
                     command_utc=cmd_utc,command_mono=cmd_mono)


def region_header(header,bounds,crop):
    for key,value in zip(('PRBX0','PRBX1','PRBY0','PRBY1'),bounds):header[key]=value
    for key,value in zip(('CAMX0','CAMX1','CAMY0','CAMY1'),crop):header[key]=value
    header['REFREG']='outside PROBE_MASK'
    header['COORDSYS']='zero-based; upper-exclusive'


def write_block(path,event,command,cube,timing,args,bounds,crop,flat_version,probe_version):
    isflat=event['kind']=='flat'
    version=flat_version if isflat else probe_version
    frame_name='FLAT_FRAMES' if isflat else 'CAM_FRAMES'
    h=fits.Header()
    h['SEQID']=event['sequence'];h['CMDID']=event['command_id'];h['MODEID']=0
    h['SCANID']=event['scan'];h['REPEAT']=event['repeat'];h['DIRECTION']=event['direction']
    h['LEVELIDX']=event['levelidx'];h['AMPIDX']=event['levelidx']
    h['GREY']=event['grey'];h['BASE']=args.baseline;h['AMPL']=float(event['grey']-args.baseline)
    h['PROBETYP']='uniform rectangle';h['WVLNM']=args.wavelength_nm
    h['CMDDATE']=timing['command_utc'];h['CMDMONO']=timing['command_mono']
    h['TIME_SRC']='HOST_COPY';h['METAEXT']=event['sequence']+1
    region_header(h,bounds,crop)
    if not isflat:
        ch=h.copy();ch['EXTNAME']='SLM_CMD';ch['EXTVER']=version
        fits.append(path,command,header=ch)
    h['EXTNAME']=frame_name;h['EXTVER']=version;h['NFRAME']=len(cube)
    h['DATE-OBS']=timing['times'][0];h['DATE-END']=timing['times'][-1]
    h['CNTFIRST']=timing['counters'][0];h['CNTLAST']=timing['counters'][-1]
    h['DROPPED']=sum(timing['gaps'])
    h['CMDREF']='FLAT_CMD' if isflat else 'SLM_CMD';h['CMDVER']=1 if isflat else version
    fits.append(path,cube,header=h)
    columns=[fits.Column(name='FRAME',format='J',array=np.arange(len(cube),dtype=np.int32)),
             fits.Column(name='COUNTER',format='K',array=np.asarray(timing['counters'],dtype=np.int64)),
             fits.Column(name='HOST_UTC',format='26A',array=np.asarray(timing['times'],dtype='S26')),
             fits.Column(name='HOST_MONO',format='D',unit='s',array=timing['monotonic']),
             fits.Column(name='MISSED',format='K',array=np.asarray(timing['gaps'],dtype=np.int64))]
    table=fits.BinTableHDU.from_columns(columns,name='FRAME_META')
    table.header['EXTVER']=event['sequence']+1;table.header['SEQID']=event['sequence']
    table.header['FRAMEEXT']=frame_name;table.header['FRAMEVER']=version
    table.header['TIME_SRC']='HOST_COPY';table.header['HARDTIME']=False
    fits.append(path,table.data,header=table.header)


def arguments():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--probe-half',choices=['left','right','top','bottom'],help='Default: left half')
    group.add_argument('--probe-roi',type=int,nargs=4,metavar=('X0','X1','Y0','Y1'),help='SLM pixels, not camera pixels')
    p.add_argument('--split-fraction',type=float,default=.5,help='Half-plane boundary as fraction of SLM width/height')
    p.add_argument('--grey-min',type=int,default=0);p.add_argument('--grey-max',type=int,default=255)
    p.add_argument('--grey-step',type=int,default=8)
    p.add_argument('--grey-levels',type=int,nargs='+',help='Explicit absolute grey levels; sorted and deduplicated')
    p.add_argument('--baseline',type=int,default=127,help='Fixed reference grey and cleanup value')
    p.add_argument('--directions',choices=['up','down','both'],default='both')
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--nframes',type=int,default=5)
    p.add_argument('--settle',type=float,default=.3);p.add_argument('--timeout',type=float,default=5.)
    p.add_argument('--screen',default='1');p.add_argument('--shm',default='dcam0_data')
    p.add_argument('--camera-roi',type=int,nargs=4,metavar=('X0','X1','Y0','Y1'),help='Stored camera crop; MUST include probe AND reference')
    p.add_argument('--output',type=Path)
    p.add_argument('--wavelength-nm',type=float,default=633.)
    p.add_argument('--exposure-us',type=float,default=17.,help='Metadata only: does not set camera exposure')
    p.add_argument('--gain-e-adu',type=float,default=.21,help='Metadata only')
    p.add_argument('--qe',type=float,default=.9,help='Metadata only')
    p.add_argument('--pixel-um',type=float,default=6.5,help='Metadata only')
    p.add_argument('--lut-note',default='not recorded',help='Record active LUT/calibration name manually')
    p.add_argument('--dry-run',action='store_true',help='Print plan without importing hardware drivers or making output files')
    p.add_argument('--slm-shape',type=int,nargs=2,default=[1605,2568],metavar=('HEIGHT','WIDTH'),help='Planning dimensions for dry-run ONLY; acquisition reads actual SLM dimensions')
    args=p.parse_args()
    if not 0<=args.baseline<=255:p.error('baseline must be in 0..255')
    if not 0<=args.grey_min<args.grey_max<=255 or args.grey_step<1:p.error('Require 0<=grey-min<grey-max<=255 and positive grey-step')
    if min(args.nframes,args.repeats,*args.slm_shape)<1:p.error('Counts and dimensions must be positive')
    if not np.isfinite(args.split_fraction) or not 0<args.split_fraction<1:p.error('split-fraction must be between 0 and 1')
    if not np.isfinite(args.settle) or args.settle<0:p.error('settle must be finite and nonnegative')
    if not all(np.isfinite(v) and v>0 for v in [args.timeout,args.wavelength_nm,args.exposure_us,args.gain_e_adu,args.pixel_um]):p.error('Timing and metadata values must be finite and positive')
    if not np.isfinite(args.qe) or not 0<args.qe<=1:p.error('QE must be in (0,1]')
    if not args.dry_run:
        if args.output is None:p.error('--output is required for acquisition')
        if args.output.exists():p.error(f'Output exists: {args.output}')
        if not args.output.parent.is_dir():p.error('Output parent directory does not exist')
    return args


def main():
    args=arguments();levels=grey_levels(args);events=list(plan(args,levels))
    nprobes=sum(e['kind']=='probe' for e in events);nflats=len(events)-nprobes
    print('Absolute probe grey levels:',levels,flush=True)
    print(f'Reference fixed at {args.baseline}; {nprobes} probe blocks + {nflats} flat blocks; '
          f'{len(events)*args.nframes} stored individual frames.',flush=True)
    print('Up/down scans have no intervening reset. Check that BOTH illuminated regions are visible.',flush=True)
    if args.dry_run:
        bounds=probe_bounds(args,tuple(args.slm_shape))
        print(f'Dry-run planning shape H,W={tuple(args.slm_shape)}; probe X0 X1 Y0 Y1={bounds}.')
        print('All other SLM pixels stay at baseline. No hardware accessed; no files created.')
        return

    # Hardware imports and connections occur only on explicit acquisition runs.
    from PyQt5 import QtWidgets
    import ImageStreamIOWrap
    from slmdisplay import SLM
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    cam=None;slm=None;entered=False;flat=None;created=False;complete=False;written=0
    try:
        cam=ImageStreamIOWrap.Image();cam.open(args.shm)
        # Learn the actual camera format and validate crop before changing SLM.
        count=int(cam.md.cnt0)
        _,sample,_,_=next_frame(cam,count,args.timeout,app)
        cam_shape=sample.shape;cam_dtype=sample.dtype;crop=crop_bounds(args,cam_shape)
        del sample
        screen=int(args.screen) if args.screen.lstrip('-').isdigit() else args.screen
        slm=SLM(screen=screen);slm.__enter__();entered=True
        shape=(slm.height,slm.width)
        flat=np.full(shape,args.baseline,dtype=np.uint8)
        bounds=probe_bounds(args,shape)
        mask=np.zeros(shape,dtype=np.uint8);x0,x1,y0,y1=bounds;mask[y0:y1,x0:x1]=1
        xlo,xhi,ylo,yhi=crop
        approximate=len(events)*args.nframes*(xhi-xlo)*(yhi-ylo)*cam_dtype.itemsize+(nprobes+2)*shape[0]*shape[1]
        print(f'Actual SLM H,W={shape}; probe={bounds}; camera crop={crop}',flush=True)
        print(f'Approximate uncompressed image data: {approximate/1e9:.2f} GB (plus FITS headers/tables).',flush=True)
        header=fits.Header()
        for key,value in dict(DATE=utc_now(),VERSION=3,BASIS='split_uniform',NMODE=1,
            NAMP=len(levels),NFRAME=args.nframes,NFLAT=nflats,NPROBE=nprobes,
            NREPEAT=args.repeats,DIRS=args.directions,BASE=args.baseline,
            GREYMIN=min(levels),GREYMAX=max(levels),SETTLE=args.settle,STREAM=args.shm,
            SLMW=shape[1],SLMH=shape[0],CAMW=cam_shape[1],CAMH=cam_shape[0],
            WVLNM=args.wavelength_nm,PIXPUM=args.pixel_um,EXPUS=args.exposure_us,
            GAIN_EAD=args.gain_e_adu,QE=args.qe,METASRC='user supplied',EXPSET=False,
            LUTNOTE=args.lut_note,COMPLETE=False,NWRITTEN=0).items():header[key]=value
        region_header(header,bounds,crop)
        header['HISTORY']='Camera settings are not set or verified by this script.'
        header['HISTORY']='Fringe phase must be measured relative to simultaneous fixed SLM region.'
        fits.PrimaryHDU(header=header).writeto(args.output,overwrite=False);created=True
        fh=fits.Header({'EXTNAME':'FLAT_CMD','EXTVER':1,'BASE':args.baseline})
        fits.append(args.output,flat,header=fh)
        mh=fits.Header({'EXTNAME':'PROBE_MASK','EXTVER':1,'MASKONE':'probe region','MASKZERO':'fixed reference region'})
        region_header(mh,bounds,crop);fits.append(args.output,mask,header=mh)
        del mask
        probe_version=0;flat_version=0
        for event in events:
            if event['kind']=='flat':
                flat_version+=1;command=flat
            else:
                probe_version+=1;command=flat.copy();command[y0:y1,x0:x1]=event['grey']
            cube,timing=acquire(slm,app,command,cam,args,cam_shape,cam_dtype,crop)
            write_block(args.output,event,command,cube,timing,args,bounds,crop,flat_version,probe_version)
            written+=1
            print(f'[{written}/{len(events)}] repeat={event["repeat"]+1} '
                  f'{event["direction"]} grey={event["grey"]} '
                  f'frames={args.nframes} missed={sum(timing["gaps"])}',flush=True)
            del cube,command
        complete=True
        print('Acquisition saved:',args.output,flush=True)
    finally:
        # Cleanup errors are reported without concealing the original acquisition error.
        if entered:
            try:
                if flat is not None:slm.show(flat);app.processEvents()
            except Exception as exc:print(f'WARNING: could not restore baseline: {exc}',file=sys.stderr)
            try:slm.__exit__(None,None,None)
            except Exception as exc:print(f'SLM cleanup warning: {exc}',file=sys.stderr)
        if cam is not None and hasattr(cam,'close'):
            try:cam.close()
            except Exception as exc:print(f'Camera cleanup warning: {exc}',file=sys.stderr)
        if created:
            try:
                with fits.open(args.output,mode='update',memmap=False) as hd:
                    hd[0].header['COMPLETE']=complete;hd[0].header['NWRITTEN']=written
                    hd[0].header['DATE-END']=utc_now();hd.flush()
            except Exception as exc:print(f'Could not update completion header: {exc}',file=sys.stderr)
        if created and not complete:
            print(f'Partial acquisition retained: {args.output}; completed blocks={written}',file=sys.stderr)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nAcquisition interrupted.',file=sys.stderr);sys.exit(130)
    except (ValueError,RuntimeError,TimeoutError,FileExistsError) as exc:raise SystemExit(f'Error: {exc}')
