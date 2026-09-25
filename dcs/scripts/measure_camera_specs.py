#!/usr/bin/env python3
"""Prompted dark/flat tests. Run beside take_darks.py: python measure_camera_specs.py
Edit the simple settings below. Flat source brightness is adjusted manually.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import sys
import time
import numpy as np
from astropy.io import fits
from take_darks import add_settings, write_fits, utc_text, ascii_text, DCAM_LIBRARY_DIR

DARK_FRAMES = 100
FRAMES_PER_FILE = 10             # Even, keeps 2048-square cubes manageable in RAM
FLAT_FRAMES = 10                 # Five independent pairs per light level initially
FLAT_TARGETS_ADU = [1000, 3000, 7000, 12000, 20000, 30000]  # Signal above dark
NOTES = ''                      # Optional environment/source notes; no terminal prompt
CLOCK_SYNC = 'unknown'           # Optional host clock status
RAW_CEILING_ADU = 50000          # Conservative test ceiling, NOT measured full well


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def confirm(message):
    answer = input(message+'\nPress Enter when ready (q to stop): ').strip().lower()
    if answer not in ('', 'yes', 'y'):
        raise RuntimeError('Stopped by operator')


def serial_key(value):
    return re.sub(r'^S/N\s*:\s*', '', str(value), flags=re.I).strip()


def select_readout(cam, mode):
    # ORCA-Fusion BT has three DCAM speed levels. Some Linux DCAM versions
    # expose READOUT SPEED as numeric, so ilabels can legitimately be empty.
    # This matches pylablib's three-speed mapping: slow=1, normal=2, fast=3.
    attr = cam.get_attribute('READOUT SPEED')
    if not attr.writable or attr.min != 1 or attr.max != 3:
        raise RuntimeError('Expected writable three-speed READOUT SPEED (range 1..3); '
                           f'camera reports {attr.min}..{attr.max}')
    chosen = {'ultra_quiet': 1, 'standard': 2, 'fast': 3}[mode]
    cam.set_attribute_value('READOUT SPEED', chosen)
    actual = cam.get_attribute_value('READOUT SPEED')
    if actual != chosen:
        raise RuntimeError(f'Readout setting did not take effect: requested {chosen}, got {actual}')
    label = {'ultra_quiet': 'Ultra quiet', 'standard': 'Standard', 'fast': 'Fast'}[mode]
    print(f'Readout: {label} (DCAM {actual})')
    return chosen, label


def preview_flat(frame, dark_level, target, path):
    """Show one raw frame and save the same view for headless sessions."""
    import matplotlib
    import matplotlib.pyplot as plt
    signal = float(frame.mean() - dark_level)
    peak = int(frame.max())
    print(f'Signal {signal:.1f} ADU above dark; maximum raw pixel {peak}')
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    im = ax.imshow(frame, origin='lower', cmap='gray')
    fig.colorbar(im, ax=ax, label='Raw ADU')
    ax.set(title=f'Flat preview: target ~{target} ADU above dark\n'
                 f'Measured {signal:.1f} ADU; peak {peak}',
           xlabel='ROI column', ylabel='ROI row')
    fig.savefig(path, dpi=120)
    print('Preview saved:', path)
    backend = matplotlib.get_backend().lower()
    if backend in ('agg', 'pdf', 'svg', 'ps', 'template', 'cairo'):
        print('No interactive display: open the saved PNG to inspect uniform illumination.')
    else:
        print('Inspect the flat image, then close its window to continue.')
        plt.show(block=True)
    plt.close(fig)
    return signal > 0 and peak < RAW_CEILING_ADU


def capture(cam, count):
    rows, cube = [], None
    for i in range(count):
        mp, hp = time.monotonic_ns(), time.time_ns()
        frames, infos = cam.grab(nframes=1, return_info=True, missing_frame='none',
                                frame_timeout=max(5.,cam.get_exposure()+5.),buff_size=1)
        he, me = time.time_ns(), time.monotonic_ns()
        if len(frames)!=1 or frames[0] is None or len(infos)!=1 or infos[0] is None:
            raise RuntimeError('Missing frame or DCAM timing metadata')
        frame, info = frames[0], infos[0]
        if cube is None:
            cube = np.empty((count,*frame.shape), dtype=frame.dtype)
        if frame.shape!=cube.shape[1:] or frame.dtype!=cube.dtype:
            raise RuntimeError('Frame dimensions/type changed')
        cube[i] = frame
        rows.append((i,hp,he,mp,me,info.timestamp_us,info.frame_index,info.framestamp,info.camerastamp))
    return cube, rows


def take_files(cam, folder, kind, count, base, level=0):
    names = []
    for start in range(0,count,FRAMES_PER_FILE):
        h = base.copy()
        h['IMAGETYP'], h['LEVEL'] = kind, level
        add_settings(h,cam.get_all_attribute_values(enum_as_str=False))
        cube, rows = capture(cam,min(FRAMES_PER_FILE,count-start))
        h['NFRAMES'] = len(cube)
        h['HOSTBEG'], h['HOSTEND'] = utc_text(rows[0][1]),utc_text(rows[-1][2])
        name = f'{kind.lower()}_level{level:02d}_{start:04d}.fits'
        write_fits(folder/name,cube,rows,h)
        names.append(name)
        print('Saved',name)
    return names


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',default='camera_1')
    p.add_argument('--device-index',type=int,default=0)
    p.add_argument('--mode',choices=['all','ultra_quiet','standard','fast'],default='all')
    p.add_argument('--dark-only',action='store_true')
    p.add_argument('--specs',type=Path,default=Path(__file__).with_name('camera_specs.json'))
    a=p.parse_args()
    document=json.loads(a.specs.read_text()); ref=document['cameras'][a.camera]
    if not ref.get('serial') or not ref.get('modes'):
        raise ValueError('Fill in this camera serial and reference specifications first')
    if any(n<2 or n%2 for n in [DARK_FRAMES,FLAT_FRAMES,FRAMES_PER_FILE]):
        raise ValueError('Frame counts must be even and >=2')
    import pylablib
    pylablib.par['devices/only_windows_dlls']=False
    pylablib.par['devices/dlls/dcamapi']=DCAM_LIBRARY_DIR
    from pylablib.devices import DCAM
    cam=DCAM.DCAMCamera(a.device_index)
    manifest=None
    try:
        device=cam.get_device_info()
        print('Connected:',device)
        if serial_key(device.serial_number)!=serial_key(ref['serial']) or device.model!=ref['model']:
            raise RuntimeError('Connected camera identity does not match JSON; check camera selection')
        conditions=ref['conditions']
        confirm('Confirm AIR cooling only, no liquid cooling, stable camera temperature, and the correct camera label')
        notes=NOTES
        clocks=CLOCK_SYNC
        root=Path(__file__).resolve().parent/'spec_tests'/a.camera/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        root.mkdir(parents=True,exist_ok=False)
        save_json(root/'camera_specs_snapshot.json',document)
        manifest={'status':'running','camera':a.camera,'reference':ref,'notes':notes,'clock_sync':clocks,
                  'cooling_operator_confirmed':'air only','pylablib':pylablib.__version__,'modes':[]}
        save_json(root/'run.json',manifest)
        modes=list(ref['modes']) if a.mode=='all' else [a.mode]
        bytes_est=len(modes)*(DARK_FRAMES+(0 if a.dark_only else len(FLAT_TARGETS_ADU)*FLAT_FRAMES))*conditions['roi_width']*conditions['roi_height']*2
        print(f'Approximate raw data: {bytes_est/2**30:.1f} GiB. Saving to {root}')
        for mode in modes:
            cam.clear_acquisition()
            mode_ref=ref['modes'][mode]
            rid,label=select_readout(cam,mode)
            cam.set_trigger_mode('int')
            cam.set_defect_correct_mode(False)
            cam.set_roi(hbin=1,vbin=1)
            x0,x1,y0,y1,_,_=cam.get_roi()
            w,h=conditions['roi_width'],conditions['roi_height']
            if x1-x0<w or y1-y0<h: raise RuntimeError('Requested ROI exceeds detector')
            left,top=x0+(x1-x0-w)//2,y0+(y1-y0-h)//2
            cam.set_roi(left,left+w,top,top+h,hbin=1,vbin=1)
            roi=tuple(cam.get_roi())
            cam.set_exposure(mode_ref['exposure_s']); actual=cam.get_exposure()
            if roi[1]-roi[0]!=w or roi[3]-roi[2]!=h or roi[4:]!=(1,1):
                raise RuntimeError('Actual ROI/binning does not match requested conditions')
            if cam.get_trigger_mode()!='int' or cam.get_defect_correct_mode():
                raise RuntimeError('Internal trigger / correction OFF not achieved')
            if cam.get_attribute_value('BIT PER CHANNEL')!=conditions['output_bits']:
                raise RuntimeError('Select 16-bit camera output before running this test')
            if abs(actual/mode_ref['exposure_s']-1)>0.05:
                raise RuntimeError(f'Exposure readback {actual:g}s is >5% from target; check sensor mode')
            print(f'{mode}: {actual*1e6:.4f} us, ROI {roi}, defect correction OFF')
            folder=root/mode; folder.mkdir()
            entry={'name':mode,'readout_id':rid,'readout_label':label,'exposure_s':actual,
                   'roi':list(roi),'dark_files':[],'flat_levels':[]}
            manifest['modes'].append(entry)
            header=fits.Header()
            for key,value in {'MODEL':device.model,'SERIAL':device.serial_number,'CAMVER':device.camera_version,
                'CAMID':a.camera,'SCANMODE':mode,'READID':rid,'READLAB':label,'EXPREQ':mode_ref['exposure_s'],
                'EXPTIME':actual,'ACQMODE':'INDIVIDUAL','COOLING':'AIR (operator confirmed)',
                'CLK_SYNC':clocks,'NOTES':notes,'BUNIT':'ADU','PYLABLIB':pylablib.__version__,
                'REFGAIN':mode_ref['gain_e_per_adu'],'REFNOISE':mode_ref['read_noise_e_rms']}.items():
                header[key]=ascii_text(value) if isinstance(value,str) else value
            for k,v in zip(['XSTART','XEND','YSTART','YEND','XBINNING','YBINNING'],roi):header[k]=int(v)
            confirm('DARKS: switch the source off and fully cover/light-seal the detector')
            entry['dark_files']=take_files(cam,folder,'DARK',DARK_FRAMES,header)
            with fits.open(folder/entry['dark_files'][0],memmap=False) as hd:
                dark_level=float(hd[0].data.mean())
            save_json(root/'run.json',manifest)
            if not a.dark_only:
                confirm('FLATS: uncover the camera and provide uniform, steady diffuse illumination. Keep camera settings unchanged')
                for level,target in enumerate(FLAT_TARGETS_ADU,1):
                    while True:
                        action=input(f'Level {level}: adjust source to about {target} ADU ABOVE dark. Enter = preview; q = stop: ')
                        if action.strip().lower()=='q':raise RuntimeError('Stopped by operator')
                        cube,_=capture(cam,1)
                        acceptable=preview_flat(cube[0],dark_level,target,folder/f'flat_preview_{level:02d}.png')
                        if not acceptable:
                            print('Adjust source: signal must be positive and every pixel below',RAW_CEILING_ADU)
                            continue
                        answer=input('Flat looks uniform and ready? Enter = capture; r = adjust/re-preview; q = stop: ').strip().lower()
                        if answer=='q':raise RuntimeError('Stopped by operator')
                        if answer in ('', 'yes', 'y'):break
                    flat_header=header.copy(); flat_header['TARGET']=target; flat_header['ADUCEIL']=RAW_CEILING_ADU
                    files=take_files(cam,folder,'FLAT',FLAT_FRAMES,flat_header,level)
                    entry['flat_levels'].append({'level':level,'target_adu':target,'files':files,'ceiling_adu':RAW_CEILING_ADU})
                    save_json(root/'run.json',manifest)
        manifest['status']='complete'
        analysis_script=Path(__file__).resolve().with_name('analyse_camera_specs.py')
        command=shlex.join([sys.executable,str(analysis_script),str(root)])
        print('\nAcquisition complete. Analyse with:\n'+command)
    except BaseException as exc:
        if manifest is not None:manifest['status']='incomplete';manifest['error']=repr(exc)
        raise
    finally:
        if manifest is not None:save_json(root/'run.json',manifest)
        cam.close()


if __name__=='__main__':main()
