#!/usr/bin/env python3
"""Restore saved camera ROI/exposure from registration.json, scan.json or basis FITS.
No SLM access. Restart/open the camera server first. No automatic server restart.
"""
import argparse
import json
import math
import re
import socket
import time
from pathlib import Path


def status(reply):
    fields=dict(re.findall(r'^\s*([\w ]+?)\s*:\s*\[\s*(.*?)\s*\]',reply,re.M))
    fw,fh=map(int,re.fullmatch(r'(\d+)x(\d+)',fields['Format size']).groups())
    w,h=map(int,re.fullmatch(r'(\d+)x(\d+)',fields['Image size']).groups())
    ox,oy=map(int,re.fullmatch(r'([+-]?\d+)\s*,\s*([+-]?\d+)',fields['Offset']).groups())
    return dict(sensor=(fw,fh),roi=[(fw-w)//2+ox,(fh-h)//2+oy,w,h],
                exposure=float(fields['Exptime']),capture=fields['Capture'],raw=reply)


def load_camera(path):
    if path.is_dir():
        path=next((path/n for n in ('registration.json','scan.json') if (path/n).is_file()),path/'registration.json')
    if path.suffix.lower() in ('.fits','.fit','.fts'):
        from astropy.io import fits
        with fits.open(path,memmap=True) as f:
            if f[0].header.get('SCHEMA')!='SLMBASIS':raise ValueError('Expected SLMBASIS FITS with PROVENANCE extension')
            payload=f['PROVENANCE'].data['JSON'][0]
            if isinstance(payload,bytes):payload=payload.decode('ascii')
            config=json.loads(payload)['registration']
    else:config=json.loads(path.read_text())
    cam=config['camera'];roi=cam['roi'];exposure=float(cam['exposure'])
    if len(roi)!=4 or any(isinstance(v,bool) or int(v)!=v for v in roi):raise ValueError('ROI must contain four integers: x,y,w,h')
    roi=list(map(int,roi));x,y,w,h=roi
    if min(x,y)<0 or min(w,h)<=0 or not math.isfinite(exposure) or exposure<=0:raise ValueError('Invalid saved ROI/exposure')
    sensor=None
    if cam.get('raw'):sensor=status(cam['raw'])['sensor']
    return path,dict(roi=roi,exposure=exposure,sensor=sensor)


def send(a,cmd):
    data=bytearray();deadline=time.monotonic()+a.timeout
    with socket.create_connection((a.host,a.port),timeout=min(5,a.timeout)) as sock:
        sock.sendall(f'{a.device} {cmd}\r'.encode('ascii'))
        while b'\r' not in data:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError(f'{cmd}: reply timeout')
            sock.settimeout(remaining);part=sock.recv(4096)
            if not part:raise RuntimeError(f'{cmd}: connection closed before reply terminator')
            data.extend(part)
            if len(data)>65536:raise RuntimeError('Oversized reply')
    reply=bytes(data).split(b'\r',1)[0].decode(errors='replace').strip()
    if re.search(r'^\s*Error:',reply,re.M):raise RuntimeError(f'{cmd}: {reply}')
    return reply


def plan(saved,sensor):
    x,y,w,h=saved['roi'];fw,fh=sensor
    if x+w>fw or y+h>fh:raise ValueError('Saved ROI lies outside current sensor')
    ox=x-(fw-w)//2;oy=y-(fh-h)//2
    # Exactly the GUI sequence: reset centred offsets before changing dimensions.
    return ['offset 0 0',f'size {w} {h}',f'offset {ox} {oy}',f'exp {saved["exposure"]:.9g}']


def verify(observed,saved):
    if observed['roi']!=saved['roi']:raise RuntimeError(f'ROI mismatch: requested {saved["roi"]}, reported {observed["roi"]}')
    # Server prints exposure to five decimal places, not full SDK precision.
    if abs(observed['exposure']-saved['exposure'])>5.1e-6:raise RuntimeError(f'Exposure mismatch: requested {saved["exposure"]}, reported {observed["exposure"]}')


def verify_frames(stream,roi,timeout):
    import ImageStreamIOWrap
    image=ImageStreamIOWrap.Image()
    try:
        rc=image.open(stream)
        if rc not in (None,0):raise RuntimeError(f'Cannot attach {stream}: {rc}')
        def counter():
            md=image.md
            if not hasattr(md,'cnt0'):md=md[0]
            return int(md.cnt0),bool(md.write)
        last,_=counter();seen=0;end=time.monotonic()+timeout
        while time.monotonic()<end:
            count,writing=counter()
            if count<last:raise RuntimeError('Frame counter reset')
            if count>last and not writing:
                import numpy as np
                frame=np.array(image,copy=True).T
                after,writing=counter()
                if after==count and not writing:
                    if frame.shape!=tuple(roi[2:][::-1]):raise RuntimeError('Shared-memory frame shape differs from restored ROI')
                    last=count;seen+=1
                    if seen>=3:return
            time.sleep(.001)
        raise TimeoutError('Did not receive three fresh stable frames')
    finally:image.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path,help='Registration directory, registration.json, scan.json, or basis.fits')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=18010)
    p.add_argument('--device',default='dcam0');p.add_argument('--timeout',type=float,default=30.)
    p.add_argument('--dry-run',action='store_true',help='Print commands without connecting')
    p.add_argument('--leave-stopped',action='store_true',help='Restore settings without starting acquisition')
    p.add_argument('--verify-frames',action='store_true',help='Also check three fresh SHM frames; needs ImageStreamIOWrap/numpy')
    p.add_argument('--stream',default='dcam0_data')
    a=p.parse_args()
    if not math.isfinite(a.timeout) or a.timeout<=0:p.error('Timeout must be positive and finite')
    if a.verify_frames and a.leave_stopped:p.error('--verify-frames requires acquisition running')
    path,saved=load_camera(a.config)
    print(f'Source: {path}\nSensor ROI [x,y,w,h]: {saved["roi"]}; exposure {saved["exposure"]:.9g} s')
    print('Restores ROI/exposure only; readout mode, trigger, gain and other unsaved settings must match separately.')
    if a.dry_run:
        if saved['sensor'] is None:raise ValueError('Offline plan needs Format size in saved camera.raw')
        print('\n'.join(['stop (if BUSY)']+plan(saved,saved['sensor'])+([] if a.leave_stopped else ['run'])+['st']))
        return
    current=status(send(a,'st'))
    if saved['sensor'] and current['sensor']!=saved['sensor']:raise ValueError('Current sensor dimensions differ from saved camera')
    commands=plan(saved,current['sensor'])
    if current['capture'] not in ('BUSY','IDLE','READY'):raise RuntimeError(f'Unexpected capture state: {current["capture"]}; open/recover camera first')
    try:
        if current['capture']=='BUSY':send(a,'stop')
        stopped=status(send(a,'st'))
        if stopped['capture']=='BUSY':raise RuntimeError('Camera still BUSY after stop')
        for cmd in commands:
            print('Sending:',cmd,flush=True);send(a,cmd)
        verify(status(send(a,'st')),saved)
        if not a.leave_stopped:send(a,'run')
        observed=status(send(a,'st'));verify(observed,saved)
        if not a.leave_stopped and observed['capture']!='BUSY':raise RuntimeError(f'Capture did not start: {observed["capture"]}')
        if a.verify_frames:verify_frames(a.stream,saved['roi'],max(a.timeout,5*saved['exposure']+2))
        print(f'Restored and readback verified. Capture: {observed["capture"]}.')
        print('Three fresh stable frames verified.' if a.verify_frames else 'Fresh frames were not checked; confirm live GUI updates or use --verify-frames.')
    except Exception:
        print('Restore failed: settings may be partially applied. No automatic retry, rollback or server restart.')
        raise

if __name__=='__main__':main()
