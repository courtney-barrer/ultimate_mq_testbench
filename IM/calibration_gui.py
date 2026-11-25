#!/usr/bin/env python3
"""Lightweight calibration dashboard. Install beside the five existing IM scripts.
Calibration runs as subprocesses; live reconstruction in a separate 1 Hz worker.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse, base64, json, signal, sys, time
from pathlib import Path
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets as W
import pyqtgraph as pg


def repo_data(scripts):
    for start in (scripts.resolve(),Path.cwd().resolve()):
        for parent in (start,*start.parents):
            if (parent/'.git').exists():return parent/'data'
    raise ValueError('No Git repository found. Launch with --data-dir /explicit/output/path')


def stamp():return time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}'


def decode_thumb(info):
    return np.frombuffer(base64.b64decode(info['data']),np.uint8).reshape(info['shape']).copy()


class Window(W.QMainWindow):
    def __init__(self,args):
        super().__init__();self.args=args;self.scripts=args.scripts.resolve();self.data=args.data_dir or repo_data(self.scripts)
        self.data.mkdir(parents=True,exist_ok=True)
        self.proc=None;self.job=None;self.after=None;self.live=False;self.buffer='';self.live_ready=False;self.lastframe=0
        self.paths={};self.actionbuttons=[];self.sliders=[];self.labels=[];self.ids=[];self.plotstamp=None;self.plotnames=[];self.plotdir=None
        self.setWindowTitle('SH-WFS calibration & live reconstruction');self.resize(1240,920)
        body=W.QWidget();self.setCentralWidget(body);outer=W.QVBoxLayout(body)
        title=W.QLabel('SH-WFS calibration');title.setStyleSheet('font-size:22px;font-weight:bold');outer.addWidget(title)
        self.status=W.QLabel('Ready. Start the camera in its normal GUI. Only one calibration/live job runs at a time.');self.status.setWordWrap(True);outer.addWidget(self.status)
        self.setup=W.QWidget();setup_layout=W.QVBoxLayout(self.setup);setup_layout.setContentsMargins(0,0,0,0);outer.addWidget(self.setup)
        form=W.QGridLayout()
        for row,(key,label) in enumerate([('registration','Registration JSON'),('basis','Basis FITS'),('im','Interaction matrix'),('reconstructor','Reconstructor FITS')]):
            edit=W.QLineEdit();edit.setPlaceholderText('Select an existing file or run the corresponding step');self.paths[key]=edit
            browse=W.QPushButton('Browse…');browse.clicked.connect(lambda _,k=key:self.browse(k));form.addWidget(W.QLabel(label),row,0);form.addWidget(edit,row,1);form.addWidget(browse,row,2)
            self.actionbuttons.append(browse)
        setup_layout.addLayout(form)
        options=W.QHBoxLayout();self.family=W.QComboBox();self.family.addItems(['Fourier · compact 24 seeds','Zernike · 15 seeds']);options.addWidget(self.family)
        options.addWidget(W.QLabel('IM modes (first N):'));self.modecount=W.QSpinBox();self.modecount.setRange(1,1000);self.modecount.setValue(12);options.addWidget(self.modecount)
        self.exploratory=W.QCheckBox('Allow unvalidated registration');options.addWidget(self.exploratory);options.addStretch();setup_layout.addLayout(options)
        row=W.QHBoxLayout()
        for text,fn in [('1  Register SLM',self.register),('2  Build basis',self.basis),('3  Build IM + reconstructor',self.build),('Analyse selected IM',self.analyse),('Restore camera',self.restore)]:
            button=W.QPushButton(text);button.clicked.connect(lambda _,f=fn:self.guard(f));row.addWidget(button);self.actionbuttons.append(button)
        setup_layout.addLayout(row)
        self.tabs=W.QTabWidget();outer.addWidget(self.tabs,1)
        calibration=W.QWidget();cal=W.QVBoxLayout(calibration)
        self.prompt=W.QLabel('Script prompts will appear below. Use Continue / Yes / No only when requested.');self.prompt.setWordWrap(True);cal.addWidget(self.prompt)
        answerrow=W.QHBoxLayout();self.answerbuttons=[]
        for text,value in [('Continue / Enter','\n'),('Yes','y\n'),('No','n\n')]:
            b=W.QPushButton(text);b.clicked.connect(lambda _,v=value:self.reply(v));answerrow.addWidget(b);self.answerbuttons.append(b)
        self.stopbutton=W.QPushButton('Stop safely');self.stopbutton.clicked.connect(self.stop);answerrow.addWidget(self.stopbutton);cal.addLayout(answerrow)
        self.plots=W.QComboBox();self.plots.currentIndexChanged.connect(self.load_plot);cal.addWidget(self.plots)
        self.picture=W.QLabel('Diagnostic plots appear here during and after calibration.');self.picture.setAlignment(QtCore.Qt.AlignCenter);self.picture.setMinimumHeight(230);cal.addWidget(self.picture,1)
        self.log=W.QPlainTextEdit();self.log.setReadOnly(True);self.log.document().setMaximumBlockCount(800);self.log.setMaximumHeight(170);cal.addWidget(self.log)
        self.tabs.addTab(calibration,'Calibration & diagnostics')
        livewidget=W.QWidget();lv=W.QVBoxLayout(livewidget)
        note=W.QLabel('Open-loop monitor · ~1 Hz · saved dark + fresh flat reference · close other SLM controllers before starting');note.setWordWrap(True);lv.addWidget(note)
        lr=W.QHBoxLayout();self.startlive=W.QPushButton('Launch live reconstructor');self.startlive.clicked.connect(lambda:self.guard(self.start_live));lr.addWidget(self.startlive)
        self.stoplive=W.QPushButton('Stop live / restore flat');self.stoplive.clicked.connect(self.stop);lr.addWidget(self.stoplive)
        self.flat=W.QPushButton('Flat SLM');self.flat.clicked.connect(self.set_flat);lr.addWidget(self.flat)
        self.reference=W.QPushButton('New flat reference');self.reference.clicked.connect(self.new_reference);lr.addWidget(self.reference)
        self.previews=W.QCheckBox('Show camera + probe previews');self.previews.toggled.connect(self.preview_toggle);lr.addWidget(self.previews);lv.addLayout(lr)
        self.live_status=W.QLabel('Stopped. No live frames saved; no automatic correction is applied.');self.live_status.setWordWrap(True);lv.addWidget(self.live_status)
        self.sliderform=W.QGridLayout();lv.addLayout(self.sliderform)
        views=W.QHBoxLayout();self.views={};self.images={};self.viewtitles={}
        for key,title in [('camera','Camera [ADU]'),('probe','Applied mode combination [grey]'),('reconstruction','Reconstruction [grey]')]:
            group=W.QGroupBox(title);layout=W.QVBoxLayout(group);plot=pg.PlotWidget();plot.setAspectLocked(True);plot.invertY(True);plot.hideAxis('left');plot.hideAxis('bottom')
            image=pg.ImageItem(axisOrder='row-major');plot.addItem(image)
            if key!='camera':
                cmap=pg.ColorMap([0,.5,1],[(35,70,160),(245,245,245),(175,35,40)]);image.setLookupTable(cmap.getLookupTable(nPts=256))
            layout.addWidget(plot);scale=W.QLabel('');layout.addWidget(scale)
            views.addWidget(group);self.views[key]=group;self.images[key]=image;self.viewtitles[key]=scale
        lv.addLayout(views,1);self.coeffplot=pg.PlotWidget(title='Recovered mode coefficients [grey RMS]');self.coeffplot.setMaximumHeight(170);lv.addWidget(self.coeffplot)
        self.bars=pg.BarGraphItem(x=[0],height=[0],width=.7,brush='#497abd');self.coeffplot.addItem(self.bars)
        self.tabs.addTab(livewidget,'Live reconstruction');self.preview_toggle(False)
        self.tabs.currentChanged.connect(lambda index:self.setup.setVisible(index==0))
        self.refresh=QtCore.QTimer(self);self.refresh.timeout.connect(self.tick);self.refresh.start(1000)
        self.busy(False)
        for key in self.paths:
            value=getattr(args,key,None)
            if value:self.paths[key].setText(str(value.resolve()))
    def guard(self,fn):
        try:fn()
        except Exception as e:W.QMessageBox.warning(self,'Cannot start',str(e))
    def path(self,key):
        p=Path(self.paths[key].text()).expanduser()
        if not p.is_file():raise ValueError(f'Select an existing {key} file')
        return p.resolve()
    def browse(self,key):
        path,_=W.QFileDialog.getOpenFileName(self,f'Select {key}',str(self.data),'JSON (*.json)' if key=='registration' else 'FITS (*.fits *.fit)')
        if path:self.paths[key].setText(path)
    def busy(self,yes):
        for b in self.actionbuttons:b.setEnabled(not yes)
        for e in self.paths.values():e.setEnabled(not yes)
        self.family.setEnabled(not yes);self.modecount.setEnabled(not yes);self.exploratory.setEnabled(not yes)
        self.startlive.setEnabled(not yes);self.stopbutton.setEnabled(yes);self.stoplive.setEnabled(yes and self.live)
        for b in self.answerbuttons:b.setEnabled(yes and not self.live)
        self.flat.setEnabled(yes and self.live and self.live_ready);self.reference.setEnabled(yes and self.live and self.live_ready)
        for s in self.sliders:s.setEnabled(yes and self.live and self.live_ready)
    def launch(self,script,arguments,plotdir=None,after=None,live=False):
        if self.proc:raise ValueError('Wait for the current task to finish')
        if self.args.simulate and script!='live_reconstructor.py':raise ValueError('Calibration actions are disabled in simulation mode')
        target=Path(__file__).with_name(script) if script=='live_reconstructor.py' else self.scripts/script
        if not target.is_file():raise ValueError(f'Missing script: {target}')
        self.live=live;self.live_ready=False;self.after=after;self.buffer='';self.plotdir=plotdir;self.job=script
        self.proc=QtCore.QProcess(self);self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        env=QtCore.QProcessEnvironment.systemEnvironment()
        for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env.insert(k,'1')
        env.insert('PYTHONUNBUFFERED','1');self.proc.setProcessEnvironment(env);self.proc.setWorkingDirectory(str(self.scripts))
        self.proc.readyReadStandardOutput.connect(self.read_output);self.proc.finished.connect(self.finished);self.proc.errorOccurred.connect(self.process_error)
        self.log.appendPlainText('Starting '+script+' '+' '.join(map(str,arguments)));self.status.setText('Running '+script);self.busy(True)
        self.proc.start(sys.executable,['-u',str(target),*map(str,arguments)])
    def flags(self):return ['--allow-unvalidated'] if self.exploratory.isChecked() else []
    def register(self):
        out=self.data/('slm_registration_gui_'+stamp())
        def done():
            if (out/'registration.json').exists():self.paths['registration'].setText(str(out/'registration.json'))
            else:self.status.setText('Registration ended without a config. Review log and plots.')
        self.launch('register_slm.py',['--positions','32','--frames-per-probe','8','--screen',self.args.screen,'--output',out],out,done)
    def basis(self):
        reg=self.path('registration');out=self.data/('slm_basis_gui_'+stamp())
        extra=['--basis','fourier','--layout','polar','--frequencies','1','2','4','--orientations','0','45','90','135'] if self.family.currentIndex()==0 else ['--basis','zernike','--modes','15']
        self.launch('build_slm_basis.py',[reg,*extra,'--buffer-pixels','16','--taper-pixels','8','--output',out,*self.flags()],out,lambda:self.paths['basis'].setText(str(out/'basis.fits')))
    def build(self):
        from astropy.io import fits
        basis=self.path('basis');h=fits.getheader(basis);n=min(self.modecount.value(),int(h['NMODES']));kind=h['BASISTYP']
        out=self.data/f'IM_{kind}_{stamp()}.fits'
        def done():self.paths['im'].setText(str(out));self.analyse()
        self.launch('build_IM.py',[basis,'--modes',*range(n),'--amplitudes','5','--frames-per-probe','16','--screen',self.args.screen,'--output',out,*self.flags()],self.data,done)
    def analyse(self):
        im=self.path('im');out=self.data/('IM_analysis_gui_'+stamp())
        self.launch('analyse_IM.py',[im,'--rcond','.02','--output',out],out,lambda:self.paths['reconstructor'].setText(str(out/'reconstructor.fits')))
    def restore(self):
        key='basis' if self.paths['basis'].text().strip() else 'registration'
        self.launch('restore_camera.py',[self.path(key),'--verify-frames'])
    def reply(self,value):
        if self.proc and not self.live:self.proc.write(value.encode());self.log.appendPlainText('> '+value.strip());self.prompt.setText('Response sent; waiting for script…')
    def read_output(self):
        if not self.proc:return
        chunk=bytes(self.proc.readAllStandardOutput()).decode(errors='replace')
        if self.live:
            self.buffer+=chunk
            while '\n' in self.buffer:
                line,self.buffer=self.buffer.split('\n',1)
                try:self.live_event(json.loads(line))
                except (ValueError,KeyError,TypeError):self.log.appendPlainText(line[-3000:])
            if len(self.buffer)>2_000_000:self.buffer='';self.live_status.setText('Worker message too large; discarded')
        else:
            cursor=self.log.textCursor();cursor.movePosition(QtGui.QTextCursor.End);cursor.insertText(chunk);self.log.setTextCursor(cursor)
            self.buffer=(self.buffer+chunk)[-4000:]
            tail=self.buffer.rstrip().split('\n')[-1]
            self.prompt.setText(tail[-700:] if tail else 'Working…')
    def process_error(self,error):
        if error==QtCore.QProcess.FailedToStart:
            self.status.setText('Failed to start process');self.after=None;self.finished(-1,QtCore.QProcess.CrashExit)
    def finished(self,code,status):
        after=self.after;waslive=self.live;self.after=None
        if self.proc:self.proc.deleteLater()
        self.proc=None;self.live=False;self.live_ready=False;self.busy(False)
        self.status.setText('Finished' if code==0 else f'Task stopped/failed (exit {code}). See log; partial data are retained.')
        if waslive:self.live_status.setText('Live stopped. Review log for any cleanup warnings.');self.clear_live()
        self.tick()
        if code==0 and after:self.guard(after)
    def send(self,obj):
        if self.proc and self.live:self.proc.write((json.dumps(obj)+'\n').encode())
    def stop(self):
        if not self.proc:return
        self.after=None;self.status.setText('Stopping safely… waiting for the current operation and SLM cleanup.');self.stopbutton.setEnabled(False);self.stoplive.setEnabled(False)
        if self.live:self.send({'action':'stop'})
        else:
            try:os.kill(int(self.proc.processId()),signal.SIGINT)
            except ProcessLookupError:pass
    def start_live(self):
        arguments=['--scripts',self.scripts,'--hz','1','--frames','4','--screen',self.args.screen]
        if self.args.simulate:arguments+=['--simulate']
        else:
            arguments+=['--basis',self.path('basis'),'--im',self.path('im'),'--reconstructor',self.path('reconstructor'),*self.flags()]
            result=W.QMessageBox.question(self,'Start live reconstruction','Laser ON and stable; camera running with calibration settings?\nClose other SLM controllers. Starting will display flat and acquire a fresh reference.\nThe saved calibration dark is reused.')
            if result!=W.QMessageBox.Yes:return
        self.clear_live();self.tabs.setCurrentIndex(1);self.launch('live_reconstructor.py',arguments,live=True)
    def clear_live(self):
        for im in self.images.values():im.clear()
        self.bars.setOpts(x=[0],height=[0]);self.lastframe=0
    def preview_toggle(self,enabled):
        for key in ('camera','probe'):self.views[key].setVisible(enabled)
        self.send({'action':'preview','enabled':enabled})
    def sliders_set(self,values):
        for i,(s,label) in enumerate(zip(self.sliders,self.labels)):
            s.blockSignals(True);s.setValue(round(values[i]*100));s.blockSignals(False);label.setText(f'{values[i]:+.2f}')
    def set_flat(self):
        self.sliders_set(np.zeros(len(self.sliders)));self.apply_sliders()
    def new_reference(self):
        if W.QMessageBox.question(self,'New flat reference','Display flat SLM and redefine zero using the current wavefront?')==W.QMessageBox.Yes:
            self.sliders_set(np.zeros(len(self.sliders)));self.send({'action':'reference'})
    def apply_sliders(self):
        if self.live_ready:self.send({'action':'set','coefficients':[s.value()/100 for s in self.sliders]})
    def live_event(self,d):
        event=d['event']
        if event=='ready':
            while self.sliderform.count():
                item=self.sliderform.takeAt(0)
                if item.widget():item.widget().deleteLater()
            self.ids=d['ids'];self.sliders=[];self.labels=[]
            for i in range(d['ncontrol']):
                label=W.QLabel('+0.00');s=W.QSlider(QtCore.Qt.Horizontal);s.setRange(-int(d['limit']*100),int(d['limit']*100));s.setSingleStep(10)
                s.valueChanged.connect(lambda v,l=label:l.setText(f'{v/100:+.2f}'));s.sliderReleased.connect(self.apply_sliders)
                # Keyboard/wheel also apply after a short debounce, not per event.
                timer=QtCore.QTimer(s);timer.setSingleShot(True);timer.setInterval(250);timer.timeout.connect(lambda sl=s:self.apply_sliders() if not sl.isSliderDown() else None)
                s.valueChanged.connect(lambda _,t=timer:t.start())
                self.sliderform.addWidget(W.QLabel(f'Mode {self.ids[i]} [grey RMS]'),i,0);self.sliderform.addWidget(s,i,1);self.sliderform.addWidget(label,i,2)
                self.sliders.append(s);self.labels.append(label)
            self.live_ready=True;self.busy(True);self.send({'action':'preview','enabled':self.previews.isChecked()})
            self.live_status.setText(d['message'])
        elif event=='frame':
            self.lastframe=time.monotonic()
            for key in self.images:
                if key in d:
                    info=d[key];self.images[key].setImage(decode_thumb(info),autoLevels=False,levels=(0,255));self.viewtitles[key].setText(f'Display range {info["lo"]:.3g} … {info["hi"]:.3g}')
            coeff=d['coefficients'];self.bars.setOpts(x=np.arange(len(coeff)),height=coeff,width=.7)
            self.coeffplot.getAxis('bottom').setTicks([list(enumerate(map(str,self.ids)))])
            self.live_status.setText(f'{d["message"]} | slope RMS {d["slope_rms"]:.4g} px | residual RMS {d["residual_rms"]:.4g} px | tick {d["elapsed"]:.2f} s')
        elif event in ('applied','rejected'):
            self.sliders_set(d['coefficients'])
            if event=='rejected':self.live_status.setText(d['message'])
        elif event=='reference':self.sliders_set(np.zeros(len(self.sliders)));self.live_status.setText(d['message'])
        elif event in ('invalid','error'):
            self.clear_live();self.live_status.setText(d['message']);self.log.appendPlainText(d['message'])
        else:self.live_status.setText(d.get('message',event))
    def tick(self):
        if self.live:
            if self.lastframe and time.monotonic()-self.lastframe>3:self.live_status.setText('STALE: no reconstruction received for >3 s; current operation may still be running.')
            return
        if not self.plotdir or not self.plotdir.exists():return
        # Only list filenames; image contents are read on selection/mtime change.
        names=sorted(self.plotdir.glob('*.png'),key=lambda p:p.stat().st_mtime)[-40:]
        if names!=self.plotnames:
            old=self.plots.currentText();self.plotnames=names;self.plots.blockSignals(True);self.plots.clear();self.plots.addItems([p.name for p in names]);self.plots.blockSignals(False)
            self.plots.setCurrentIndex(len(names)-1);self.load_plot()
        elif self.plots.currentIndex()>=0:
            p=names[self.plots.currentIndex()]
            if (p,p.stat().st_mtime)!=self.plotstamp:self.load_plot()
    def load_plot(self):
        i=self.plots.currentIndex()
        if i<0 or i>=len(self.plotnames):return
        p=self.plotnames[i];reader=QtGui.QImageReader(str(p));size=reader.size();size.scale(1050,500,QtCore.Qt.KeepAspectRatio);reader.setScaledSize(size);im=reader.read()
        if not im.isNull():self.picture.setPixmap(QtGui.QPixmap.fromImage(im));self.plotstamp=(p,p.stat().st_mtime)
    def closeEvent(self,event):
        if self.proc:
            self.stop();W.QMessageBox.information(self,'Stopping','Waiting for the worker to finish safely. Close the window again once it has stopped.');event.ignore()
        else:event.accept()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scripts',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--data-dir',type=Path);p.add_argument('--screen',default='1');p.add_argument('--simulate',action='store_true',help='Hardware-free live preview; calibration actions are disabled')
    for key in ('registration','basis','im','reconstructor'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();pg.setConfigOptions(imageAxisOrder='row-major',antialias=False)
    app=W.QApplication(sys.argv[:1]);win=Window(a);win.show();return app.exec_()

if __name__=='__main__':sys.exit(main())
