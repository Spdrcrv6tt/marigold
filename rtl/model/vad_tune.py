"""Parameter sweep for the VAD (prints miss/FA/noise-only per noise type at 20/10/5/0 dB SNR).
Run: python rtl/model/vad_tune.py ["dict(K_ON=10)" ...]"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vad_eval as ve
import numpy as np
sets = [dict(), dict(K_Z=16), dict(UP_IDLE=4, DOWN=3), dict(K_ON=10, K_OFF=7, K_Z=16), dict(UP_ACT=6, UP_IDLE=4, K_Z=16)]
if len(sys.argv) > 1:
    sets = [eval(a) for a in sys.argv[1:]]
for kw in sets:
    r = ve.evaluate(verbose=False, **kw)
    c = r['cond']
    print(kw)
    for kind in ('white','pink','hum','babble'):
        print('  %-7s' % kind, ' '.join('%5.1f/%5.1f/%5.1f' % (100*c['%s_%d'%(kind,s)]['miss'], 100*c['%s_%d'%(kind,s)]['fa'], 100*c['%s_%d'%(kind,s)]['noise_only']) for s in (20,10,5,0)))
    print('  clean   %5.1f/%5.1f' % (100*c['clean']['miss'], 100*c['clean']['fa']), flush=True)
