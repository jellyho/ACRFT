import collections, sys, pathlib, os
sys.path.insert(0,'/home/rllab4/jellyho/ACRFT-sweep/slurm')
mm = None
try:
    import make_master_report as mm
except FileNotFoundError:
    mm = sys.modules.get('make_master_report')
if mm is None:
    print("import failed hard"); sys.exit(1)
c=collections.Counter(s for _,_,_,s,_ in mm.ENTRIES)
print("total", len(mm.ENTRIES), c)
import sync_hub as sh
for d,e,t,s,b in mm.ENTRIES:
    if s not in ('완결','진행 중','살아있음'):
        p = sh._entry_payload(d,e,t,s,b, mm.EN_BODIES.get(e))
        print('OTHER status=%r eid=%s -> payload status=%r' % (s, e, p['status']))
        print('   title:', t)
