"""
Crease error and validity on lattice/body junctions of the arm link
(examples/fields/arm-link-graded.json with its FE stress swapped for a cheap
stand-in, so meshing runs alone; junction "hard" drops the fillet between
skin, collars and lattice, so the walls meet the skin in a crease).

Meshes the part's organic leaf (sharp.py's extended field, which the exact
bores are later cut from) plainly and with mesh.creases, and measures the
true crease curve against each (crease_bench.true_crease), binned by the
angle between the two surfaces' normals; crease points the plain mesh is
over 2 pitch from (pores and islands it did not resolve) are left out.

    python bench/crease_lattices.py gyroid|octet hard|soft PITCH
Writes arm_<core>_<junction>_<pitch>.json (and the reference) in the cwd.
Results of 2026-10-31: bench/results/crease_lattices.json.
"""

import copy, json
BASE=json.load(open(__import__('pathlib').Path(__file__).resolve().parents[1] / 'examples/fields/arm-link-graded.json'))
def doc(core='gyroid', junction='hard'):
    d=copy.deepcopy(BASE); n=d['nodes']
    n['stress']={"op":"remap","input":"nearBore","from":[3,60],"to":[0.5,0.05]}
    if core=='octet':
        n['core']={"op":"lattice","type":"octet","cell":"cell","thickness":"wall","blend":0.4}
        n['wall']['to']=[1.0,2.4]
    if junction=='hard':
        n['joined'].pop('blend',None)
    return d


import sys, time, json, numpy as np, trimesh
ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'moonshots' / 'round2'))
import crease_bench as cb
from loadpath.fields import FieldGraph, field_mesh
from loadpath.sharp import plan, _dont_care, _Extended, mesh_distance
from loadpath.crease import CreaseTree
core,j,p=sys.argv[1],sys.argv[2],float(sys.argv[3])
g=FieldGraph(doc(core,j)); pl=plan(g)
(leaf,regions),=list(_dont_care(pl))
fld=_Extended(leaf.field,regions,eps=p,core=p)
lo,hi=g.bounds()
tree=CreaseTree.from_graph(g,leaf.names[0]); margin=fld.eps+4*p
tree.ignore=lambda P: fld.g(np.asarray(P,float))<margin
print(core,j,p,'leaves',[l.name for l in tree.leaves],flush=True)
key=f'arm_{core}_{j}_{p}'
import os
if os.path.exists(key+'_ref.npz'):
    R=np.load(key+'_ref.npz'); X,pairs,phi=R['X'],R['pairs'],R['phi']
else:
    t=time.time(); fine=field_mesh(fld,lo,hi,pitch=p/1.5)
    X,pairs=cb.true_crease(tree,fine,p/1.5,spacing=p/4); del fine
    keep=~tree.ignore(X); X,pairs=X[keep],pairs[keep]
    phi=cb.true_angle(tree,X,pairs); np.savez(key+'_ref.npz',X=X,pairs=pairs,phi=phi); print('ref',len(X),time.time()-t,flush=True)
names=[l.name for l in tree.leaves]
up,cnt=np.unique(pairs,axis=0,return_counts=True); print('pairs',[(names[a],names[b],c) for (a,b),c in zip(up,cnt)])
bins=[0,60,90,120,135,150,180]; out={}
for cr in (False,True):
    t=time.time(); m=field_mesh(fld,lo,hi,pitch=p,creases=tree if cr else None); dt=time.time()-t
    V,F=np.asarray(m.vertices),np.asarray(m.faces)
    print("meshed",dt,len(F),flush=True); d=mesh_distance(m,X,edge=max(p,0.2)); print("dist",flush=True)
    if not cr: res=d<2*p; print('unresolved (plain mesh > 2 pitch away)',int((~res).sum()),'of',len(d),flush=True)
    d=d[res]
    topo=cb.topology(V,F)
    from scipy.spatial import cKDTree
    dc,_=cKDTree(X).query(V[F].mean(1),distance_upper_bound=2*p); region=np.flatnonzero(np.isfinite(dc))
    from loadpath.crease import crossing_faces; sx=int(len(crossing_faces(V,F,region)))
    row={'seconds':round(dt,1),'faces':len(F),'max':float(d.max()),'rms':float(np.sqrt((d**2).mean())),'p95':float(np.percentile(d,95)),
         'watertight':topo['watertight'],'winding':topo['winding'],'manifold3d':topo['manifold3d'],'selfX':sx,'stats':m.metadata.get('creases'),'unresolved':int((~res).sum()),'n':int(len(res))}
    for a,b in zip(bins[:-1],bins[1:]):
        s=((phi>=a)&(phi<b))[res]
        if s.any(): row[f'{a}-{b}']=(int(s.sum()),round(float(d[s].max()),3),round(float(np.sqrt((d[s]**2).mean())),4))
    out['after' if cr else 'before']=row; print('creases' if cr else 'plain',row,flush=True)
    m.export(f'{key}_{int(cr)}.stl')
json.dump(out,open(key+'.json','w'),indent=1)
