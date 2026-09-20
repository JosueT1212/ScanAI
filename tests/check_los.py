"""Mirrors the slab/cast/occluded math in index.html. Fails loudly if the
line-of-sight split changes."""
import math
SENSOR=(50,34); ROOM=dict(x=.6,y=.6,w=98.8,h=66.8)
BENCHES=[dict(x=18,y=22,w=16,h=5),dict(x=74,y=22,w=14,h=5),
         dict(x=20,y=41,w=16,h=5),dict(x=56,y=45,w=13,h=5)]
LOCS={"ELEC-A":(15,17),"ELEC-B":(34,17),"FAB-1":(67,17),"FAB-2":(86,17),
      "STOR-C":(15,52),"STOR-D":(34,52),"CLASS-1":(67,52),"CHRG-1":(86,52)}

def slab(ox,oy,dx,dy,r):
    ix,iy=1/(dx or 1e-9),1/(dy or 1e-9)
    t1,t2=(r["x"]-ox)*ix,(r["x"]+r["w"]-ox)*ix
    tmin,tmax=min(t1,t2),max(t1,t2)
    t1,t2=(r["y"]-oy)*iy,(r["y"]+r["h"]-oy)*iy
    tmin=max(tmin,min(t1,t2)); tmax=min(tmax,max(t1,t2))
    return None if tmax<max(tmin,0) else (tmin,tmax)

def cast(ox,oy,dx,dy):
    best=float("inf")
    o=slab(ox,oy,dx,dy,ROOM)
    if o: best=o[1]
    for b in BENCHES:
        s=slab(ox,oy,dx,dy,b)
        if s and .001<s[0]<best: best=s[0]
    return best

def occluded(p):
    dx,dy=p[0]-SENSOR[0],p[1]-SENSOR[1]; L=math.hypot(dx,dy)
    return cast(*SENSOR,dx/L,dy/L) < L-.5

blind={k for k,v in LOCS.items() if occluded(v)}
assert blind=={"ELEC-A","FAB-2","STOR-C","CLASS-1"}, f"LOS split changed: {sorted(blind)}"

# Every ray must terminate on a real surface, inside the C1's 12 m ceiling.
rs=[cast(*SENSOR,math.sin(math.radians(a*.72)),-math.cos(math.radians(a*.72)))*.2
    for a in range(500)]
assert all(0<r<=12.0 for r in rs), (min(rs),max(rs))
assert sum(r<8 for r in rs)>40, "benches should produce some short returns"
print(f"OK  blind={sorted(blind)}  visible={sorted(set(LOCS)-blind)}")
print(f"    range {min(rs):.2f}-{max(rs):.2f} m over {len(rs)} returns")

# --- fusion pass coverage: mirrors the pass module in index.html ------------
FOV, RANGE_M, MPU = 70, 5.0, .2
R_U = RANGE_M / MPU
COS_HALF = math.cos(math.radians(FOV / 2))
ROUTE = [(10,30),(90,30),(90,46),(10,46),(10,30)]

def sees_from(px, py, t):
    dx, dy = t[0]-px, t[1]-py; L = math.hypot(dx, dy)
    return L and cast(px, py, dx/L, dy/L) >= L - .5

def coverage():
    got = {}
    for i in range(len(ROUTE)-1):
        a, b = ROUTE[i], ROUTE[i+1]
        L = math.hypot(b[0]-a[0], b[1]-a[1]); hx, hy = (b[0]-a[0])/L, (b[1]-a[1])/L
        for s in range(int(L*4)+1):           # 0.25-unit steps
            t = s/4/L
            px, py = a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t
            for name, loc in LOCS.items():
                if name in got: continue
                vx, vy = loc[0]-px, loc[1]-py; ln = math.hypot(vx, vy)
                if not ln or ln > R_U: continue
                ux, uy = vx/ln, vy/ln
                if max(-hy*ux + hx*uy, hy*ux - hx*uy) < COS_HALF: continue
                if not sees_from(px, py, loc): continue
                got[name] = round(ln*MPU, 1)
    return got

cov = coverage()
missed = sorted(set(LOCS) - set(cov))
print(f"\nfusion pass: {len(cov)}/{len(LOCS)} locations confirmed")
for k in sorted(cov): print(f"    {k:8s} at {cov[k]:.1f} m")
print(f"    missed: {missed or 'none'}")
assert 4 <= len(cov) <= 7, f"coverage {len(cov)}/8 makes a poor demo — retune ROUTE/FOV/RANGE"
