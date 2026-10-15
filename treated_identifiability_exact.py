"""Exact Taylor-jet rank certificates for the treated first inter-dose interval.

Model assumptions used in the manuscript extension:
  * u = V^(1/3)
  * fixed Hill exponent n = 2
  * K = EC50^2
  * first inter-dose interval after a known positive bolus C(0)=D

Three certificates are produced:
  1) V only with C(t) regarded as a known nonconstant input: rank 8/8
  2) joint V,C with k unknown: rank 9/9
  3) joint V,E,C: rank 11/11

All arithmetic in the jet and parameter sensitivities is exact rational arithmetic.
Only SymPy is required for the final exact matrix-rank and determinant calculations.
"""
from fractions import Fraction as F
import math
import sympy as sp

class Dual:
    __slots__=("v","g")
    def __init__(self,v,g): self.v=F(v); self.g=tuple(F(x) for x in g)
    def __add__(self,o):
        o=asdual(o,len(self.g)); return Dual(self.v+o.v,[a+b for a,b in zip(self.g,o.g)])
    __radd__=__add__
    def __neg__(self): return Dual(-self.v,[-x for x in self.g])
    def __sub__(self,o): return self+(-asdual(o,len(self.g)))
    def __rsub__(self,o): return asdual(o,len(self.g))-self
    def __mul__(self,o):
        o=asdual(o,len(self.g)); return Dual(self.v*o.v,[self.v*b+o.v*a for a,b in zip(self.g,o.g)])
    __rmul__=__mul__
    def inv(self): return Dual(1/self.v,[-x/(self.v*self.v) for x in self.g])
    def __truediv__(self,o): return self*asdual(o,len(self.g)).inv()
    def __rtruediv__(self,o): return asdual(o,len(self.g))*self.inv()

def asdual(x,P): return x if isinstance(x,Dual) else Dual(F(x),[F(0)]*P)
def z(P): return asdual(0,P)
def o(P): return asdual(1,P)
def parameter(v,i,P):
    g=[F(0)]*P; g[i]=F(1); return Dual(F(v),g)

def add(A,B,N):
    P=len(A[0].g); return [(A[i] if i<len(A) else z(P))+(B[i] if i<len(B) else z(P)) for i in range(N+1)]
def neg(A,N):
    P=len(A[0].g); return [-(A[i] if i<len(A) else z(P)) for i in range(N+1)]
def sub(A,B,N): return add(A,neg(B,N),N)
def mul(A,B,N):
    P=len(A[0].g); R=[z(P) for _ in range(N+1)]
    for i in range(min(len(A),N+1)):
        for j in range(min(len(B),N+1-i)): R[i+j]=R[i+j]+A[i]*B[j]
    return R
def scale(A,c,N):
    P=len(A[0].g); return [(A[i] if i<len(A) else z(P))*c for i in range(N+1)]
def inv(A,N):
    P=len(A[0].g); R=[z(P) for _ in range(N+1)]; R[0]=A[0].inv()
    for n in range(1,N+1):
        s=z(P)
        for j in range(1,min(n,len(A)-1)+1): s=s+A[j]*R[n-j]
        R[n]=-(R[0]*s)
    return R
def div(A,B,N): return mul(A,inv(B,N),N)
def power(A,n,N):
    P=len(A[0].g); R=[o(P)]+[z(P)]*N; B=A
    while n:
        if n&1: R=mul(R,B,N)
        B=mul(B,B,N); n//=2
    return R
def const(d,P,N): return [d]+[z(P)]*N

def matrix(rows):
    return sp.Matrix([[sp.Rational(x.numerator,x.denominator) for x in row] for row in rows])
def certificate(M):
    rank=M.rank(); piv=list(M.T.rref()[1][:rank]); minor=M[piv,:]
    det=sp.factor(minor.det()) if minor.rows==minor.cols else None
    return rank,piv,det

def known_exposure():
    names=['a','h','delta','qp','beta0','q0','Emax','K']; P=len(names)
    values=[F(2),F(1),F(1),F(2),F(1),F(1),F(3,2),F(1)]
    p={n:parameter(v,i,P) for i,(n,v) in enumerate(zip(names,values))}
    u=[asdual(1,P)]; beta=[p['beta0']]
    # Known profile C(t)=2 exp(-t); no parameter sensitivity is attached to C.
    C=[asdual(F(2*((-1)**j),math.factorial(j)),P) for j in range(9)]
    rows=[]
    for m in range(8):
        N=m; us=u+[z(P)]*(N+1-len(u)); bs=beta+[z(P)]*(N+1-len(beta)); Cs=C[:N+1]
        C2=power(Cs,2,N); Q=sub(const(o(P),P,N),scale(div(C2,add(const(p['K'],P,N),C2,N),N),p['Emax'],N),N)
        fu=scale(sub(scale(Q,p['a'],N),mul(bs,us,N),N),F(1,3),N)
        u3=power(us,3,N); rec=div(u3,add(u3,const(p['h'],P,N),N),N)
        fb=sub(add(const(p['q0'],P,N),scale(rec,p['qp'],N),N),scale(bs,p['delta'],N),N)
        u.append(fu[m]/F(m+1)); beta.append(fb[m]/F(m+1)); rows.append(list(u[m+1].g))
    M=matrix(rows); return names,M,certificate(M)

def reduced_observed_C():
    names=['a','h','delta','qp','beta0','q0','k','Emax','K']; P=len(names)
    values=[F(2),F(1),F(1),F(2),F(1),F(1),F(1),F(3,2),F(1)]
    p={n:parameter(v,i,P) for i,(n,v) in enumerate(zip(names,values))}
    u=[asdual(1,P)]; beta=[p['beta0']]; C=[asdual(2,P)]; rows=[]
    for m in range(8):
        N=m; us=u+[z(P)]*(N+1-len(u)); bs=beta+[z(P)]*(N+1-len(beta)); Cs=C+[z(P)]*(N+1-len(C))
        C2=power(Cs,2,N); Q=sub(const(o(P),P,N),scale(div(C2,add(const(p['K'],P,N),C2,N),N),p['Emax'],N),N)
        fu=scale(sub(scale(Q,p['a'],N),mul(bs,us,N),N),F(1,3),N)
        u3=power(us,3,N); rec=div(u3,add(u3,const(p['h'],P,N),N),N)
        fb=sub(add(const(p['q0'],P,N),scale(rec,p['qp'],N),N),scale(bs,p['delta'],N),N)
        fC=scale(Cs,-p['k'],N)
        u.append(fu[m]/F(m+1)); beta.append(fb[m]/F(m+1)); C.append(fC[m]/F(m+1))
        rows.extend([list(u[m+1].g),list(C[m+1].g)])
    M=matrix(rows); return names,M,certificate(M)

def joint_observation():
    names=['a','b','rho','s','p','h','delta','E0','k','Emax','K']; P=len(names)
    values=[F(2),F(1),F(1),F(1),F(2),F(1),F(1),F(1),F(1),F(3,2),F(1)]
    q={n:parameter(v,i,P) for i,(n,v) in enumerate(zip(names,values))}
    u=[asdual(1,P)]; E=[q['E0']]; C=[asdual(2,P)]; rows=[list(E[0].g)]
    for m in range(5):
        N=m; us=u+[z(P)]*(N+1-len(u)); Es=E+[z(P)]*(N+1-len(E)); Cs=C+[z(P)]*(N+1-len(C))
        C2=power(Cs,2,N); Q=sub(const(o(P),P,N),scale(div(C2,add(const(q['K'],P,N),C2,N),N),q['Emax'],N),N)
        loss=add(const(q['b'],P,N),scale(Es,q['rho'],N),N)
        fu=scale(sub(scale(Q,q['a'],N),mul(loss,us,N),N),F(1,3),N)
        u3=power(us,3,N); rec=div(u3,add(u3,const(q['h'],P,N),N),N)
        fE=sub(add(const(q['s'],P,N),scale(rec,q['p'],N),N),scale(Es,q['delta'],N),N)
        fC=scale(Cs,-q['k'],N)
        u.append(fu[m]/F(m+1)); E.append(fE[m]/F(m+1)); C.append(fC[m]/F(m+1))
        rows.extend([list(u[m+1].g),list(E[m+1].g),list(C[m+1].g)])
    M=matrix(rows); return names,M,certificate(M)

def show(label,fn):
    names,M,(rank,piv,det)=fn()
    print(label)
    print('parameters:', ', '.join(names))
    print(f'rank: {rank}/{len(names)}')
    print('independent row indices:', piv)
    print('exact determinant of displayed full-rank minor:', det)
    print()

if __name__=='__main__':
    show('Known C(t), V observed',known_exposure)
    show('V and C observed, k unknown',reduced_observed_C)
    show('V, E and C jointly observed',joint_observation)
