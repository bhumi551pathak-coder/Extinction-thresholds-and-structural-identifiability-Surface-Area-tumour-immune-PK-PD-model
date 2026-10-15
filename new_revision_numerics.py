"""Reproduce numerical additions introduced after the original Zenodo archive.

This script generates the new/extended numerical checks supporting the revised manuscript:
  * Section 6.4 matched-killing comparison (Figure 5)
  * Section 6.4 Emax=1 versus Emax=1.5 comparison (Figure 6)
  * Section 6.5 symmetry and synthetic-noise illustration (Figure 7)
  * near-boundary periodic-average checks cited in Section 6.3

The model is integrated in u = V^(1/3), E. For bolus dosing the PK state is evaluated
analytically between jumps. DOP853 is used for the reported reference calculations.

Run from repository root:
    python scripts/new_revision_numerics.py

Dependencies: numpy, scipy, pandas, matplotlib
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

@dataclass(frozen=True)
class Params:
    a: float = 0.8
    b: float = 0.02
    rho: float = 0.015
    s: float = 0.3
    p: float = 0.5
    h: float = 200.0
    delta: float = 0.2
    k: float = 0.7
    Emax: float = 1.5
    EC50: float = 2.0
    n: float = 2.0

P = Params()
VCELL = 1e-6


def H(C: float | np.ndarray, p: Params=P):
    C = np.asarray(C)
    return C**p.n / (p.EC50**p.n + C**p.n)


def Q(C: float | np.ndarray, p: Params=P):
    return 1.0 - p.Emax * H(C, p)


def ccrit(p: Params=P):
    if p.Emax <= 1:
        return np.inf
    return p.EC50 * (p.Emax - 1.0) ** (-1.0 / p.n)


def qbar(D: float, tau: float, p: Params=P):
    """Closed-form cycle average of Q on the limiting periodic PK orbit."""
    ep = math.exp(-p.k * tau)
    Cplus = D / (1.0 - ep)
    Cminus = Cplus * ep
    return 1.0 - p.Emax / (p.n * p.k * tau) * math.log(
        (p.EC50**p.n + Cplus**p.n) / (p.EC50**p.n + Cminus**p.n)
    )


def rhs_surface_const(t, y, Cc, p=P):
    u, E = y
    upos = max(u, 0.0)
    V = upos**3
    du = (p.a * float(Q(Cc, p)) - (p.b + p.rho * E) * upos) / 3.0
    dE = p.s + p.p * V / (V + p.h) - p.delta * E
    return [du, dE]


def rhs_linear_const(t, y, Cc, gamma, p=P):
    u, E = y
    upos = max(u, 0.0)
    V = upos**3
    du = (p.a - (p.b + gamma * Cc + p.rho * E) * upos) / 3.0
    dE = p.s + p.p * V / (V + p.h) - p.delta * E
    return [du, dE]


def zero_event(t, y, *args):
    return y[0]
zero_event.terminal = True
zero_event.direction = -1


def cutoff_event(t, y, *args):
    return y[0] - VCELL ** (1.0/3.0)
cutoff_event.terminal = True
cutoff_event.direction = -1


def integrate_const(rhs, y0, t_end, args, event=None, max_step=0.05):
    return solve_ivp(
        lambda t, y: rhs(t, y, *args),
        (0.0, t_end), y0,
        method="DOP853", rtol=1e-10, atol=1e-12, max_step=max_step,
        events=event, dense_output=True,
    )


def gamma_matched(V0, Cc=10.0, p=P):
    return p.a * p.Emax * float(H(Cc, p)) / (Cc * V0 ** (1.0/3.0))


def matched_killing():
    Cc = 10.0
    cases = [(100.0, 110.0), (2*VCELL, 0.5)]
    summary = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (V0, t_end) in zip(axes, cases):
        gam = gamma_matched(V0, Cc)
        y0 = [V0**(1/3), 1.0]
        surf = integrate_const(rhs_surface_const, y0, t_end, (Cc, P), event=zero_event)
        lin = integrate_const(rhs_linear_const, y0, t_end, (Cc, gam, P), event=None)
        lin_cut = integrate_const(rhs_linear_const, y0, t_end, (Cc, gam, P), event=cutoff_event)
        surf_zero = float(surf.t_events[0][0]) if len(surf.t_events[0]) else np.nan
        lin_cut_t = float(lin_cut.t_events[0][0]) if len(lin_cut.t_events[0]) else np.nan
        # values used in manuscript text
        if V0 == 100.0:
            lin_eval_t = 110.0
        else:
            lin_eval_t = 0.5
        ulin = float(lin.sol(lin_eval_t)[0])
        Vlin = max(ulin,0.0)**3

        # surface theorem bound Eq. (16)
        eta = -P.a * float(Q(Cc, P))
        bound = (3/P.b) * math.log(1 + P.b * V0**(1/3) / eta)
        summary.append({
            "V0_mm3": V0,
            "gamma_L_mg_day": gam,
            "surface_zero_day": surf_zero,
            "theorem_bound_day": bound,
            "linear_cutoff_crossing_day": lin_cut_t,
            "linear_volume_at_report_time_mm3": Vlin,
            "linear_report_time_day": lin_eval_t,
        })

        # plot using dense solutions
        tt = np.linspace(0, t_end, 800)
        vs = np.maximum(surf.sol(np.minimum(tt, surf.t[-1]))[0], 0.0)**3
        if np.isfinite(surf_zero):
            vs[tt >= surf_zero] = 0.0
        vl = np.maximum(lin.sol(tt)[0], 0.0)**3
        floor=1e-10
        ax.semilogy(tt, np.maximum(vs,floor), label="Surface-scaled loss")
        ax.semilogy(tt, np.maximum(vl,floor), linestyle="--", label="Linear killing")
        ax.axhline(VCELL, linestyle=":", label="Cell cutoff")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Volume (mm³)")
        ax.set_title(f"Initial volume {V0:g} mm³")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_matched_killing.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig5_matched_killing.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary).to_csv(RESULTS / "matched_killing_summary.csv", index=False)
    return summary


def integrate_bolus(D, tau, t_end, p=P, V0=100.0, E0=1.0, max_step=0.05):
    """DOP853 integration with exact exponential PK within each interval; first bolus at t=0."""
    times=[]; us=[]; Es=[]; Cs=[]
    u0=V0**(1/3); e0=E0; Cpre=0.0; zero_t=None
    t=0.0
    while t < t_end - 1e-12 and zero_t is None:
        Cpost = Cpre + D
        next_t = min(t+tau, t_end)
        def rhs(s,y):
            u,E=y
            C=Cpost*math.exp(-p.k*(s-t))
            up=max(u,0.0); V=up**3
            return [(p.a*float(Q(C,p))-(p.b+p.rho*E)*up)/3,
                    p.s+p.p*V/(V+p.h)-p.delta*E]
        sol=solve_ivp(rhs,(t,next_t),[u0,e0],method='DOP853',rtol=1e-10,atol=1e-12,
                      max_step=max_step,events=zero_event,dense_output=True)
        sample=np.linspace(t,sol.t[-1],max(2,int(math.ceil((sol.t[-1]-t)/0.02))+1))
        yy=sol.sol(sample)
        cc=Cpost*np.exp(-p.k*(sample-t))
        if times and abs(sample[0]-times[-1])<1e-12:
            sample=sample[1:]; yy=yy[:,1:]; cc=cc[1:]
        times.extend(sample.tolist()); us.extend(yy[0].tolist()); Es.extend(yy[1].tolist()); Cs.extend(cc.tolist())
        if len(sol.t_events[0]):
            zero_t=float(sol.t_events[0][0]); break
        u0=float(sol.y[0,-1]); e0=float(sol.y[1,-1]); Cpre=Cpost*math.exp(-p.k*(next_t-t)); t=next_t
    if zero_t is not None and times[-1] < t_end:
        # post-extinction plotting continuation, immune only not needed for Fig 6
        times.append(t_end); us.append(0.0); Es.append(Es[-1]); Cs.append(Cs[-1]*math.exp(-p.k*(t_end-times[-2])))
    return np.array(times),np.array(us),np.array(Es),np.array(Cs),zero_t


def emax_comparison():
    rows=[]
    fig,ax=plt.subplots(figsize=(7,4))
    for em, ls in [(1.0,'--'),(1.5,'-')]:
        pp=replace(P,Emax=em)
        t,u,E,C,zt=integrate_bolus(10.0,3.0,730.0,pp)
        V=np.maximum(u,0)**3
        rows.append({
            "Emax":em,
            "Qbar_D10_tau3":qbar(10.0,3.0,pp),
            "zero_event_day":zt if zt is not None else np.nan,
            "V_day730_mm3":0.0 if zt is not None else float(V[-1]),
        })
        ax.semilogy(t,np.maximum(V,1e-8),linestyle=ls,label=f"Emax = {em:g}")
    ax.axhline(VCELL,linestyle=':',label='Cell cutoff')
    ax.set_xlabel('Time (days)'); ax.set_ylabel('Volume (mm³)')
    ax.legend(loc='best'); fig.tight_layout()
    fig.savefig(FIGURES/'fig6_emax.png',dpi=300,bbox_inches='tight')
    fig.savefig(FIGURES/'fig6_emax.pdf',bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RESULTS/'emax_comparison_summary.csv',index=False)
    return rows


def untreated_rhs(t,y,theta):
    a,b,rho,s,p,h,delta=theta
    u,E=y; V=max(u,0)**3
    return [(a-(b+rho*E)*u)/3, s+p*V/(V+h)-delta*E]


def solve_untreated(theta_full,t_eval):
    a,b,rho,s,p,h,delta,E0=theta_full
    sol=solve_ivp(lambda t,y: untreated_rhs(t,y,(a,b,rho,s,p,h,delta)),(0,float(max(t_eval))),[100**(1/3),E0],
                  t_eval=np.asarray(t_eval),method='DOP853',rtol=1e-12,atol=1e-14,max_step=0.05)
    return sol.y[0]**3,sol.y[1]


def symmetry_noise_demo():
    thetaA=(0.8,0.02,0.015,0.3,0.5,200.0,0.2,1.0)
    thetaB=(0.8,0.025,0.0075,7/15,1.0,200.0,0.2,4/3)
    t=np.linspace(0,60,601)
    VA,EA=solve_untreated(thetaA,t); VB,EB=solve_untreated(thetaB,t)
    maxdiff=float(np.max(np.abs(VA-VB)))

    obs_t=np.arange(0,61,5,dtype=float)
    Vobs_true,Eobs_true=solve_untreated(thetaA,obs_t)
    rng=np.random.default_rng(20261015)
    sigma=0.05
    logVobs=np.log(Vobs_true)+rng.normal(0,sigma,len(obs_t))
    logEobs=np.log(Eobs_true)+rng.normal(0,sigma,len(obs_t))
    Vobs=np.exp(logVobs); Eobs=np.exp(logEobs)
    pd.DataFrame({"day":obs_t,"V_true":Vobs_true,"E_true":Eobs_true,"V_observed":Vobs,"E_observed":Eobs}).to_csv(RESULTS/'symmetry_observations.csv',index=False)

    lambdas=np.linspace(0.5,2.0,301)
    wrss_v=[]; wrss_joint=[]
    # Pure scaling family only: b fixed, rho->rho/lambda, s,p,E0 -> lambda*(s,p,E0)
    a,b,rho,s,p,h,delta,E0=thetaA
    # Along the pure scaling symmetry, V(t) is invariant analytically and E(t) scales by lambda.
    # Reusing the baseline predictions avoids unnecessary repeated integrations and makes the
    # flat volume-only likelihood slice explicit.
    for lam in lambdas:
        Vpred=Vobs_true
        Epred=lam*Eobs_true
        rv=((np.log(Vobs)-np.log(Vpred))/sigma)**2
        re=((np.log(Eobs)-np.log(Epred))/sigma)**2
        wrss_v.append(float(rv.sum()))
        wrss_joint.append(float(rv.sum()+re.sum()))
    wrss_v=np.array(wrss_v); wrss_joint=np.array(wrss_joint)
    pd.DataFrame({"lambda":lambdas,"wrss_V_only":wrss_v,"wrss_V_and_E":wrss_joint,
                  "delta_wrss_V_only":wrss_v-wrss_v.min(),"delta_wrss_V_and_E":wrss_joint-wrss_joint.min()}).to_csv(RESULTS/'symmetry_lambda_scan.csv',index=False)
    lam_min=float(lambdas[np.argmin(wrss_joint)])

    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    axes[0].plot(t,VA,label='Set A'); axes[0].plot(t,VB,linestyle='--',label='Set B')
    axes[0].set_xlabel('Time (days)'); axes[0].set_ylabel('Volume (mm³)'); axes[0].legend(fontsize=8)
    axes[1].plot(t,EA,label='Set A'); axes[1].plot(t,EB,linestyle='--',label='Set B')
    axes[1].set_xlabel('Time (days)'); axes[1].set_ylabel('Effectors (10⁶ cells)')
    axes[2].plot(lambdas,wrss_v-wrss_v.min(),label='V only')
    axes[2].plot(lambdas,wrss_joint-wrss_joint.min(),label='V and E')
    axes[2].set_xlabel('Immune scale λ'); axes[2].set_ylabel('Δ weighted residual sum'); axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES/'fig7_symmetry.png',dpi=300,bbox_inches='tight'); fig.savefig(FIGURES/'fig7_symmetry.pdf',bbox_inches='tight'); plt.close(fig)

    return {"max_abs_volume_difference_mm3":maxdiff,"joint_slice_lambda_min":lam_min,"n_observations":len(obs_t)}


def boundary_checks():
    pts=[(26,5.5),(28,5.625)]
    rows=[]
    cc=ccrit(P)
    for D,tau in pts:
        thresh=cc*(1-math.exp(-P.k*tau))
        rows.append({"D_mg_L":D,"tau_day":tau,"Qbar":qbar(D,tau,P),"necessary_D_threshold_mg_L":thresh,"above_necessary_threshold":D>thresh})
    pd.DataFrame(rows).to_csv(RESULTS/'near_boundary_checks.csv',index=False)
    return rows


def write_summary(matched,emax,sym,boundary):
    with open(RESULTS/'new_revision_numerical_checks.txt','w',encoding='utf-8') as f:
        f.write('New revision numerical checks\n=============================\n\n')
        f.write('Matched-killing comparison\n')
        for r in matched: f.write(str(r)+'\n')
        f.write('\nEmax comparison\n')
        for r in emax: f.write(str(r)+'\n')
        f.write('\nSymmetry/noise demonstration\n'+str(sym)+'\n')
        f.write('\nNear-boundary periodic averages\n')
        for r in boundary: f.write(str(r)+'\n')

if __name__=='__main__':
    matched=matched_killing()
    emax=emax_comparison()
    sym=symmetry_noise_demo()
    boundary=boundary_checks()
    write_summary(matched,emax,sym,boundary)
    print('Wrote results to', RESULTS)
    print('Wrote figures to', FIGURES)
