#!/usr/bin/env python3
"""Ge76/Se76 PES figures, adapted from the supplied Plot_v1.py.

Run after run_ge76_se76_pes.py, optionally with --method IMSRG3f2 or --method IMSRG2.
The default plots both methods using a common color scale. Failed points and all
triangles touching a failed point are masked; no extrapolation fills missing data.
Outputs individual and combined PNG/PDF figures with a common relative-energy scale.
"""
from pathlib import Path
import argparse,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'Output/ge76_se76_pes/outputs'


def load(root,nucleus):
    data=pd.read_csv(root/nucleus/'surface.csv').sort_values(['beta','gamma_deg']).reset_index(drop=True)
    required={'beta','gamma_deg','energy_MeV','converged','accepted_energy_MeV','stability_checked'}
    if not required.issubset(data): raise ValueError(f'{nucleus}: missing columns {required-set(data)}')
    for col in ('converged','stability_checked'):
        data[col]=data[col].astype(str).str.lower().eq('true')
    data['accepted']=data.converged & data.stability_checked & np.isfinite(data.accepted_energy_MeV)
    if data.accepted.sum()<3: raise ValueError('too few accepted points for a surface')
    minimum=data.loc[data.accepted,'accepted_energy_MeV'].min()
    data['relative_MeV']=data.accepted_energy_MeV-minimum
    return data,minimum


def xy(beta,gamma):
    theta=np.deg2rad(gamma)
    return beta*np.cos(theta),beta*np.sin(theta)


def sector_grid(ax,r):
    theta=np.linspace(0,np.pi/3,400)
    for b in np.arange(.02,r+.001,.02):
        ax.plot(b*np.cos(theta),b*np.sin(theta),color='black',lw=.5,alpha=.16,zorder=2)
        if round(b*100)%4==0:
            ax.text(b,-r*.048,f'{b:.2f}',ha='center',va='top',fontsize=10)
    for g in range(0,61,10):
        angle=np.deg2rad(g)
        ax.plot([0,r*np.cos(angle)],[0,r*np.sin(angle)],color='black',lw=.5,alpha=.18,zorder=2)
        ax.text(r*1.063*np.cos(angle),r*1.063*np.sin(angle),rf'${g}^\circ$',fontsize=10,ha='center',va='center')
    ax.text(r*.49,-r*.155,r'quadrupole deformation $\beta$',fontsize=11,ha='center',va='top')
    ax.text(r*1.05,r*.64,r'$\gamma$',fontsize=12,ha='center')


def panel(ax,nucleus,data,minimum,levels,norm,method,emax):
    x,y=xy(data.beta.to_numpy(),data.gamma_deg.to_numpy())
    triangulation=mtri.Triangulation(x,y)
    valid=data.accepted.to_numpy()
    triangulation.set_mask(np.any(~valid[triangulation.triangles],axis=1))
    values=np.where(valid,data.relative_MeV.to_numpy(),0.)
    filled=ax.tricontourf(triangulation,values,levels=levels,cmap='Spectral_r',norm=norm,zorder=1,extend='max')
    filled.set_edgecolor('face')  # prevent white hairlines between PDF contour bands
    contour_levels=[v for v in (.5,1,2,5,10,20,30) if v<max(values)]
    if contour_levels:
        lines=ax.tricontour(triangulation,values,levels=contour_levels,colors='black',linewidths=.55,alpha=.6,zorder=3)
        ax.clabel(lines,fmt='%g',fontsize=7,inline=True)
    ax.scatter(x[valid],y[valid],c=values[valid],cmap='Spectral_r',norm=norm,s=10,edgecolors='black',linewidths=.25,zorder=4)
    if (~valid).any():
        ax.scatter(x[~valid],y[~valid],marker='x',s=35,color='black',lw=.8,zorder=5)
    lowest=data.loc[data.accepted_energy_MeV.where(data.accepted).idxmin()]
    mx,my=xy(lowest.beta,lowest.gamma_deg)
    ax.scatter([mx],[my],marker='*',s=190,color='white',edgecolors='black',linewidths=1.1,zorder=7)
    ax.annotate(rf"$\beta={lowest.beta:.2f},\ \gamma={lowest.gamma_deg:.0f}^\circ$"+'\n'+
                rf"$E_{{\min}}={minimum:.3f}$ MeV",xy=(mx,my),xytext=(.035,.82),textcoords='axes fraction',
                fontsize=9,ha='left',va='top',bbox=dict(boxstyle='round,pad=.4',facecolor='white',edgecolor='.65',alpha=.96),
                arrowprops=dict(arrowstyle='->',color='.2',lw=.8),zorder=8)
    radius=.16
    sector_grid(ax,radius)
    element=nucleus[:2]
    ax.set_title(rf'$^{{76}}\mathrm{{{element}}}$ Hartree–Fock PES',fontsize=16,pad=57)
    ax.text(.5,1.085,f'{method} · 1.8/2.0 (EM)',transform=ax.transAxes,ha='center',fontsize=12)
    ax.text(.5,1.025,rf'$e_{{\max}}={emax}$ · jj44 · $\hbar\omega=12$ MeV',
            transform=ax.transAxes,ha='center',fontsize=10,color='.3')
    ax.text(.48,-.01,f'{valid.sum()}/{len(valid)} points accepted · forward/reverse minimum',
            transform=ax.transAxes,ha='center',va='top',fontsize=8,color='.3')
    handles=[Line2D([],[],marker='o',ls='',ms=4,mfc='.8',mec='black',mew=.5,label='Converged point'),
             Line2D([],[],marker='*',ls='',ms=10,mfc='white',mec='black',label='Grid minimum')]
    if (~valid).any(): handles.append(Line2D([],[],marker='x',ls='',color='black',label='Not accepted'))
    ax.legend(handles=handles,loc='upper right',fontsize=8,framealpha=.95)
    ax.set_aspect('equal');ax.set_xlim(-.012,.183);ax.set_ylim(-.035,.16);ax.axis('off')
    return filled,dict(nucleus=nucleus,method=method,emax=emax,interaction_display_name='1.8/2.0 (EM)',
                       minimum_MeV=minimum,beta=float(lowest.beta),gamma_deg=float(lowest.gamma_deg),
                       accepted=int(valid.sum()),failed=int((~valid).sum()))


def plot_method(root,method,datasets,levels,norm):
    filename='interaction.snt' if method=='IMSRG3f2' else 'IMSRG2_jj44_Ge76_e12_hw12_E328.snt'
    header=(ROOT/'input'/filename).read_text()
    emax=int(re.search(r'e1max:\s*(\d+)',header).group(1))
    summaries=[]
    for nucleus,(data,minimum) in datasets.items():
        fig,ax=plt.subplots(figsize=(7.7,7.3),layout='constrained')
        filled,summary=panel(ax,nucleus,data,minimum,levels,norm,method,emax)
        cb=fig.colorbar(filled,ax=ax,pad=.035,shrink=.8)
        cb.set_label(r'$E_{\rm HF}-E_{\min}$ [MeV]',fontsize=11)
        fig.savefig(root/f'{nucleus}_PES.png',dpi=240,bbox_inches='tight')
        fig.savefig(root/f'{nucleus}_PES.pdf',bbox_inches='tight')
        plt.close(fig);summaries.append(summary)
    fig,axes=plt.subplots(1,2,figsize=(13.8,7.05),layout='constrained')
    for ax,(nucleus,(data,minimum)) in zip(axes,datasets.items()):
        filled,_=panel(ax,nucleus,data,minimum,levels,norm,method,emax)
    cb=fig.colorbar(filled,ax=axes,pad=.025,shrink=.78)
    cb.set_label(r'$E_{\rm HF}-E_{\min}$ [MeV]',fontsize=11)
    fig.suptitle('Same Ge76-derived interaction for both nuclei',fontsize=10,color='.35',y=1.035)
    fig.savefig(root/'Ge76_Se76_PES.png',dpi=220,bbox_inches='tight')
    fig.savefig(root/'Ge76_Se76_PES.pdf',bbox_inches='tight')
    plt.close(fig)
    (root/'plot_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['all','IMSRG3f2','IMSRG2'],default='all')
    parser.add_argument('--output-dir',type=Path,default=ROOT,help='directory containing input/ and per-method surfaces')
    args=parser.parse_args()
    ROOT=args.output_dir.expanduser().resolve()
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.unicode_minus':True})
    methods=['IMSRG3f2','IMSRG2'] if args.method=='all' else [args.method]
    roots={method:ROOT/('IMSRG2' if method=='IMSRG2' else '.') for method in methods}
    datasets={method:{n:load(roots[method],n) for n in ('Ge76','Se76')} for method in methods}
    upper=max(5.,5*np.ceil(max(d.relative_MeV.max() for ds in datasets.values() for d,e in ds.values())/5))
    levels=np.linspace(0,upper,41);norm=BoundaryNorm(levels,ncolors=256,clip=True)
    for method in methods: plot_method(roots[method],method,datasets[method],levels,norm)


if __name__=='__main__': main()
