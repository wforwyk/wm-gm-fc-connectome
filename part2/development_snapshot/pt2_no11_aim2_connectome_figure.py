#!/usr/bin/env python3
# =============================================================================
# pt2_no11_aim2_connectome_figure.py
# PURPOSE:
#   Single integrated Aim 2 reporting script.  Produces a publication-scale
#   figure combining: (A) overall primary/matched interaction, (B) tract-by-
#   phenotype heatmap, and (C) circular tract connectogram of WM-connected
#   AAL endpoints.  It replaces the separate No.11/No.12 visualization scripts.
# INPUT: No.5 primary per-subject FC CSVs; No.9 matched FC/balance CSVs;
#   AAL region category CSV for node names.
# OUTPUT: aim2_integrated_figure.png; aim2_tract_effects.csv; short report.
# USER SETTINGS: edit only the block below.
# =============================================================================
import os, glob
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# =============================================================================
# USER SETTINGS
# =============================================================================
PRIMARY_INPUT_DIR='/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no5_endpoint_fc'
MATCHED_INPUT_DIR='/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no9_matched_endpoint_fc'
TEMPLATE_DIR='/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/template/AAL3'
OUTPUT_DIR='/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no11_aim2_connectome_figure'
STRICT_MATCHED_ONLY=True; MIN_MATCHED_VOXELS=10; MAX_ABSOLUTE_SMD=.10
# =============================================================================

COMM={'cc.genu','cc.rostrum','cc.splenium','cc.bodyc','cc.bodyp','cc.bodypf','cc.bodypm','cc.bodyt','lh.fx','rh.fx','mcp','acomm'}
PROJ={'lh.atr','rh.atr','lh.ar','rh.ar','lh.or','rh.or'}
def system(t): return 'Commissural' if t in COMM else ('Projection' if t in PROJ else 'Association')
COL={'Commissural':'#bd5b55','Projection':'#3f83b7','Association':'#3d9970'}
def files(d,p):
    x=glob.glob(os.path.join(d,p))
    if not x: raise RuntimeError(f'No {p} files in {d}')
    return pd.concat([pd.read_csv(f) for f in x],ignore_index=True)
def calc(d,ep,ctl,eligible=None):
    if eligible is not None:d=d.merge(eligible,on=['subject','tract_name','endpoint'],how='inner')
    u=d.dropna(subset=['mean_fc_z']).groupby(['subject','tract_name','endpoint','source_roi','target_roi','voxel_class','target_type'],as_index=False).mean_fc_z.mean()
    # ``endpoint`` is both a tract-end index column and a voxel-class value;
    # rename the pivoted voxel-class columns before reset_index to avoid a
    # pandas duplicate-column collision.
    w=u.pivot_table(index=['subject','tract_name','endpoint','source_roi','target_roi','target_type'],columns='voxel_class',values='mean_fc_z').dropna()
    w=w.rename(columns={ep:'endpoint_fc',ctl:'control_fc'}).reset_index();w['effect']=w['endpoint_fc']-w['control_fc']
    # tract effects: adjacent ROIs averaged before endpoint/subject averaging.
    q=w.groupby(['subject','tract_name','endpoint','target_type'],as_index=False).effect.mean().pivot_table(index=['subject','tract_name','endpoint'],columns='target_type',values='effect').dropna().reset_index();q['interaction']=q.wm_connected_distant-q.physically_adjacent
    tract=q.groupby('tract_name',as_index=False).agg(adjacent=('physically_adjacent','mean'),distant=('wm_connected_distant','mean'),interaction=('interaction','mean'),n_subjects=('subject','nunique'))
    # modal source-target endpoint pair per tract supports a readable circular connectogram.
    edge=w[w.target_type=='wm_connected_distant'].groupby(['tract_name','source_roi','target_roi'],as_index=False).agg(effect=('effect','mean'),n=('subject','nunique')).sort_values('n',ascending=False).drop_duplicates('tract_name')
    return tract,edge
def curve(ax,p1,p2,col,lw):
    mid=(p1+p2)/2*.18; path=Path([p1,mid,p2],[Path.MOVETO,Path.CURVE3,Path.CURVE3]);ax.add_patch(PathPatch(path,fill=False,edgecolor=col,lw=lw,alpha=.7))
def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    primary_raw=files(PRIMARY_INPUT_DIR,'*_aim2_endpoint_fc.csv');primary,edges=calc(primary_raw,'endpoint','non_endpoint_control')
    matched_raw=files(MATCHED_INPUT_DIR,'*_aim2_matched_endpoint_fc.csv'); eligible=None
    if STRICT_MATCHED_ONLY:
        b=files(MATCHED_INPUT_DIR,'*_aim2_matching_balance.csv');good=b[(b.n_matched>=MIN_MATCHED_VOXELS)&(b.smd_wm_distance.abs()<=MAX_ABSOLUTE_SMD)&(b.smd_gm_probability.abs()<=MAX_ABSOLUTE_SMD)];eligible=good[['subject','tract','endpoint']].rename(columns={'tract':'tract_name'})
    matched,_=calc(matched_raw,'endpoint','matched_non_endpoint',eligible)
    both=pd.concat([primary.assign(analysis='Primary'),matched.assign(analysis='Matched')]);both['system']=both.tract_name.map(system);both.to_csv(os.path.join(OUTPUT_DIR,'aim2_tract_effects.csv'),index=False)
    order=[]
    for s in ['Commissural','Projection','Association']:order+=sorted(both[both.system==s].tract_name.unique())
    fig=plt.figure(figsize=(22,15));gs=fig.add_gridspec(2,2,width_ratios=[1.25,1],height_ratios=[.52,1],hspace=.28,wspace=.26)
    # A overall interaction comparison, from tract-resolved estimates.
    ax=fig.add_subplot(gs[0,0]); summary=both.groupby('analysis').interaction.agg(['mean','sem']).reindex(['Primary','Matched'])
    for i,(name,r) in enumerate(summary.iterrows()):ax.errorbar(r['mean'],i,xerr=1.96*r['sem'],fmt='o',color='#1d587c' if name=='Primary' else '#c76a45',capsize=4,ms=9);ax.text(r['mean'],i-.18,f"{r['mean']:.4f}",ha='center',fontsize=10)
    ax.axvline(0,color='.4',ls='--');ax.set(yticks=[0,1],yticklabels=['Primary','Matched sensitivity'],xlabel='Tract-mean distant − adjacent interaction (Fisher Z)',title='A. Overall endpoint preference');ax.invert_yaxis()
    # B tract phenotype heatmap.
    ax=fig.add_subplot(gs[1,0]);cols=['adjacent','distant','interaction'];mat=[]
    for t in order:
        row=[]
        for an in ['Primary','Matched']:
            x=both[(both.analysis==an)&(both.tract_name==t)].set_index('tract_name');row+=x[cols].iloc[0].tolist() if len(x) else [np.nan]*3
        mat.append(row)
    vmax=max(.001,np.nanmax(np.abs(mat)));im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax));ax.set_yticks(range(len(order)));ax.set_yticklabels(order,fontsize=8);ax.set_xticks(range(6));ax.set_xticklabels(['Adj.','Distant','Interaction','Adj.','Distant','Interaction']);ax.set_title('B. Tract-resolved endpoint FC phenotype')
    ax.text(1,.99,'Primary',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontweight='bold');ax.text(4,.99,'Strict matched',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontweight='bold')
    for i,t in enumerate(order):ax.get_yticklabels()[i].set_color(COL[system(t)])
    fig.colorbar(im,ax=ax,shrink=.6,label='Endpoint − control FC (Fisher Z)')
    # C circular connectogram, uses remote WM-connected edges only.
    ax=fig.add_subplot(gs[:,1]);nodes=sorted(set(edges.source_roi).union(edges.target_roi));theta=np.linspace(np.pi/2,np.pi/2+2*np.pi,len(nodes),endpoint=False);pos={n:np.array([np.cos(a),np.sin(a)]) for n,a in zip(nodes,theta)};names=dict(zip(pd.read_csv(os.path.join(TEMPLATE_DIR,'aal3_region_category.csv')).roi_id,pd.read_csv(os.path.join(TEMPLATE_DIR,'aal3_region_category.csv')).roi_name))
    scale=max(.0001,edges.effect.abs().max())
    for r in edges.itertuples(index=False):curve(ax,pos[r.source_roi],pos[r.target_roi],COL[system(r.tract_name)],.6+4*abs(r.effect)/scale)
    for n in nodes:
        ax.scatter(*pos[n],s=18,color='#222',zorder=4);p=pos[n]*1.12;ax.text(*p,names.get(n,f'ROI {n}'),fontsize=5.5,ha='center',va='center')
    ax.set(xlim=(-1.35,1.35),ylim=(-1.35,1.35),title='C. WM-connected AAL endpoint connectogram');ax.axis('off')
    for s,c in COL.items():ax.plot([],[],color=c,lw=3,label=s)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.03),ncol=3,frameon=False,title='TRACULA pathway class')
    fig.suptitle('Aim 2: endpoint functional-connectivity phenotype across the TRACULA connectome',fontsize=18,y=.98);fig.savefig(os.path.join(OUTPUT_DIR,'aim2_integrated_figure.png'),dpi=350,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
