import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh


# ============================================================
# CONFIG
# ============================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
SUMMARY_URL = f"{API_BASE_URL}/api/v1/audit/summary"
RECENT_URL = f"{API_BASE_URL}/api/v1/audit/recent"
HEALTH_URL = f"{API_BASE_URL}/health"
MODEL_URL = f"{API_BASE_URL}/model-info"

st.set_page_config(
    page_title="AI-CyberShield | Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st_autorefresh(interval=5000, key="cybershield_live")


# ============================================================
# CYBER UI
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #040812;
        --panel: rgba(8,18,34,.84);
        --panel2: rgba(13,28,49,.76);
        --line: rgba(87,186,255,.16);
        --text: #eef8ff;
        --muted: #8ba6bd;
        --cyan: #22d3ee;
        --blue: #60a5fa;
        --violet: #a78bfa;
        --green: #34d399;
        --amber: #fbbf24;
        --orange: #fb923c;
        --red: #fb7185;
    }

    .stApp {
        background:
          radial-gradient(circle at 50% -10%, rgba(34,211,238,.13), transparent 30%),
          radial-gradient(circle at 100% 18%, rgba(96,165,250,.10), transparent 24%),
          radial-gradient(circle at 0% 80%, rgba(167,139,250,.09), transparent 24%),
          linear-gradient(180deg, #030711 0%, #050a14 55%, #030711 100%);
    }

    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1650px; padding-top: 1rem; padding-bottom: 2.5rem; }

    .topbar {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border: 1px solid var(--line);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(15,31,52,.94), rgba(3,10,21,.94));
        box-shadow: 0 18px 60px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
        overflow: hidden;
    }

    .topbar:after {
        content: "";
        position: absolute;
        left: -15%;
        right: -15%;
        bottom: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
        opacity: .65;
        animation: sweep 5s linear infinite;
    }

    .brand { display: flex; align-items: center; gap: 13px; }
    .logo {
        width: 52px; height: 52px; border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        font-size: 29px;
        background: linear-gradient(145deg, rgba(34,211,238,.22), rgba(167,139,250,.14));
        border: 1px solid rgba(120,225,255,.27);
        box-shadow: 0 0 36px rgba(34,211,238,.14), inset 0 0 24px rgba(255,255,255,.05);
    }

    .brand-title { font-size: 31px; font-weight: 900; letter-spacing: -.7px; line-height: 1; }
    .brand-subtitle {
        margin-top: 6px; color: var(--muted); font-size: 10px;
        text-transform: uppercase; letter-spacing: 1.7px; font-weight: 800;
    }

    .badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
    .badge {
        padding: 7px 11px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,.08);
        background: rgba(255,255,255,.035);
        color: #d8eafd; font-size: 10px; font-weight: 800; letter-spacing: .55px;
    }

    .hero-grid { display: grid; grid-template-columns: 1.16fr .84fr; gap: 16px; margin-bottom: 16px; }
    .hero, .signals, .panel, .threat-space, .metric-card {
        border: 1px solid var(--line);
        background: linear-gradient(145deg, rgba(13,28,49,.86), rgba(4,11,22,.90));
        box-shadow: 0 22px 65px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.035);
    }

    .hero {
        min-height: 430px; border-radius: 28px; position: relative;
        display: flex; align-items: center; justify-content: center; overflow: hidden;
    }

    .hero:before {
        content: ""; position: absolute; inset: -35%;
        background: conic-gradient(from 0deg, transparent, rgba(34,211,238,.09), transparent 18%, transparent 58%, rgba(167,139,250,.09), transparent);
        animation: spin 18s linear infinite;
    }

    .hero:after {
        content: ""; position: absolute; inset: 9%; border: 1px dashed rgba(91,185,255,.10);
        border-radius: 50%; animation: spinReverse 25s linear infinite;
    }

    .orbital-core { width: 350px; height: 350px; border-radius: 50%; position: relative; z-index: 2; display:flex; align-items:center; justify-content:center; }
    .orbital-core:before, .orbital-core:after { content:""; position:absolute; border-radius:50%; border:1px solid rgba(34,211,238,.18); box-shadow:0 0 45px rgba(34,211,238,.07); }
    .orbital-core:before { inset:4%; animation:pulse 3.5s ease-in-out infinite; }
    .orbital-core:after { inset:15%; border-style:dashed; border-color:rgba(167,139,250,.20); animation:spinReverse 10s linear infinite; }

    .dot { position:absolute; width:10px; height:10px; border-radius:50%; background:var(--cyan); box-shadow:0 0 19px var(--cyan); }
    .dot-a { top:22px; left:50%; animation:orbitA 7s linear infinite; }
    .dot-b { right:45px; top:45%; background:var(--violet); box-shadow:0 0 19px var(--violet); animation:orbitB 9s linear infinite; }

    .shield {
        width: 174px; height: 196px; position:relative; z-index:5;
        clip-path: polygon(50% 0%, 88% 14%, 100% 48%, 82% 79%, 50% 100%, 18% 79%, 0% 48%, 12% 14%);
        display:flex; align-items:center; justify-content:center;
        background: linear-gradient(145deg, rgba(212,251,255,.24), rgba(34,211,238,.17) 30%, rgba(37,99,235,.26) 57%, rgba(167,139,250,.26));
        border:1px solid rgba(173,239,255,.48);
        box-shadow: 0 0 28px rgba(34,211,238,.23), 0 0 90px rgba(96,165,250,.14), inset 0 0 34px rgba(255,255,255,.09);
        transform: perspective(850px) rotateX(8deg) rotateY(-8deg);
        animation: floatShield 4s ease-in-out infinite;
        backdrop-filter: blur(10px);
    }

    .shield:before { content:""; position:absolute; inset:13px; clip-path:inherit; border:1px solid rgba(255,255,255,.14); }
    .shield:after { content:""; position:absolute; left:16%; right:16%; top:34%; height:2px; background:linear-gradient(90deg, transparent, rgba(255,255,255,.86), transparent); box-shadow:0 0 18px rgba(255,255,255,.55); animation:scan 2.4s ease-in-out infinite; }
    .shield-icon { font-size:72px; filter:drop-shadow(0 0 19px rgba(255,255,255,.25)); }
    .hero-caption { position:absolute; bottom:24px; left:0; right:0; text-align:center; z-index:6; }
    .kicker { color:var(--muted); font-size:10px; font-weight:850; text-transform:uppercase; letter-spacing:2px; }
    .hero-state { margin-top:3px; font-size:25px; font-weight:900; letter-spacing:1px; }
    .hero-risk { color:#b7ccdf; font-size:12px; margin-top:2px; }

    .signals { min-height:430px; border-radius:28px; padding:20px; }
    .section-title { color:#a9c0d6; font-size:11px; font-weight:900; letter-spacing:1.45px; text-transform:uppercase; margin-bottom:16px; }
    .signal { margin-bottom:17px; }
    .signal-head { display:flex; justify-content:space-between; margin-bottom:7px; font-size:12px; }
    .signal-label { color:#9eb6cd; font-weight:700; }
    .signal-number { font-weight:900; }
    .signal-bar { height:8px; border-radius:99px; overflow:hidden; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.07); }
    .signal-fill { height:100%; border-radius:99px; background:linear-gradient(90deg, var(--cyan), var(--blue), var(--violet)); box-shadow:0 0 17px rgba(34,211,238,.22); }
    .live-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:20px; }
    .live-cell { padding:13px; border-radius:15px; background:rgba(255,255,255,.024); border:1px solid rgba(255,255,255,.065); }
    .live-label { color:#718ba2; font-size:9px; text-transform:uppercase; letter-spacing:1px; font-weight:800; }
    .live-value { margin-top:4px; font-size:16px; font-weight:900; line-height:1.15; }

    .threat-space { border-radius:22px; padding:10px 10px 0; margin-bottom:16px; }
    .metric-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:11px; margin-bottom:16px; }
    .metric-card { border-radius:19px; padding:16px; }
    .metric-label { color:#7f98af; font-size:10px; font-weight:850; text-transform:uppercase; letter-spacing:1px; }
    .metric-value { font-size:29px; font-weight:900; margin-top:4px; }
    .metric-note { color:#61798f; font-size:10px; margin-top:3px; }

    .status-rail { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:16px; padding:13px 16px; border-radius:18px; border:1px solid rgba(34,211,238,.13); background:linear-gradient(90deg, rgba(34,211,238,.06), rgba(167,139,250,.045), rgba(255,255,255,.02)); }
    .status-dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:8px; box-shadow:0 0 17px currentColor; animation:blink 1.5s ease-in-out infinite; }
    .status-text { font-size:12px; font-weight:850; letter-spacing:.9px; text-transform:uppercase; }
    .status-meta { color:#7f96ad; font-size:10px; }

    .panel { border-radius:20px; padding:14px; margin-bottom:16px; }
    .alert-card { padding:15px 17px; margin-bottom:10px; border-radius:16px; border:1px solid rgba(255,255,255,.075); background:linear-gradient(135deg, rgba(255,255,255,.025), rgba(255,255,255,.012)); }
    .alert-top { display:flex; justify-content:space-between; gap:10px; color:#95aec4; font-size:10px; }
    .alert-title { color:#edf7ff; font-size:14px; font-weight:850; margin-top:5px; }
    .alert-metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:9px; }
    .alert-metric { padding:9px; border-radius:12px; background:rgba(255,255,255,.024); }
    .alert-metric span { display:block; color:#6d859c; font-size:9px; text-transform:uppercase; letter-spacing:.8px; }
    .alert-metric b { display:block; margin-top:3px; font-size:13px; }

    .scenario-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
    .scenario { min-height:108px; padding:12px; border-radius:16px; border:1px solid rgba(255,255,255,.06); background:rgba(255,255,255,.022); }
    .scenario-name { color:#c6d8e8; font-size:11px; font-weight:850; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .scenario-state { margin-top:9px; font-size:13px; font-weight:900; }
    .scenario-meta { color:#7a92a8; font-size:10px; margin-top:5px; line-height:1.35; }

    .topology { display:flex; align-items:center; justify-content:space-between; gap:8px; overflow-x:auto; padding:17px; border-radius:18px; border:1px solid var(--line); background:linear-gradient(90deg, rgba(14,27,46,.86), rgba(5,11,22,.90)); }
    .node { min-width:122px; padding:12px 9px; border-radius:14px; text-align:center; border:1px solid rgba(255,255,255,.065); background:rgba(255,255,255,.022); }
    .node-icon { font-size:23px; }
    .node-name { color:#a7bbce; font-size:9px; text-transform:uppercase; letter-spacing:.75px; font-weight:850; margin-top:4px; }
    .arrow { color:#527593; font-size:20px; flex:0 0 auto; }

    .footer { text-align:center; color:#587089; font-size:10px; padding:18px 0 5px; letter-spacing:.45px; }

    @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
    @keyframes spinReverse { from{transform:rotate(360deg)} to{transform:rotate(0deg)} }
    @keyframes pulse { 0%,100%{transform:scale(.96);opacity:.60} 50%{transform:scale(1.04);opacity:1} }
    @keyframes blink { 0%,100%{opacity:.45} 50%{opacity:1} }
    @keyframes floatShield { 0%,100%{transform:perspective(850px) rotateX(8deg) rotateY(-8deg) translateY(0)} 50%{transform:perspective(850px) rotateX(3deg) rotateY(6deg) translateY(-10px)} }
    @keyframes scan { 0%,100%{top:30%;opacity:.1} 50%{top:64%;opacity:1} }
    @keyframes orbitA { from{transform:rotate(0deg) translateX(145px) rotate(0deg)} to{transform:rotate(360deg) translateX(145px) rotate(-360deg)} }
    @keyframes orbitB { from{transform:rotate(360deg) translateX(130px) rotate(-360deg)} to{transform:rotate(0deg) translateX(130px) rotate(0deg)} }
    @keyframes sweep { from{transform:translateX(-20%)} to{transform:translateX(20%)} }

    @media(max-width:1100px){
        .hero-grid{grid-template-columns:1fr}
        .metric-grid{grid-template-columns:repeat(2,1fr)}
        .topbar{flex-direction:column;align-items:flex-start}
        .badges{justify-content:flex-start}
        .scenario-grid{grid-template-columns:repeat(2,1fr)}
    }
    @media(max-width:650px){
        .metric-grid{grid-template-columns:1fr}
        .scenario-grid{grid-template-columns:1fr}
        .live-grid{grid-template-columns:1fr}
        .brand-title{font-size:24px}
        .orbital-core{width:290px;height:290px}
        .hero,.signals{min-height:380px}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&display=swap');
    @property --n { syntax:'<integer>'; initial-value:0; inherits:false; }
    html, body, .stApp, [data-testid="stMarkdownContainer"] { font-family:'Rajdhani',sans-serif; }
    .brand-title,.hero-state,.metric-value,.section-title,.status-text,.live-value { font-family:'Orbitron',sans-serif; }
    .brand-title { font-size:26px; letter-spacing:1px; background:linear-gradient(90deg,#eef8ff,#22d3ee,#a78bfa); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .metric-value { font-size:25px; }
    .section-title { font-size:12px; margin-top:6px; }
    .block-container { position:relative; z-index:1; }
    .stApp:before { content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
        background-image:linear-gradient(rgba(34,211,238,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(34,211,238,.055) 1px,transparent 1px);
        background-size:56px 56px; -webkit-mask-image:radial-gradient(ellipse at 50% 0,#000,transparent 78%); mask-image:radial-gradient(ellipse at 50% 0,#000,transparent 78%);
        animation:gridmove 9s linear infinite; }
    .stApp:after { content:""; position:fixed; left:0; right:0; top:-160px; height:160px; pointer-events:none; z-index:0;
        background:linear-gradient(transparent,rgba(34,211,238,.07),transparent); animation:beam 8s linear infinite; }
    .metric-card { position:relative; overflow:hidden; transition:transform .35s, box-shadow .35s; }
    .metric-card:before { content:""; position:absolute; top:0; left:-60%; width:60%; height:2px; background:linear-gradient(90deg,transparent,#22d3ee,transparent); animation:edge 3.4s linear infinite; }
    .node { transition:transform .35s, box-shadow .35s; box-shadow:0 0 22px rgba(34,211,238,.06); }
    .metric-card:hover,.node:hover { transform:perspective(700px) rotateX(7deg) translateY(-5px); box-shadow:0 14px 40px rgba(34,211,238,.22); }
    .count { counter-reset:n var(--n); animation:countup 1.6s ease-out; }
    .count:after { content:counter(n); }
    .ticker { overflow:hidden; white-space:nowrap; margin-bottom:16px; padding:9px 0; border:1px solid var(--line); border-radius:14px; background:rgba(4,12,24,.72); color:#9fc3dd; font-size:13px; letter-spacing:.6px; }
    .ticker > div { display:inline-block; padding-left:100%; animation:marq 45s linear infinite; }
    .arrow { position:relative; flex:1 1 34px; min-width:34px; height:2px; font-size:0; background:linear-gradient(90deg,transparent,rgba(34,211,238,.45),transparent); }
    .arrow:after { content:""; position:absolute; top:-3px; left:0; width:8px; height:8px; border-radius:50%; background:#22d3ee; box-shadow:0 0 12px #22d3ee; animation:packet 1.8s linear infinite; }
    [data-testid="stMetric"] { padding:14px 16px; border-radius:16px; border:1px solid var(--line); background:linear-gradient(145deg,rgba(13,28,49,.86),rgba(4,11,22,.9)); }
    iframe { border:0; }
    @keyframes gridmove { to { background-position:0 56px, 56px 0; } }
    @keyframes beam { to { transform:translateY(130vh); } }
    @keyframes edge { to { left:120%; } }
    @keyframes countup { from { --n:0; } }
    @keyframes marq { to { transform:translateX(-100%); } }
    @keyframes packet { from { left:0; } to { left:calc(100% - 8px); } }
    @media (prefers-reduced-motion:reduce) { * { animation:none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

HERO_TPL = r'''<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@900&family=Rajdhani:wght@600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
html,body{margin:0;background:transparent;overflow:hidden;font-family:Rajdhani,sans-serif}
#w{position:relative;height:430px;border-radius:28px;overflow:hidden;border:1px solid rgba(87,186,255,.16);
background:radial-gradient(circle at 50% 42%,__COLOR__26,transparent 55%),linear-gradient(145deg,rgba(13,28,49,.92),rgba(4,11,22,.95))}
canvas{display:block}
#c{position:absolute;left:0;right:0;bottom:20px;text-align:center;color:#b7ccdf;font-size:15px;pointer-events:none}
#c small{display:block;color:#8ba6bd;font-size:11px;letter-spacing:2px}
#c b.s{display:block;margin:4px 0;font:900 23px Orbitron,sans-serif;letter-spacing:2px;color:__COLOR__;text-shadow:0 0 18px __COLOR__}
</style></head><body><div id="w"><div id="c"><small>LIVE DEFENSE STATUS</small><b class="s">__STATE__</b>Threat index <b>__RISK__/100</b> &middot; __ACTION__</div></div>
<script>
const W=document.getElementById('w'),H=430,col=new THREE.Color('__COLOR__'),sp=1+__RISKNUM__/45;
const R=new THREE.WebGLRenderer({antialias:true,alpha:true});R.setPixelRatio(Math.min(devicePixelRatio,2));R.setSize(W.clientWidth,H);W.insertBefore(R.domElement,W.firstChild);
const S=new THREE.Scene(),C=new THREE.PerspectiveCamera(45,W.clientWidth/H,.1,100);C.position.set(0,.35,6.6);
S.add(new THREE.AmbientLight(0x88aaff,.55));
const L1=new THREE.PointLight(0x22d3ee,1.6,30);L1.position.set(3,3,4);S.add(L1);
const L2=new THREE.PointLight(0xa78bfa,1.3,30);L2.position.set(-3,-2,3);S.add(L2);
const sh=new THREE.Shape();sh.moveTo(0,1.5);sh.bezierCurveTo(.5,1.2,.9,1.15,1.2,1.2);sh.lineTo(1.2,.1);sh.bezierCurveTo(1.2,-.7,.7,-1.2,0,-1.6);sh.bezierCurveTo(-.7,-1.2,-1.2,-.7,-1.2,.1);sh.lineTo(-1.2,1.2);sh.bezierCurveTo(-.9,1.15,-.5,1.2,0,1.5);
const geo=new THREE.ExtrudeGeometry(sh,{depth:.35,bevelEnabled:true,bevelThickness:.09,bevelSize:.08,bevelSegments:3,curveSegments:24});geo.center();
const g=new THREE.Group();S.add(g);g.position.y=.3;
const mat=new THREE.MeshPhysicalMaterial({color:0x0b3a5c,emissive:col,emissiveIntensity:.3,metalness:.6,roughness:.25,clearcoat:1,transparent:true,opacity:.66});
g.add(new THREE.Mesh(geo,mat));
g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo,25),new THREE.LineBasicMaterial({color:col})));
const wf=new THREE.Mesh(geo,new THREE.MeshBasicMaterial({color:0x22d3ee,wireframe:true,transparent:true,opacity:.08}));wf.scale.setScalar(1.02);g.add(wf);
const core=new THREE.Mesh(new THREE.IcosahedronGeometry(.42,1),new THREE.MeshBasicMaterial({color:0xa78bfa,wireframe:true}));core.position.z=.05;g.add(core);
const glow=new THREE.Mesh(new THREE.SphereGeometry(.2,24,24),new THREE.MeshBasicMaterial({color:col}));g.add(glow);
const rings=[];[[2.3,1.25,0],[2.6,.4,1.0],[2.9,-.8,.5]].forEach((p,i)=>{const r=new THREE.Mesh(new THREE.TorusGeometry(p[0],.012,8,140),new THREE.MeshBasicMaterial({color:i==1?0xa78bfa:0x22d3ee,transparent:true,opacity:.55}));r.rotation.set(p[1],p[2],0);const d=new THREE.Mesh(new THREE.SphereGeometry(.07,12,12),new THREE.MeshBasicMaterial({color:i==1?0xa78bfa:0x22d3ee}));d.position.x=p[0];r.add(d);g.add(r);rings.push(r);});
const scan=new THREE.Mesh(new THREE.TorusGeometry(1.32,.014,8,80),new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.75}));scan.rotation.x=Math.PI/2;g.add(scan);
const N=550,pa=new Float32Array(N*3);for(let i=0;i<N;i++){const r=3+Math.random()*3,a=Math.random()*6.283,b=Math.acos(2*Math.random()-1);pa[i*3]=r*Math.sin(b)*Math.cos(a);pa[i*3+1]=r*Math.sin(b)*Math.sin(a);pa[i*3+2]=r*Math.cos(b);}
const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(pa,3));
const pts=new THREE.Points(pg,new THREE.PointsMaterial({color:0x60a5fa,size:.035,transparent:true,opacity:.8}));S.add(pts);
let mx=0,my=0;W.addEventListener('mousemove',e=>{const b=W.getBoundingClientRect();mx=(e.clientX-b.left)/b.width-.5;my=(e.clientY-b.top)/b.height-.5;});
W.addEventListener('mouseleave',()=>{mx=0;my=0;});
const clk=new THREE.Clock();
(function loop(){requestAnimationFrame(loop);const t=clk.getElapsedTime();
g.rotation.y+=((Math.sin(t*.6*sp)*.55+mx*.9)-g.rotation.y)*.06;g.rotation.x+=((my*.5)-g.rotation.x)*.06;g.position.y=.3+Math.sin(t*1.3)*.08;
core.rotation.x=t*1.2*sp;core.rotation.y=t*1.6*sp;glow.scale.setScalar(1+.25*Math.sin(t*3*sp));
mat.emissiveIntensity=.3+.22*Math.sin(t*2*sp);rings.forEach((r,i)=>{r.rotation.z=t*(.35+i*.2)*sp*(i%2?-1:1);});
scan.position.y=Math.sin(t*1.6)*1.3;scan.material.opacity=.35+.4*Math.abs(Math.cos(t*1.6));pts.rotation.y=t*.05*sp;pts.rotation.x=t*.02;
R.render(S,C);})();
addEventListener('resize',()=>{R.setSize(W.clientWidth,H);C.aspect=W.clientWidth/H;C.updateProjectionMatrix();});
</script></body></html>'''


def hero_html(color, risk, state, action):
    return (HERO_TPL.replace("__COLOR__", color).replace("__RISKNUM__", str(float(risk)))
            .replace("__RISK__", f"{risk:.2f}").replace("__STATE__", state)
            .replace("__ACTION__", action.replace("_", " ")))


def rotating_plot(fig, height=500):
    return f"""<html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>html,body{{margin:0;background:transparent}}#p{{height:{height}px;border:1px solid rgba(87,186,255,.16);border-radius:22px;overflow:hidden;background:linear-gradient(145deg,rgba(13,28,49,.86),rgba(4,11,22,.9))}}</style></head>
<body><div id="p"></div><script>
const f={fig.to_json()};let a=0.78,spin=true;
Plotly.newPlot('p',f.data,f.layout,{{displayModeBar:false,responsive:true}});
const el=document.getElementById('p');el.addEventListener('mousedown',()=>spin=false);el.addEventListener('mouseleave',()=>spin=true);
(function t(){{if(spin){{a+=.004;Plotly.relayout('p',{{'scene.camera.eye':{{x:1.9*Math.cos(a),y:1.9*Math.sin(a),z:1.15}}}});}}requestAnimationFrame(t);}})();
</script></body></html>"""


def safe_json(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def percent(value):
    try:
        return float(value) * 100.0
    except (TypeError, ValueError):
        return 0.0


def risk_level(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "LOW"
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 30:
        return "MEDIUM"
    return "LOW"


def level_color(level):
    return {
        "LOW": "#34d399",
        "MEDIUM": "#fbbf24",
        "HIGH": "#fb923c",
        "CRITICAL": "#fb7185",
    }.get(level, "#22d3ee")


def clean_endpoint(value):
    return str(value or "microgrid-simulator").replace("microgrid-simulator-", "")


# ============================================================
# DATA
# ============================================================

summary = safe_json(SUMMARY_URL)
events_payload = safe_json(RECENT_URL, {"limit": 100})
health = safe_json(HEALTH_URL)
model_info = safe_json(MODEL_URL)

if summary is None:
    st.error("AI-CyberShield backend is unavailable. Check the FastAPI ECS service and ALB connection.")
    st.stop()

if isinstance(events_payload, dict):
    events = events_payload.get("events", events_payload.get("data", []))
else:
    events = events_payload if isinstance(events_payload, list) else []

events_df = pd.DataFrame(events)

if not events_df.empty:
    if "timestamp" in events_df.columns:
        events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], errors="coerce")
        events_df = events_df.sort_values("timestamp")

    for col in ["attack_probability", "anomaly_score", "risk_score", "criticality"]:
        if col in events_df.columns:
            events_df[col] = pd.to_numeric(events_df[col], errors="coerce")

risk_distribution = summary.get("risk_distribution", {}) or {}
critical_count = int(risk_distribution.get("CRITICAL", 0) or 0)
high_count = int(risk_distribution.get("HIGH", 0) or 0)
medium_count = int(risk_distribution.get("MEDIUM", 0) or 0)
low_count = int(risk_distribution.get("LOW", 0) or 0)

latest = events_df.iloc[-1].to_dict() if not events_df.empty else {}
latest_attack = float(latest.get("attack_probability", 0) or 0)
latest_anomaly = float(latest.get("anomaly_score", 0) or 0)
latest_criticality = float(latest.get("criticality", 0.5) or 0.5)
latest_risk = float(latest.get("risk_score", summary.get("average_risk_score", 0)) or 0)
latest_level = str(latest.get("risk_level") or risk_level(latest_risk))
latest_action = str(latest.get("recommended_action", "NORMAL_OPERATION"))
latest_endpoint = clean_endpoint(latest.get("endpoint", "microgrid-simulator"))

state_text = {
    "LOW": "SYSTEM SECURE",
    "MEDIUM": "ELEVATED RISK",
    "HIGH": "HIGH RISK",
    "CRITICAL": "CRITICAL THREAT",
}.get(latest_level, "MONITORING")
state_color = level_color(latest_level)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="logo">🛡️</div>
            <div>
                <div class="brand-title">AI-CyberShield</div>
                <div class="brand-subtitle">Microgrid Cybersecurity Command Center</div>
            </div>
        </div>
        <div class="badges">
            <div class="badge">🟢 AWS ONLINE</div>
            <div class="badge">🧠 AI ENGINE READY</div>
            <div class="badge">📡 AUDIT STREAM LIVE</div>
            <div class="badge">◉ SIMULATED TELEMETRY</div>
            <div class="badge">🔒 PHYSICAL ACTUATION OFF</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3D SHIELD + SIGNAL MATRIX
# ============================================================

ticks = []
for _, r in events_df.tail(14).iterrows():
    sc = float(r.get("risk_score", 0) or 0)
    ticks.append(f'<span style="color:{level_color(risk_level(sc))}">◆</span> {clean_endpoint(r.get("endpoint"))} · risk {sc:.1f}')
ticker = " &nbsp;&nbsp;&nbsp;&nbsp; ".join(ticks) or "Awaiting telemetry…"
st.markdown(f'<div class="ticker"><div>{ticker}</div></div>', unsafe_allow_html=True)

hero_col, sig_col = st.columns([1.16, .84], gap="medium")
with hero_col:
    components.html(hero_html(state_color, latest_risk, state_text, latest_action), height=436)
with sig_col:
    st.markdown(
        f"""
        <div class="signals">
            <div class="section-title">⚡ Live Signal Matrix</div>
            <div class="signal">
                <div class="signal-head"><span class="signal-label">Attack Probability</span><b class="signal-number">{percent(latest_attack):.1f}%</b></div>
                <div class="signal-bar"><div class="signal-fill" style="width:{max(0,min(100,percent(latest_attack))):.1f}%"></div></div>
            </div>
            <div class="signal">
                <div class="signal-head"><span class="signal-label">Anomaly Score</span><b class="signal-number">{percent(latest_anomaly):.1f}%</b></div>
                <div class="signal-bar"><div class="signal-fill" style="width:{max(0,min(100,percent(latest_anomaly))):.1f}%"></div></div>
            </div>
            <div class="signal">
                <div class="signal-head"><span class="signal-label">Asset Criticality</span><b class="signal-number">{percent(latest_criticality):.1f}%</b></div>
                <div class="signal-bar"><div class="signal-fill" style="width:{max(0,min(100,percent(latest_criticality))):.1f}%"></div></div>
            </div>
            <div class="signal">
                <div class="signal-head"><span class="signal-label">Composite Risk</span><b class="signal-number">{latest_risk:.2f}/100</b></div>
                <div class="signal-bar"><div class="signal-fill" style="width:{max(0,min(100,latest_risk)):.1f}%"></div></div>
            </div>

            <div class="live-grid">
                <div class="live-cell"><div class="live-label">Latest Endpoint</div><div class="live-value">{latest_endpoint}</div></div>
                <div class="live-cell"><div class="live-label">Defense Mode</div><div class="live-value">{latest_action.replace('_', ' ')}</div></div>
                <div class="live-cell"><div class="live-label">Attack Signal</div><div class="live-value">{'DETECTED' if bool(latest.get('attack_prediction', False)) else 'CLEAR'}</div></div>
                <div class="live-cell"><div class="live-label">Anomaly Signal</div><div class="live-value">{'ANOMALY' if bool(latest.get('anomaly_prediction', False)) else 'NORMAL'}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 3D THREAT SPACE
# ============================================================

if not events_df.empty and {"attack_probability", "anomaly_score", "risk_score"}.issubset(events_df.columns):
    threat_df = events_df.tail(100).copy()
    threat_df["attack_probability"] = threat_df["attack_probability"].fillna(0).clip(0, 1)
    threat_df["anomaly_score"] = threat_df["anomaly_score"].fillna(0).clip(0, 1)
    threat_df["risk_score"] = threat_df["risk_score"].fillna(0).clip(0, 100)
    threat_df["risk_level"] = threat_df.apply(
        lambda row: str(row.get("risk_level") or risk_level(row["risk_score"])),
        axis=1,
    )

    colors = threat_df["risk_level"].map({
        "LOW": "#34d399",
        "MEDIUM": "#fbbf24",
        "HIGH": "#fb923c",
        "CRITICAL": "#fb7185",
    }).fillna("#22d3ee")

    fig = go.Figure(go.Scatter3d(
        x=threat_df["attack_probability"],
        y=threat_df["anomaly_score"],
        z=threat_df["risk_score"],
        mode="markers",
        marker=dict(
            size=(4 + threat_df["risk_score"] / 12).tolist(),
            color=colors.tolist(),
            opacity=.9,
            line=dict(width=.5, color="rgba(255,255,255,.42)"),
        ),
        customdata=threat_df[["risk_level", "endpoint"]].fillna("").to_numpy()
        if "endpoint" in threat_df.columns else threat_df[["risk_level"]].fillna("").to_numpy(),
        hovertemplate=(
            "<b>Security Event</b><br>"
            "Attack: %{x:.3f}<br>"
            "Anomaly: %{y:.3f}<br>"
            "Risk: %{z:.2f}<br>"
            "Level: %{customdata[0]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a7bdd2", size=10),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Attack Probability", range=[0,1], gridcolor="rgba(96,165,250,.12)"),
            yaxis=dict(title="Anomaly Score", range=[0,1], gridcolor="rgba(167,139,250,.11)"),
            zaxis=dict(title="Risk Score", range=[0,100], gridcolor="rgba(34,211,238,.11)"),
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.18)),
        ),
        showlegend=False,
    )
    fig.add_trace(go.Scatter3d(x=threat_df["attack_probability"], y=threat_df["anomaly_score"], z=threat_df["risk_score"],
                               mode="lines", line=dict(color="rgba(34,211,238,.35)", width=2), hoverinfo="skip"))
    fig.add_trace(go.Mesh3d(x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[80] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                            color="#fb7185", opacity=.12, hoverinfo="skip"))
    st.markdown('<div class="section-title">🌌 3D Threat Space · Attack × Anomaly × Risk <span style="opacity:.6">(auto-rotating · drag to take control)</span></div>', unsafe_allow_html=True)
    components.html(rotating_plot(fig, 500), height=506)


# ============================================================
# SECURITY OVERVIEW
# ============================================================

overall_status = (
    "CRITICAL SECURITY EVENT DETECTED" if critical_count > 0 else
    "HIGH-RISK SECURITY EVENT DETECTED" if high_count > 0 else
    "ELEVATED SECURITY ACTIVITY" if medium_count > 0 else
    "NO HIGH OR CRITICAL SECURITY EVENTS DETECTED"
)
overall_color = "#fb7185" if critical_count > 0 else "#fb923c" if high_count > 0 else "#fbbf24" if medium_count > 0 else "#34d399"

st.markdown(
    f"""
    <div class="status-rail">
        <div><span class="status-dot" style="background:{overall_color}; color:{overall_color};"></span><span class="status-text" style="color:{overall_color};">{overall_status}</span></div>
        <div class="status-meta">AWS audit database · 5s refresh · ML detection + deterministic safety policy</div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpis = [
    ("TOTAL EVENTS", summary.get("total_events", 0), "audit history"),
    ("ATTACK SIGNALS", summary.get("attack_events", 0), "XGBoost detector"),
    ("ANOMALIES", summary.get("anomaly_events", 0), "Isolation Forest"),
    ("ACTIVE ALERTS", summary.get("alert_events", 0), "safety engine"),
    ("AVERAGE RISK", f"{float(summary.get('average_risk_score', 0) or 0):.2f}", "0–100 composite"),
]

kpi_html = '<div class="metric-grid">'
for label, value, note in kpis:
    shown = f'<span class="count" style="--n:{value}"></span>' if isinstance(value, int) else value
    kpi_html += f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{shown}</div><div class="metric-note">{note}</div></div>'
kpi_html += '</div>'
st.markdown(kpi_html, unsafe_allow_html=True)


# ============================================================
# ALERT COMMAND CENTER
# ============================================================

left, right = st.columns([1.03, .97], gap="large")

with left:
    st.markdown('<div class="section-title">🚨 Alert Command Center</div>', unsafe_allow_html=True)

    alerts_df = events_df.copy()
    if not alerts_df.empty and "alert" in alerts_df.columns:
        alerts_df = alerts_df[alerts_df["alert"].fillna(False).astype(bool)].sort_values("timestamp", ascending=False)
    else:
        alerts_df = pd.DataFrame()

    if alerts_df.empty:
        st.success("No active alerts in the latest audit stream.")
    else:
        for _, row in alerts_df.head(8).iterrows():
            lvl = str(row.get("risk_level") or risk_level(row.get("risk_score", 0)))
            color = level_color(lvl)
            st.markdown(
                f"""
                <div class="alert-card" style="border-left:3px solid {color}; box-shadow: inset 8px 0 22px -20px {color};">
                    <div class="alert-top"><span>{lvl} SECURITY EVENT</span><span>{pd.to_datetime(row.get('timestamp'), errors='coerce').strftime('%d %b %H:%M:%S') if pd.notna(pd.to_datetime(row.get('timestamp'), errors='coerce')) else '—'}</span></div>
                    <div class="alert-title">🛰️ {clean_endpoint(row.get('endpoint', 'microgrid-simulator'))}</div>
                    <div class="alert-metrics">
                        <div class="alert-metric"><span>Risk</span><b>{float(row.get('risk_score', 0) or 0):.2f}</b></div>
                        <div class="alert-metric"><span>Attack</span><b>{percent(row.get('attack_probability', 0)):.1f}%</b></div>
                        <div class="alert-metric"><span>Anomaly</span><b>{percent(row.get('anomaly_score', 0)):.1f}%</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right:
    st.markdown('<div class="section-title">📡 Threat Pulse</div>', unsafe_allow_html=True)
    if not events_df.empty and "timestamp" in events_df.columns:
        pulse = events_df.tail(100).copy()
        pulse["risk_score"] = pd.to_numeric(pulse.get("risk_score", 0), errors="coerce").fillna(0)
        pulse["attack_probability"] = pd.to_numeric(pulse.get("attack_probability", 0), errors="coerce").fillna(0) * 100
        pulse["anomaly_score"] = pd.to_numeric(pulse.get("anomaly_score", 0), errors="coerce").fillna(0) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pulse["timestamp"], y=pulse["risk_score"], mode="lines+markers", fill="tozeroy", fillcolor="rgba(34,211,238,.10)", name="Risk", line=dict(width=3, color="#22d3ee"), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=pulse["timestamp"], y=pulse["attack_probability"], mode="lines", name="Attack", line=dict(width=2, dash="dot", color="#fb7185")))
        fig.add_trace(go.Scatter(x=pulse["timestamp"], y=pulse["anomaly_score"], mode="lines", name="Anomaly", line=dict(width=2, dash="dash", color="#a78bfa")))
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a8bed3", size=10),
            xaxis=dict(gridcolor="rgba(120,180,255,.07)", zeroline=False),
            yaxis=dict(gridcolor="rgba(120,180,255,.07)", zeroline=False),
            legend=dict(orientation="h", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Threat pulse will appear after telemetry arrives.")


# ============================================================
# RISK DISTRIBUTION + SCENARIOS
# ============================================================

left, right = st.columns([.78, 1.22], gap="large")

with left:
    st.markdown('<div class="section-title">🎯 Risk Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=[low_count, medium_count, high_count, critical_count],
        y=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        orientation="h",
        marker=dict(color=["#34d399", "#fbbf24", "#fb923c", "#fb7185"]),
        text=[low_count, medium_count, high_count, critical_count],
        textposition="outside",
    ))
    fig.update_layout(height=360, margin=dict(l=0,r=25,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#a8bed3", size=10), xaxis=dict(gridcolor="rgba(120,180,255,.07)"), yaxis=dict(gridcolor="rgba(120,180,255,.05)"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    st.markdown('<div class="section-title">🎯 Scenario Threat Matrix</div>', unsafe_allow_html=True)
    if not events_df.empty:
        work = events_df.copy()
        work["scenario"] = work.get("endpoint", pd.Series("microgrid-simulator", index=work.index)).astype(str).map(clean_endpoint)
        for col in ["attack_prediction", "anomaly_prediction", "alert", "risk_score"]:
            if col not in work.columns:
                work[col] = 0
        grouped = work.groupby("scenario").agg(
            events=("scenario", "size"),
            attacks=("attack_prediction", "sum"),
            anomalies=("anomaly_prediction", "sum"),
            alerts=("alert", "sum"),
            max_risk=("risk_score", "max"),
        ).reset_index().sort_values("max_risk", ascending=False)

        html = '<div class="scenario-grid">'
        for _, row in grouped.iterrows():
            score = float(row.get("max_risk", 0) or 0)
            lvl = risk_level(score)
            col = level_color(lvl)
            html += f'<div class="scenario" style="border-color:{col}44"><div class="scenario-name">{row["scenario"]}</div><div class="scenario-state" style="color:{col}">{lvl}</div><div class="scenario-meta">Events {int(row["events"])} · Attacks {int(row["attacks"])} · Anomalies {int(row["anomalies"])}<br>Alerts {int(row["alerts"])} · Max Risk {score:.1f}</div></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("Scenario monitoring will populate when telemetry arrives.")


# ============================================================
# DEFENSE TOPOLOGY
# ============================================================

st.markdown('<div class="section-title">🌐 Defense Topology</div>', unsafe_allow_html=True)
nodes = [("📡", "Telemetry"), ("🌐", "AWS ALB"), ("⚙️", "FastAPI"), ("🧠", "ML Engine"), ("⚖️", "Risk Engine"), ("🛡️", "Safety Engine"), ("🗄️", "PostgreSQL")]
html = '<div class="topology">'
for i, (icon, name) in enumerate(nodes):
    html += f'<div class="node"><div class="node-icon">{icon}</div><div class="node-name">{name}</div></div>'
    if i < len(nodes) - 1:
        html += '<div class="arrow">→</div>'
html += '</div>'
st.markdown(html, unsafe_allow_html=True)


# ============================================================
# MODEL / SAFETY STATUS
# ============================================================

st.markdown('<div class="section-title" style="margin-top:18px;">🧠 Model Intelligence & Safety</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
base_features = model_info.get("base_features", "—") if isinstance(model_info, dict) else "—"
temporal_features = model_info.get("temporal_features", "—") if isinstance(model_info, dict) else "—"
total_features = model_info.get("total_model_features", "—") if isinstance(model_info, dict) else "—"
m1.metric("Base Features", base_features)
m2.metric("Temporal Features", temporal_features)
m3.metric("Total Model Features", total_features)
m4.metric("Physical Breaker Control", "DISABLED")
st.info("ML performs detection and risk assessment. The deterministic safety layer does not directly actuate physical breakers.")


# ============================================================
# SHAP / LATEST EVENT EXPLANATION
# ============================================================

if latest and latest.get("shap_values") is not None:
    st.markdown('<div class="section-title" style="margin-top:18px;">🔍 Explainable AI · Latest Event</div>', unsafe_allow_html=True)
    shap_data = latest.get("shap_values")
    rows = []
    if isinstance(shap_data, list):
        for item in shap_data:
            if isinstance(item, dict):
                feature = item.get("feature") or item.get("name") or item.get("feature_name")
                value = item.get("shap_value", item.get("value"))
                try:
                    if feature is not None and value is not None:
                        rows.append((str(feature), float(value)))
                except (TypeError, ValueError):
                    pass
    elif isinstance(shap_data, dict):
        for feature, value in shap_data.items():
            try:
                rows.append((str(feature), float(value)))
            except (TypeError, ValueError):
                pass

    if rows:
        shap_df = pd.DataFrame(rows, columns=["Feature", "SHAP Value"]).sort_values("SHAP Value")
        shap_df = pd.concat([shap_df.head(8), shap_df.tail(8)]).drop_duplicates()
        fig = go.Figure(go.Bar(x=shap_df["SHAP Value"], y=shap_df["Feature"], orientation="h", marker_color="#22d3ee"))
        fig.update_layout(height=440, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#a8bed3", size=10), xaxis=dict(gridcolor="rgba(120,180,255,.07)"), yaxis=dict(gridcolor="rgba(120,180,255,.06)"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================
# RECENT EVENTS
# ============================================================

st.markdown('<div class="section-title" style="margin-top:18px;">📋 Latest Security Events</div>', unsafe_allow_html=True)
if not events_df.empty:
    table_cols = [
        "timestamp", "endpoint", "attack_probability", "attack_prediction",
        "anomaly_score", "anomaly_prediction", "risk_score", "risk_level",
        "recommended_action", "alert",
    ]
    cols = [c for c in table_cols if c in events_df.columns]
    table = events_df.sort_values("timestamp", ascending=False)[cols].head(100).copy()
    if "timestamp" in table.columns:
        table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "attack_probability" in table.columns:
        table["attack_probability"] = (table["attack_probability"] * 100).round(2)
    if "anomaly_score" in table.columns:
        table["anomaly_score"] = (table["anomaly_score"] * 100).round(2)
    if "risk_score" in table.columns:
        table["risk_score"] = table["risk_score"].round(2)
    st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.info("No audit events available yet.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"<div class=\"footer\">AI-CyberShield · XGBoost + Isolation Forest + SHAP · Last refresh {datetime.now().strftime('%d %b %Y %H:%M:%S')} · Simulated telemetry for controlled demonstration</div>",
    unsafe_allow_html=True,
)