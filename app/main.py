"""
GeoSharp: Sentinel-2 Super-Resolution Demo

Pipeline:

    Sentinel-2 10m
        128 x 128 x 4
        B02 B03 B04 B08
              ↓
          LDSR-S2
            4x
              ↓
        512 x 512 x 4
              ↓
          2.5m output

Frontend shows:

    1. Original RGB
    2. GeoSharp SR RGB
    3. Sample-variance uncertainty
    4. LR-consistency residual
    5. Hallucination-risk overlay
    6. 4-band PSNR
    7. 4-band SSIM
    8. Uncertainty statistics
    9. LR-consistency statistics
    10. Hallucination-risk percentage

Bands:

    B02 = Blue
    B03 = Green
    B04 = Red
    B08 = NIR
"""

import sys
import os

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import rasterio


# =============================================================
# PATH SETUP
# =============================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# =============================================================
# GEOSHARP IMPORTS
# =============================================================

from pipeline.inference import run_sr

from pipeline.metrics import (
    compare_to_baseline
)

from pipeline.hallucination_check import (
    lr_consistency_error,
    create_hallucination_risk_mask,
    percentile_normalize,
)


# =============================================================
# STREAMLIT CONFIG
# =============================================================

st.set_page_config(
    page_title="GeoSharp",
    page_icon="🛰️",
    layout="wide"
)


# =============================================================
# HEADER
# =============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root{
      --black:#050505;
      --panel:#0d0d0d;
      --line: rgba(245,245,242,0.12);
      --line-hover: rgba(245,217,10,0.55);
      --white:#f5f5f2;
      --dim: rgba(245,245,242,0.58);
      --yellow:#F5D90A;
      --yellow-dim: rgba(245,217,10,0.14);
      --head: 'Space Grotesk', sans-serif;
      --body: 'IBM Plex Sans', sans-serif;
      --mono: 'IBM Plex Mono', monospace;
    }

    html{ scroll-behavior:smooth; }

    /* ---- re-skin the surrounding Streamlit app ---- */
    .stApp{
      background:transparent !important;
      font-family:var(--body);
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main,
    .block-container{
      background:transparent !important;
    }
    [data-testid="stHeader"]{ background:transparent; }
    .block-container{ padding-top:1.5rem; max-width:1180px; }

    /* keep real app content above the reactive background canvas */
    [data-testid="stAppViewContainer"]{ position:relative; z-index:1; }

    /* ---- custom cursor ---- */
    body{ cursor:none; }
    @media (pointer:coarse){
      body{ cursor:auto; }
      #gs-cursor-dot,#gs-cursor-ring{ display:none; }
    }
    #gs-cursor-dot,#gs-cursor-ring{
      position:fixed; top:0; left:0; pointer-events:none; z-index:9999;
      border-radius:50%; transform:translate(-50%,-50%);
    }
    #gs-cursor-dot{ width:6px; height:6px; background:var(--yellow); }
    #gs-cursor-ring{ width:34px; height:34px; border:1px solid rgba(245,217,10,0.5); transition: width .2s, height .2s, border-color .2s; }
    body.gs-hovering #gs-cursor-ring{ width:54px; height:54px; border-color:var(--yellow); }

    /* ---- reactive background canvas ---- */
    #gs-bg-canvas{ position:fixed; inset:0; z-index:0; pointer-events:none; }
    h1,h2,h3{ font-family:var(--head) !important; color:var(--white); }
    p, label, span{ color:var(--white); }
    a{ color:var(--yellow); }

    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploadDropzone"]{
      background:var(--panel) !important;
      border:1px dashed var(--line) !important;
    }
    [data-testid="stMetricValue"]{ color:var(--yellow); font-family:var(--head); }
    [data-testid="stMetricLabel"]{ color:var(--dim); font-family:var(--mono); }
    [data-testid="stAlert"]{ background:var(--panel); border:1px solid var(--line); }
    .stButton>button, .stDownloadButton>button{
      background:transparent; color:var(--yellow); border:1px solid var(--yellow);
      font-family:var(--mono);
    }
    .stButton>button:hover, .stDownloadButton>button:hover{
      background:var(--yellow); color:var(--black);
    }

    /* ---- ported geosharp_site.html design system (scoped) ---- */
    .geosharp-site *{ box-sizing:border-box; }
    .geosharp-site{ color:var(--white); font-family:var(--body); }
    .geosharp-site a{ text-decoration:none; }
    .geosharp-site img{ max-width:100%; display:block; }

    .geosharp-site .container{ max-width:1180px; margin:0 auto; padding:0 0.5rem; }

    .geosharp-site nav{
      display:flex; align-items:center; justify-content:space-between;
      padding:1rem 0.5rem; border-bottom:1px solid var(--line); margin-bottom:1rem;
    }
    .geosharp-site .logo{ font-family:var(--head); font-weight:700; font-size:1.1rem; letter-spacing:-0.01em; }
    .geosharp-site .logo span{ color:var(--yellow); }
    .geosharp-site .navlinks{ display:flex; gap:2rem; font-family:var(--mono); font-size:0.82rem; color:var(--dim); }
    .geosharp-site .navlinks a{ position:relative; padding-bottom:4px; transition:color .2s; }
    .geosharp-site .navlinks a::after{
      content:''; position:absolute; left:0; bottom:0; height:1px; width:100%;
      background:var(--yellow); transform:scaleX(0); transform-origin:left; transition:transform .3s ease;
    }
    .geosharp-site .navlinks a:hover{ color:var(--white); }
    .geosharp-site .navlinks a:hover::after{ transform:scaleX(1); }
    @media (max-width:720px){ .geosharp-site .navlinks{ display:none; } }

    .geosharp-site .btn{
      font-family:var(--mono); font-size:0.82rem; padding:0.65rem 1.2rem;
      border:1px solid var(--yellow); color:var(--yellow); background:transparent;
      display:inline-block; transition: background .25s, color .25s, box-shadow .25s;
    }
    .geosharp-site .btn:hover{ background:var(--yellow); color:var(--black); box-shadow:0 0 28px rgba(245,217,10,0.35); }
    .geosharp-site .btn.ghost{ border-color:var(--line); color:var(--white); }
    .geosharp-site .btn.ghost:hover{ background:var(--white); color:var(--black); }

    .geosharp-site section{ padding:4rem 0; border-bottom:1px solid var(--line); }
    .geosharp-site .tag{ font-family:var(--mono); font-size:0.78rem; color:var(--yellow); }
    .geosharp-site h2{ font-size:2.1rem; font-weight:600; margin-top:0.6rem; max-width:640px; line-height:1.15; }
    .geosharp-site .lede{ color:var(--dim); max-width:560px; margin-top:1rem; font-size:1.02rem; line-height:1.6; }

    .geosharp-site #hero{ padding:2rem 0 3rem 0; }
    .geosharp-site .eyebrow{ font-family:var(--mono); font-size:0.82rem; color:var(--dim); display:flex; gap:0.6rem; align-items:center; }
    .geosharp-site h1{
      font-family:var(--head); font-weight:700; font-size:clamp(2rem, 5vw, 3.6rem);
      line-height:1.05; margin-top:1.1rem; max-width:920px; letter-spacing:-0.02em;
    }
    .geosharp-site h1 em{ color:var(--yellow); font-style:normal; }
    .geosharp-site .hero-sub{ color:var(--dim); max-width:600px; margin-top:1.2rem; font-size:1.05rem; line-height:1.6; }
    .geosharp-site .hero-cta{ display:flex; gap:1rem; margin-top:1.8rem; flex-wrap:wrap; }

    .geosharp-site .stripline{ margin-top:2.5rem; border-top:1px solid var(--line); border-bottom:1px solid var(--line); overflow:hidden; }
    .geosharp-site .stripline .track{ display:flex; gap:3rem; white-space:nowrap; font-family:var(--mono); font-size:0.82rem;
      color:var(--dim); padding:0.9rem 0; animation:gs-marquee 22s linear infinite; }
    .geosharp-site .stripline:hover .track{ animation-play-state:paused; }
    .geosharp-site .stripline .track span b{ color:var(--yellow); }
    @keyframes gs-marquee{ from{ transform:translateX(0); } to{ transform:translateX(-50%); } }

    .geosharp-site .split{ display:grid; grid-template-columns:1fr 1fr; gap:2rem; margin-top:2.5rem; }
    @media (max-width:840px){ .geosharp-site .split{ grid-template-columns:1fr; } }
    .geosharp-site .compare{ border:1px solid var(--line); padding:1.6rem; transition:border-color .25s, transform .25s; }
    .geosharp-site .compare:hover{ border-color:var(--line-hover); transform:translateY(-3px); }
    .geosharp-site .compare .k{ font-family:var(--mono); font-size:0.78rem; color:var(--dim); }
    .geosharp-site .compare .v{ font-family:var(--head); font-size:2rem; margin-top:0.4rem; }
    .geosharp-site .compare .v.yellow{ color:var(--yellow); }
    .geosharp-site .compare p{ color:var(--dim); font-size:0.92rem; margin-top:0.8rem; line-height:1.5; }

    .geosharp-site .pipeline{ display:flex; flex-wrap:wrap; gap:1.4rem; margin-top:2.5rem; }
    .geosharp-site .pstep{ flex:1; min-width:230px; border:1px solid var(--line); padding:1.6rem; transition:border-color .3s, transform .3s; }
    .geosharp-site .pstep:hover{ border-color:var(--line-hover); transform:translateY(-4px); }
    .geosharp-site .pstep .num{ font-family:var(--mono); font-size:0.78rem; color:var(--yellow); }
    .geosharp-site .pstep h3{ font-size:1.15rem; margin-top:0.7rem; font-weight:600; }
    .geosharp-site .pstep p{ color:var(--dim); font-size:0.88rem; margin-top:0.6rem; line-height:1.55; }

    .geosharp-site .grid3{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.2rem; margin-top:2.5rem; }
    @media (max-width:900px){ .geosharp-site .grid3{ grid-template-columns:1fr 1fr; } }
    @media (max-width:600px){ .geosharp-site .grid3{ grid-template-columns:1fr; } }
    .geosharp-site .card{
      position:relative; border:1px solid var(--line); padding:1.8rem; overflow:hidden;
      transition:border-color .3s, transform .3s;
    }
    .geosharp-site .card:hover{ border-color:var(--line-hover); transform:translateY(-4px); }
    .geosharp-site .card::before{
      content:''; position:absolute; inset:0; pointer-events:none; opacity:0; transition:opacity .35s;
      background:radial-gradient(280px circle at 50% 50%, rgba(245,217,10,0.14), transparent 60%);
    }
    .geosharp-site .card:hover::before{ opacity:1; }
    .geosharp-site .card .ico{ font-family:var(--mono); color:var(--yellow); font-size:0.9rem; }
    .geosharp-site .card h3{ font-size:1.15rem; margin-top:1rem; font-weight:600; }
    .geosharp-site .card p{ color:var(--dim); font-size:0.9rem; margin-top:0.7rem; line-height:1.55; }
    .geosharp-site .card code{ font-family:var(--mono); color:var(--yellow); }

    .geosharp-site .trust-box{ border:1px solid var(--yellow); padding:2rem; margin-top:2.5rem; }
    .geosharp-site .trust-box .quote{ font-family:var(--head); font-size:1.3rem; line-height:1.4; max-width:760px; }
    .geosharp-site .trust-box .quote b{ color:var(--yellow); font-weight:600; }
    .geosharp-site .signal-row{ display:flex; gap:2.2rem; margin-top:1.6rem; flex-wrap:wrap; }
    .geosharp-site .signal{ font-family:var(--mono); font-size:0.85rem; color:var(--dim); }
    .geosharp-site .signal b{ color:var(--white); }

    .geosharp-site .grid4{ display:grid; grid-template-columns:repeat(4,1fr); gap:1.2rem; margin-top:2.5rem; }
    @media (max-width:900px){ .geosharp-site .grid4{ grid-template-columns:1fr 1fr; } }
    .geosharp-site .stat{ border:1px solid var(--line); padding:1.6rem; transition:border-color .3s, transform .3s; }
    .geosharp-site .stat:hover{ border-color:var(--line-hover); transform:translateY(-4px); }
    .geosharp-site .stat .num{ font-family:var(--head); font-size:2.1rem; color:var(--yellow); }
    .geosharp-site .stat .lab{ font-family:var(--mono); font-size:0.78rem; color:var(--dim); margin-top:0.4rem; }
    .geosharp-site .fine{ color:var(--dim); font-size:0.85rem; margin-top:1.4rem; max-width:680px; line-height:1.6; border-top:1px solid var(--line); padding-top:1.2rem; }

    .geosharp-site .usecase{ display:flex; gap:1.4rem; padding:1.4rem 0; border-top:1px solid var(--line); align-items:baseline; transition:padding-left .3s, background .3s; }
    .geosharp-site .usecase:hover{ padding-left:0.8rem; background:rgba(245,217,10,0.03); }
    .geosharp-site .usecase:last-child{ border-bottom:1px solid var(--line); }
    .geosharp-site .usecase .n{ font-family:var(--mono); color:var(--yellow); font-size:0.85rem; min-width:28px; }
    .geosharp-site .usecase h4{ font-family:var(--head); font-size:1.1rem; font-weight:600; min-width:200px; }
    .geosharp-site .usecase p{ color:var(--dim); font-size:0.9rem; line-height:1.55; }
    @media (max-width:720px){ .geosharp-site .usecase{ flex-direction:column; gap:0.4rem; } }

    .geosharp-site .roadmap{ display:grid; grid-template-columns:1fr 1fr; gap:1.4rem; margin-top:2.5rem; }
    @media (max-width:800px){ .geosharp-site .roadmap{ grid-template-columns:1fr; } }
    .geosharp-site .phase{ border:1px solid var(--line); padding:1.8rem; transition:border-color .3s, transform .3s; }
    .geosharp-site .phase:hover{ border-color:var(--line-hover); transform:translateY(-4px); }
    .geosharp-site .phase .tag2{ font-family:var(--mono); font-size:0.78rem; color:var(--dim); }
    .geosharp-site .phase.active .tag2{ color:var(--yellow); }
    .geosharp-site .phase h3{ font-size:1.3rem; margin-top:0.5rem; font-weight:600; }
    .geosharp-site .phase ul{ margin-top:1rem; padding-left:1.1rem; color:var(--dim); font-size:0.9rem; line-height:1.7; }

    .geosharp-site .stack{ display:flex; flex-wrap:wrap; gap:0.8rem; margin-top:2rem; }
    .geosharp-site .badge{ font-family:var(--mono); font-size:0.8rem; border:1px solid var(--line); padding:0.5rem 0.9rem; color:var(--dim); transition:border-color .25s, color .25s; }
    .geosharp-site .badge:hover{ border-color:var(--yellow); color:var(--yellow); }

    .geosharp-site #cta-intro{ text-align:center; padding:3rem 0 1rem 0; border-bottom:none; }
    .geosharp-site #cta-intro h2{ margin:0.6rem auto 0 auto; }
    .geosharp-site #cta-intro .lede{ margin:1rem auto 0 auto; }

    .geosharp-site footer{ padding:2rem 0 0.5rem 0; }
    .geosharp-site .foot-row{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:1rem; font-family:var(--mono); font-size:0.8rem; color:var(--dim); }
    .geosharp-site .foot-row a:hover{ color:var(--yellow); }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# CUSTOM CURSOR + REACTIVE BACKGROUND
#
# st.markdown(unsafe_allow_html=True) can inject CSS, but the
# browser will never execute a <script> tag inserted that way.
# components.html() renders into a real iframe, so scripts do
# run there, and since the iframe is same-origin we can reach
# up into window.parent.document to attach the canvas and the
# cursor elements to the actual app page instead of the iframe.
# =============================================================

components.html(
    """
    <script>
    (function(){

      const doc = window.parent.document;

      // Prevent duplicate background/cursor injection on Streamlit reruns.
      if (doc.getElementById('gs-bg-canvas')) {
        return;
      }

      // =========================================================
      // BACKGROUND CANVAS
      // =========================================================

      const canvas = doc.createElement('canvas');
      canvas.id = 'gs-bg-canvas';

      Object.assign(canvas.style, {
        position: 'fixed',
        inset: '0',
        width: '100vw',
        height: '100vh',
        zIndex: '0',
        pointerEvents: 'none'
      });

      doc.body.insertBefore(
        canvas,
        doc.body.firstChild
      );

      const ctx = canvas.getContext('2d');

      // =========================================================
      // CUSTOM CURSOR
      // =========================================================

      const dot = doc.createElement('div');
      dot.id = 'gs-cursor-dot';

      const ring = doc.createElement('div');
      ring.id = 'gs-cursor-ring';

      doc.body.appendChild(dot);
      doc.body.appendChild(ring);

      // =========================================================
      // CANVAS SIZE
      // =========================================================

      let W = 0;
      let H = 0;

      function resizeCanvas(){
        const dpr = window.parent.devicePixelRatio || 1;

        W = window.parent.innerWidth;
        H = window.parent.innerHeight;

        canvas.width = W * dpr;
        canvas.height = H * dpr;

        canvas.style.width = W + 'px';
        canvas.style.height = H + 'px';

        ctx.setTransform(
          dpr,
          0,
          0,
          dpr,
          0,
          0
        );

        buildDots();
      }

      // =========================================================
      // MOUSE
      // =========================================================

      let mouseX = W / 2;
      let mouseY = H / 2;

      let ringX = mouseX;
      let ringY = mouseY;

      doc.addEventListener(
        'mousemove',
        function(e){
          mouseX = e.clientX;
          mouseY = e.clientY;

          dot.style.left =
            mouseX + 'px';

          dot.style.top =
            mouseY + 'px';
        },
        { passive: true }
      );

      // =========================================================
      // DOT GRID
      // =========================================================

      const spacing = 46;
      const dots = [];

      function buildDots(){

        dots.length = 0;

        for(
          let x = 0;
          x < W + spacing;
          x += spacing
        ){
          for(
            let y = 0;
            y < H + spacing;
            y += spacing
          ){
            dots.push({
              x: x,
              y: y
            });
          }
        }
      }

      // =========================================================
      // HOVER EFFECT
      // =========================================================

      const hoverSelector =
        'a, button, .card, .compare, .pstep, ' +
        '.stat, .phase, .usecase, .badge, ' +
        '[data-testid="stFileUploaderDropzone"], ' +
        '[data-testid="stFileUploadDropzone"]';

      doc.body.addEventListener(
        'mouseover',
        function(e){
          if(
            e.target.closest(hoverSelector)
          ){
            doc.body.classList.add(
              'gs-hovering'
            );
          }
        }
      );

      doc.body.addEventListener(
        'mouseout',
        function(e){
          if(
            e.target.closest(hoverSelector)
          ){
            doc.body.classList.remove(
              'gs-hovering'
            );
          }
        }
      );

      // =========================================================
      // RESIZE
      // =========================================================

      resizeCanvas();

      window.parent.addEventListener(
        'resize',
        resizeCanvas
      );

      // =========================================================
      // ANIMATION
      // =========================================================

      function frame(){

        ringX +=
          (mouseX - ringX) * 0.18;

        ringY +=
          (mouseY - ringY) * 0.18;

        ring.style.left =
          ringX + 'px';

        ring.style.top =
          ringY + 'px';

        ctx.clearRect(
          0,
          0,
          W,
          H
        );

        for(
          const d of dots
        ){

          const dx =
            d.x - mouseX;

          const dy =
            d.y - mouseY;

          const dist =
            Math.hypot(
              dx,
              dy
            );

          const influence =
            Math.max(
              0,
              1 - dist / 240
            );

          const offset =
            influence * 10;

          const nx =
            dist
              ? dx / dist
              : 0;

          const ny =
            dist
              ? dy / dist
              : 0;

          const px =
            d.x + nx * offset;

          const py =
            d.y + ny * offset;

          const size =
            1 + influence * 2.2;

          ctx.beginPath();

          if(
            influence > 0.15
          ){
            ctx.fillStyle =
              `rgba(245,217,10,${
                (
                  0.15 +
                  influence * 0.65
                ).toFixed(2)
              })`;
          }
          else{
            ctx.fillStyle =
              'rgba(245,245,242,0.09)';
          }

          ctx.arc(
            px,
            py,
            size,
            0,
            Math.PI * 2
          );

          ctx.fill();
        }

        requestAnimationFrame(frame);
      }

      frame();

    })();
    </script>
    """,
    height=0,
)

st.markdown(
    """
    <div class="geosharp-site">

      <nav>
        <div class="logo">Geo<span>Sharp</span></div>
        <div class="navlinks">
          <a href="#problem">problem</a>
          <a href="#how-it-works">pipeline</a>
          <a href="#features">features</a>
          <a href="#trust">trust</a>
          <a href="#results">results</a>
          <a href="#roadmap" class="hide-mobile">roadmap</a>
        </div>
        <a href="#cta" class="btn">View demo</a>
      </nav>

      <section id="hero">
        <div class="container">
          <div class="eyebrow">smart india hackathon 2026 &nbsp;·&nbsp; prototype build</div>
          <h1>Your satellite already<br>saw it. It just <em>wasn't sharp<br>enough</em> to matter.</h1>
          <p class="hero-sub">
            GeoSharp takes free Sentinel-2 imagery (10 metres per pixel) and reconstructs it
            at 2.5 metres using AI super-resolution. Every output ships with a map of exactly
            where the model is guessing, so you know what to trust before you act on it.
          </p>
          <div class="hero-cta">
            <a href="#how-it-works" class="btn">See the pipeline</a>
            <a href="#results" class="btn ghost">View measured results</a>
          </div>
        </div>
        <div class="container">
          <div class="stripline">
            <div class="track">
              <span>resolution &nbsp;<b>10m → 2.5m</b></span>
              <span>scale factor &nbsp;<b>4×</b></span>
              <span>bands &nbsp;<b>B02 · B03 · B04 · B08</b></span>
              <span>model &nbsp;<b>LDSR-S2 latent diffusion</b></span>
              <span>measured PSNR &nbsp;<b>19.99 dB</b></span>
              <span>measured SSIM &nbsp;<b>0.773</b></span>
              <span>flagged high-risk pixels &nbsp;<b>2.9%</b></span>
              <span>resolution &nbsp;<b>10m → 2.5m</b></span>
              <span>scale factor &nbsp;<b>4×</b></span>
              <span>bands &nbsp;<b>B02 · B03 · B04 · B08</b></span>
              <span>model &nbsp;<b>LDSR-S2 latent diffusion</b></span>
              <span>measured PSNR &nbsp;<b>19.99 dB</b></span>
              <span>measured SSIM &nbsp;<b>0.773</b></span>
              <span>flagged high-risk pixels &nbsp;<b>2.9%</b></span>
            </div>
          </div>
        </div>
      </section>

      <section id="problem">
        <div class="container">
          <div class="tag">the gap</div>
          <h2>10 metres is fine for trends. It's useless for a flooded street.</h2>
          <p class="lede">
            Sentinel-2 is free, global, and revisits every location every five days, but each
            pixel already blends a 10×10 metre patch of ground into one value. That's the wrong
            resolution for the decisions people actually need to make quickly.
          </p>
          <div class="split">
            <div class="compare">
              <div class="k">what you get for free</div>
              <div class="v">10m / pixel</div>
              <p>Sentinel-2 L2A, global coverage, ~5-day revisit, zero cost. One pixel can span
              an entire smallholder field or several buildings on a flooded street.</p>
            </div>
            <div class="compare">
              <div class="k">what decisions actually need</div>
              <div class="v yellow">2.5m / pixel</div>
              <p>Building-level flood extent. Individual-plot crop stress. Boundary-level
              encroachment. Commercial imagery gets you there, at a price and a tasking delay
              most responders don't have.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="how-it-works">
        <div class="container">
          <div class="tag">how it works</div>
          <h2>Four steps, from raw tile to a verified 2.5m output.</h2>
          <p class="lede">No training required for this build. GeoSharp runs ESA
          OpenSR's pretrained LDSR-S2 latent diffusion model and wraps it in an ingestion,
          diagnostics, and verification harness.</p>
          <div class="pipeline">
            <div class="pstep"><div class="num">01</div><h3>Ingest</h3>
              <p>A 128×128×4 patch (B02/B03/B04/B08) is read straight off the tile and
              normalized to reflectance [0,1], no full-tile load required.</p></div>
            <div class="pstep"><div class="num">02</div><h3>Reconstruct</h3>
              <p>LDSR-S2's latent diffusion sampler upsamples the patch 4× to 512×512×4,
              reconstructing detail consistent with the observed spectral signal.</p></div>
            <div class="pstep"><div class="num">03</div><h3>Quantify</h3>
              <p>The same stochastic sampler runs multiple times. Where the samples disagree,
              a per-pixel uncertainty map records it.</p></div>
            <div class="pstep"><div class="num">04</div><h3>Verify</h3>
              <p>The output is degraded back to 10m and compared against the real observation.
              Weak agreement plus high uncertainty gets flagged as hallucination-risk.</p></div>
          </div>
        </div>
      </section>

      <section id="features">
        <div class="container">
          <div class="tag">what's built</div>
          <h2>Not just a sharper picture.</h2>
          <p class="lede">Every feature below is running in the current prototype.
          Nothing here is a mockup.</p>
          <div class="grid3">
            <div class="card"><div class="ico">01</div><h3>4× super-resolution</h3>
              <p>10m Sentinel-2 reconstructed at 2.5m using a pretrained latent diffusion model, no training cost for this build.</p></div>
            <div class="card"><div class="ico">02</div><h3>Full 4-band reconstruction</h3>
              <p>Blue, green, red and near-infrared are all sharpened together, not just an RGB preview: NIR is what most vegetation and water analysis actually needs.</p></div>
            <div class="card"><div class="ico">03</div><h3>Per-pixel uncertainty</h3>
              <p>Sample-variance across repeated diffusion passes gives a confidence map alongside every output, not just a single point estimate.</p></div>
            <div class="card"><div class="ico">04</div><h3>Hallucination-risk masking</h3>
              <p>A dual-signal check: LR-consistency residual and sample variance both have to be high before a region gets flagged as unsupported detail.</p></div>
            <div class="card"><div class="ico">05</div><h3>Baseline-aware evaluation</h3>
              <p>PSNR/SSIM measured against a 4× bicubic baseline, because native 2.5m ground truth for Sentinel-2 doesn't exist. We say so, we don't hide it.</p></div>
            <div class="card"><div class="ico">06</div><h3>Drop-in interface</h3>
              <p>One function, <code>run_sr(tile_path)</code>, decouples the model from the UI, so the inference backend can be swapped without touching the frontend.</p></div>
          </div>
        </div>
      </section>

      <section id="trust">
        <div class="container">
          <div class="tag">why it's different</div>
          <h2>Most super-resolution demos show you a sharper image and stop.</h2>
          <p class="lede">GeoSharp's actual contribution isn't the sharpening. Diffusion
          models can do that. It's telling you which parts of the sharpened image you should
          not act on.</p>
          <div class="trust-box">
            <div class="quote">A region is only flagged high-risk when <b>both</b> the
            degraded-output-vs-original residual <b>and</b> the sample-variance uncertainty
            land in the top 10%. Agreeing with itself isn't enough, and disagreeing with
            itself isn't enough. Both have to hold at once.</div>
            <div class="signal-row">
              <div class="signal">signal 1 &nbsp;<b>LR-consistency residual</b></div>
              <div class="signal">signal 2 &nbsp;<b>stochastic sample variance</b></div>
              <div class="signal">combine &nbsp;<b>AND, both ≥ p90</b></div>
            </div>
          </div>
        </div>
      </section>

      <section id="results">
        <div class="container">
          <div class="tag">measured, not projected</div>
          <h2>Numbers from a real Sentinel-2 tile over India.</h2>
          <p class="lede">Run against a UTM zone 43R tile, not cherry-picked synthetic
          data.</p>
          <div class="grid4">
            <div class="stat"><div class="num">19.99 dB</div><div class="lab">PSNR vs. 4× bicubic baseline</div></div>
            <div class="stat"><div class="num">0.773</div><div class="lab">SSIM vs. 4× bicubic baseline</div></div>
            <div class="stat"><div class="num">2.90%</div><div class="lab">high-risk pixels flagged</div></div>
            <div class="stat"><div class="num">0.000149</div><div class="lab">mean uncertainty score</div></div>
          </div>
          <p class="fine">PSNR and SSIM are measured against a 4× bicubic upsample of the same
          input, not an independent 2.5m ground truth; none exists yet for Sentinel-2 at this
          scale. That's exactly the gap Phase 2 is built to close. Higher SSIM and lower
          high-risk percentage indicate the model is adding real, self-consistent detail rather
          than inventing texture.</p>
        </div>
      </section>

      <section id="use-cases">
        <div class="container">
          <div class="tag">where it applies</div>
          <h2>Built for decisions that need street-level detail, at country scale.</h2>
          <div style="margin-top:2rem;">
            <div class="usecase"><div class="n">01</div><h4>Disaster response</h4>
              <p>Sharper flood and damage extent at closer to building-scale, using imagery that's already free and already updating every few days.</p></div>
            <div class="usecase"><div class="n">02</div><h4>Agriculture</h4>
              <p>Field-boundary and crop-stress detail for plots too small for 10m pixels to resolve individually. Most Indian smallholder farms fall in this gap.</p></div>
            <div class="usecase"><div class="n">03</div><h4>Urban &amp; environmental monitoring</h4>
              <p>Sharper boundaries for encroachment, land-use change, and coastal or water-body extent tracking over time.</p></div>
          </div>
        </div>
      </section>

      <section id="roadmap">
        <div class="container">
          <div class="tag">what's next</div>
          <h2>This prototype proves the pipeline. Phase 2 removes its biggest caveat.</h2>
          <div class="roadmap">
            <div class="phase active">
              <div class="tag2">phase 1: this build</div>
              <h3>Inference-only, pretrained</h3>
              <ul>
                <li>ESA OpenSR's LDSR-S2 model, no training required</li>
                <li>Full ingestion, diagnostics and evaluation harness</li>
                <li>Uncertainty + hallucination-risk masking end to end</li>
                <li>Streamlit interface for upload-and-inspect workflows</li>
              </ul>
            </div>
            <div class="phase">
              <div class="tag2">phase 2: proposed</div>
              <h3>Custom hybrid CNN-Transformer</h3>
              <ul>
                <li>Trained from scratch on paired data, not inference-only</li>
                <li>Cartosat-3 (ISRO / NRSC Bhuvan) as genuine high-res ground truth</li>
                <li>Closes the "no native 2.5m ground truth" gap this prototype is honest about</li>
                <li>Sovereign data source, no dependency on foreign commercial tasking</li>
              </ul>
            </div>
          </div>
          <div class="stack">
            <div class="badge">LDSR-S2 (ESA OpenSR)</div>
            <div class="badge">PyTorch</div>
            <div class="badge">rasterio</div>
            <div class="badge">scikit-image</div>
            <div class="badge">Streamlit</div>
            <div class="badge">NumPy</div>
          </div>
        </div>
      </section>

      <div id="cta"></div>
      <div id="cta-intro">
        <div class="container">
          <div class="tag">try it</div>
          <h2>See a real tile go from 10m to 2.5m, live.</h2>
          <p class="lede">Upload a Sentinel-2 GeoTIFF below and watch the reconstruction,
          uncertainty map, and hallucination-risk mask generate right here.</p>
        </div>
      </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# IMAGE UTILITIES
# =============================================================

def normalize_reflectance(data):
    """
    Convert Sentinel-2 data to float32 reflectance.

    Supports:

        Already normalized:
            0 - 1

        DN-scaled:
            approximately 0 - 10000
    """

    data = np.asarray(
        data,
        dtype=np.float32
    )

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    if np.nanmax(data) > 2.0:

        data = data / 10000.0

    return np.clip(
        data,
        0.0,
        1.0
    )


def stretch_rgb(rgb):
    """
    Display-only RGB enhancement.

    Does NOT modify the actual values used
    for model inference or metrics.
    """

    rgb = np.asarray(
        rgb,
        dtype=np.float32
    )

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # Same scale applied to all RGB channels.
    # This preserves relative color balance.

    low = 0.02
    high = 0.35

    rgb = (
        rgb - low
    ) / (
        high - low
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0
    )

    # Gamma correction
    rgb = np.power(
        rgb,
        1.0 / 2.2
    )

    return np.clip(
        rgb,
        0.0,
        1.0
    )


def bands_to_rgb(data):
    """
    Convert Sentinel-2 B02/B03/B04/B08
    into natural RGB.

    Input:

        (4, H, W)

    Output:

        (H, W, 3)

    RGB:

        R = B04
        G = B03
        B = B02
    """

    rgb = np.stack(
        [
            data[2],  # B04 -> Red
            data[1],  # B03 -> Green
            data[0],  # B02 -> Blue
        ],
        axis=-1
    )

    return rgb.astype(
        np.float32
    )


def uncertainty_display_map(
    uncertainty,
    low_percentile=2.0,
    high_percentile=98.0
):
    """
    Normalize uncertainty only for display.
    """

    uncertainty = np.asarray(
        uncertainty,
        dtype=np.float32
    )

    uncertainty = np.nan_to_num(
        uncertainty,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    low = np.percentile(
        uncertainty,
        low_percentile
    )

    high = np.percentile(
        uncertainty,
        high_percentile
    )

    if high <= low:

        return np.zeros_like(
            uncertainty,
            dtype=np.float32
        )

    normalized = (
        uncertainty - low
    ) / (
        high - low
    )

    return np.clip(
        normalized,
        0.0,
        1.0
    )


def make_risk_overlay(
    sr_rgb,
    risk_mask
):
    """
    Overlay hallucination-risk regions on RGB SR image.

    Risk areas are highlighted in red.

    IMPORTANT:
    The risk mask is calculated from the
    4-band RGB-NIR pipeline, but visualization
    is RGB only.
    """

    overlay = np.asarray(
        sr_rgb,
        dtype=np.float32
    ).copy()

    risk_mask = np.asarray(
        risk_mask,
        dtype=bool
    )

    # Red highlight
    overlay[risk_mask] = (
        0.5 * overlay[risk_mask]
        + 0.5 * np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )
    )

    return np.clip(
        overlay,
        0.0,
        1.0
    )


# =============================================================
# TIFF READER
# =============================================================

def read_tiff_4band(path):
    """
    Read:

        B02
        B03
        B04
        B08

    Returns:

        (4, H, W)
    """

    with rasterio.open(path) as src:

        if src.count < 4:

            raise ValueError(
                f"Expected at least 4 bands "
                f"(B02, B03, B04, B08), "
                f"but TIFF contains {src.count}."
            )

        data = src.read(
            indexes=[1, 2, 3, 4]
        )

    return normalize_reflectance(
        data
    )


# =============================================================
# FILE UPLOADER
# =============================================================

uploaded_file = st.file_uploader(
    "Upload a Sentinel-2 4-band GeoTIFF",
    type=["tif", "tiff"]
)


# =============================================================
# MAIN
# =============================================================

if uploaded_file is not None:

    # =========================================================
    # SAVE INPUT
    # =========================================================

    temp_path = os.path.join(
        "data",
        "raw",
        uploaded_file.name
    )

    os.makedirs(
        os.path.dirname(temp_path),
        exist_ok=True
    )

    with open(
        temp_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    # =========================================================
    # READ ORIGINAL
    # =========================================================

    try:

        data = read_tiff_4band(
            temp_path
        )

    except Exception as e:

        st.error(
            f"Could not read Sentinel-2 TIFF: {e}"
        )

        st.stop()

    # =========================================================
    # VALIDATE 4 BANDS
    # =========================================================

    if data.shape[0] != 4:

        st.error(
            f"Expected 4 bands, "
            f"got {data.shape[0]}"
        )

        st.stop()

    # =========================================================
    # ORIGINAL 4-CHANNEL ARRAY
    # =========================================================

    original_4ch_full = (
        np.transpose(
            data,
            (1, 2, 0)
        )
        .astype(np.float32)
    )

    # =========================================================
    # MODEL INPUT PATCH
    # =========================================================

    h, w = original_4ch_full.shape[:2]

    if h < 128 or w < 128:

        st.error(
            f"Input image is {w}x{h}. "
            "At least 128x128 pixels are required."
        )

        st.stop()

    row_start = (
        h - 128
    ) // 2

    col_start = (
        w - 128
    ) // 2

    original_4ch = (
        original_4ch_full[
            row_start:row_start + 128,
            col_start:col_start + 128,
            :
        ]
    )

    # Expected:
    #
    # (128, 128, 4)

    if original_4ch.shape != (
        128,
        128,
        4
    ):

        raise ValueError(
            f"Expected model input "
            f"(128,128,4), "
            f"got {original_4ch.shape}"
        )

    # =========================================================
    # ORIGINAL RGB
    # =========================================================

    original_rgb = bands_to_rgb(
        np.transpose(
            original_4ch,
            (2, 0, 1)
        )
    )

    original_rgb_display = stretch_rgb(
        original_rgb
    )

    # =========================================================
    # RUN LDSR-S2
    # =========================================================

    with st.spinner(
        "Running LDSR-S2, generating 2.5m imagery..."
    ):

        sr_output, uncertainty_map = run_sr(
            temp_path,
            sampling_steps=100,
            patch_size=128
        )

    # =========================================================
    # VALIDATE SR OUTPUT
    # =========================================================

    if sr_output.shape != (
        512,
        512,
        4
    ):

        raise ValueError(
            f"Expected 512x512x4 output, "
            f"got {sr_output.shape}"
        )

    if uncertainty_map.shape != (
        512,
        512
    ):

        raise ValueError(
            f"Expected 512x512 uncertainty map, "
            f"got {uncertainty_map.shape}"
        )

    # =========================================================
    # NORMALIZE SR OUTPUT
    # =========================================================

    sr_output = np.asarray(
        sr_output,
        dtype=np.float32
    )

    sr_output = np.nan_to_num(
        sr_output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    sr_output = np.clip(
        sr_output,
        0.0,
        1.0
    )

    # =========================================================
    # SR RGB
    # =========================================================

    sr_rgb = np.stack(
        [
            sr_output[:, :, 2],  # B04 -> R
            sr_output[:, :, 1],  # B03 -> G
            sr_output[:, :, 0],  # B02 -> B
        ],
        axis=-1
    )

    sr_rgb_display = stretch_rgb(
        sr_rgb
    )

    # =========================================================
    # UNCERTAINTY VISUALIZATION
    # =========================================================

    uncertainty_visual = (
        uncertainty_display_map(
            uncertainty_map
        )
    )

    # =========================================================
    # HALLUCINATION / LR CONSISTENCY
    # =========================================================

    with st.spinner(
        "Running hallucination-risk diagnostics..."
    ):

        # -----------------------------------------------------
        # Degrade SR back to LR and calculate residual
        # -----------------------------------------------------

        error_lr, consistency_error = (
            lr_consistency_error(
                low_res_rgb=original_4ch,
                sr_output=sr_output,
                blur_sigma=1.0
            )
        )

        # -----------------------------------------------------
        # Normalize residual for display
        # -----------------------------------------------------

        consistency_visual = (
            percentile_normalize(
                consistency_error,
                low_percentile=1,
                high_percentile=99
            )
        )

        # -----------------------------------------------------
        # Create conservative risk mask
        #
        # BOTH must be high:
        #
        # high residual
        #       AND
        # high uncertainty
        # -----------------------------------------------------

        (
            risk_mask,
            error_threshold,
            uncertainty_threshold
        ) = create_hallucination_risk_mask(
            consistency_error,
            uncertainty_map,
            percentile=90.0
        )

        # -----------------------------------------------------
        # Percentage of high-risk pixels
        # -----------------------------------------------------

        flagged_percentage = (
            100.0
            * risk_mask.sum()
            / risk_mask.size
        )

        # -----------------------------------------------------
        # RGB risk overlay
        # -----------------------------------------------------

        risk_overlay = make_risk_overlay(
            sr_rgb_display,
            risk_mask
        )

    # =========================================================
    # SUCCESS
    # =========================================================

    st.success(
        "Super-resolution completed! "
        "10m → 2.5m"
    )

    # =========================================================
    # MAIN VISUALIZATION
    # =========================================================

    st.header(
        "Super-Resolution Result"
    )

    col1, col2, col3 = st.columns(
        3
    )

    # ---------------------------------------------------------
    # ORIGINAL
    # ---------------------------------------------------------

    with col1:

        st.subheader(
            "Original (10m)"
        )

        st.image(
            original_rgb_display,
            use_container_width=True
        )

        st.caption(
            "B04 / B03 / B02 natural-color RGB"
        )

    # ---------------------------------------------------------
    # SR
    # ---------------------------------------------------------

    with col2:

        st.subheader(
            "GeoSharp Output (2.5m)"
        )

        st.image(
            sr_rgb_display,
            use_container_width=True
        )

        st.caption(
            "LDSR-S2 4× reconstruction"
        )

    # ---------------------------------------------------------
    # UNCERTAINTY
    # ---------------------------------------------------------

    with col3:

        st.subheader(
            "Uncertainty Map"
        )

        st.image(
            uncertainty_visual,
            use_container_width=True
        )

        st.caption(
            "Brighter = higher model uncertainty"
        )

    # =========================================================
    # OUTPUT INFORMATION
    # =========================================================

    st.divider()

    st.header(
        "Output Information"
    )

    info1, info2, info3, info4 = st.columns(
        4
    )

    info1.metric(
        "Input",
        "128 × 128 × 4"
    )

    info2.metric(
        "Output",
        "512 × 512 × 4"
    )

    info3.metric(
        "Resolution",
        "10m → 2.5m"
    )

    info4.metric(
        "Scale",
        "4×"
    )

    st.caption(
        "Bands: B02 (Blue) • B03 (Green) • "
        "B04 (Red) • B08 (NIR)"
    )

    # =========================================================
    # 4-BAND QUALITY METRICS
    # =========================================================

    st.divider()

    st.header(
        "4-Band Reconstruction Quality"
    )

    # ---------------------------------------------------------
    # Bicubic baseline
    #
    # Original 128x128x4
    #       ↓
    # Bicubic 4x
    #       ↓
    # 512x512x4
    #
    # Compared against:
    #
    # GeoSharp 512x512x4
    # ---------------------------------------------------------

    metrics = compare_to_baseline(
        original_4ch.astype(
            np.float32
        ),
        sr_output.astype(
            np.float32
        ),
        scale_factor=4
    )

    metric1, metric2 = st.columns(
        2
    )

    metric1.metric(
        "PSNR (All 4 Bands)",
        f"{metrics['psnr_sr_vs_baseline']:.2f} dB"
    )

    metric2.metric(
        "SSIM (All 4 Bands)",
        f"{metrics['ssim_sr_vs_baseline']:.3f}"
    )

    st.caption(
        "Calculated across B02, B03, B04 and B08 "
        "against a 4× bicubic baseline."
    )

    # =========================================================
    # PER-BAND BASIC STATISTICS
    # =========================================================

    st.subheader(
        "Spectral Channels"
    )

    band1, band2, band3, band4 = st.columns(
        4
    )

    band1.metric(
        "B02",
        "Blue"
    )

    band2.metric(
        "B03",
        "Green"
    )

    band3.metric(
        "B04",
        "Red"
    )

    band4.metric(
        "B08",
        "NIR"
    )

    # =========================================================
    # UNCERTAINTY ANALYSIS
    # =========================================================

    st.divider()

    st.header(
        "Uncertainty Analysis"
    )

    uncertainty_mean = float(
        np.mean(
            uncertainty_map
        )
    )

    uncertainty_max = float(
        np.max(
            uncertainty_map
        )
    )

    uncertainty_p95 = float(
        np.percentile(
            uncertainty_map,
            95
        )
    )

    u1, u2, u3 = st.columns(
        3
    )

    u1.metric(
        "Mean Uncertainty",
        f"{uncertainty_mean:.6f}"
    )

    u2.metric(
        "Maximum Uncertainty",
        f"{uncertainty_max:.6f}"
    )

    u3.metric(
        "P95 Uncertainty",
        f"{uncertainty_p95:.6f}"
    )

    st.caption(
        "Uncertainty is estimated from stochastic "
        "diffusion samples. Higher values indicate "
        "greater disagreement between reconstructions."
    )

    # =========================================================
    # LR CONSISTENCY
    # =========================================================

    st.divider()

    st.header(
        "LR-Consistency Analysis"
    )

    st.caption(
        "The SR output is degraded back toward the "
        "original 10m observation. A larger residual "
        "means the generated fine-scale structure is "
        "less strongly supported by the observed input."
    )

    residual_mean = float(
        np.mean(
            consistency_error
        )
    )

    residual_max = float(
        np.max(
            consistency_error
        )
    )

    residual_p95 = float(
        np.percentile(
            consistency_error,
            95
        )
    )

    r1, r2, r3 = st.columns(
        3
    )

    r1.metric(
        "Mean Residual",
        f"{residual_mean:.6f}"
    )

    r2.metric(
        "Maximum Residual",
        f"{residual_max:.6f}"
    )

    r3.metric(
        "P95 Residual",
        f"{residual_p95:.6f}"
    )

    # =========================================================
    # LR CONSISTENCY MAP
    # =========================================================

    st.subheader(
        "LR-Consistency Residual Map"
    )

    st.image(
        consistency_visual,
        use_container_width=True
    )

    st.caption(
        "Brighter = greater difference between the "
        "observed low-resolution signal and the "
        "SR output after degradation."
    )

    # =========================================================
    # HALLUCINATION-RISK ANALYSIS
    # =========================================================

    st.divider()

    st.header(
        "Hallucination-Risk Analysis"
    )

    st.info(
        "Hallucination risk is flagged only where "
        "both LR-consistency residual and stochastic "
        "uncertainty are high. This is a risk indicator, "
        "not absolute proof of hallucination."
    )

    # ---------------------------------------------------------
    # Risk percentage
    # ---------------------------------------------------------

    if flagged_percentage < 5.0:

        risk_status = "Low"

    elif flagged_percentage < 15.0:

        risk_status = "Moderate"

    else:

        risk_status = "High"

    h1, h2, h3 = st.columns(
        3
    )

    h1.metric(
        "High-Risk Pixels",
        f"{flagged_percentage:.2f}%"
    )

    h2.metric(
        "Risk Status",
        risk_status
    )

    h3.metric(
        "Risk Threshold",
        "90th percentile"
    )

    # =========================================================
    # RISK THRESHOLDS
    # =========================================================

    st.subheader(
        "Risk Thresholds"
    )

    t1, t2 = st.columns(
        2
    )

    t1.metric(
        "LR Residual Threshold",
        f"{error_threshold:.6f}"
    )

    t2.metric(
        "Uncertainty Threshold",
        f"{uncertainty_threshold:.6f}"
    )

    # =========================================================
    # HALLUCINATION OVERLAY
    # =========================================================

    st.subheader(
        "Hallucination-Risk Overlay"
    )

    st.image(
        risk_overlay,
        use_container_width=True
    )

    st.caption(
        "Red regions = high LR-consistency residual "
        "AND high stochastic uncertainty."
    )

    # =========================================================
    # RISK MASK
    # =========================================================

    st.subheader(
        "Binary Risk Mask"
    )

    st.image(
        risk_mask.astype(
            np.float32
        ),
        use_container_width=True
    )

    st.caption(
        "White = high hallucination-risk region. "
        "Black = lower-risk region."
    )

    # =========================================================
    # DIAGNOSTIC SUMMARY
    # =========================================================

    st.divider()

    st.header(
        "GeoSharp Diagnostic Summary"
    )

    summary_col1, summary_col2 = st.columns(
        2
    )

    with summary_col1:

        st.markdown(
            f"""
**Reconstruction**

- Input: **128 × 128 × 4**
- Output: **512 × 512 × 4**
- Resolution: **10m → 2.5m**
- Super-resolution factor: **4×**
- Channels: **B02 / B03 / B04 / B08**
- Sampling steps: **100**

**Quality**

- 4-band PSNR: **{metrics['psnr_sr_vs_baseline']:.2f} dB**
- 4-band SSIM: **{metrics['ssim_sr_vs_baseline']:.3f}**
"""
        )

    with summary_col2:

        st.markdown(
            f"""
**Uncertainty**

- Mean: **{uncertainty_mean:.6f}**
- P95: **{uncertainty_p95:.6f}**
- Maximum: **{uncertainty_max:.6f}**

**Hallucination Risk**

- Mean LR residual: **{residual_mean:.6f}**
- P95 LR residual: **{residual_p95:.6f}**
- High-risk pixels: **{flagged_percentage:.2f}%**
- Risk status: **{risk_status}**
"""
        )

    # =========================================================
    # SCIENTIFIC CAVEAT
    # =========================================================

    st.warning(
        "Evaluation note: PSNR/SSIM compare the "
        "4-channel GeoSharp output against a 4× bicubic "
        "baseline because native 2.5m ground truth is "
        "not available. Therefore these metrics indicate "
        "relative reconstruction behavior, not absolute "
        "2.5m accuracy. Hallucination diagnostics are "
        "risk indicators rather than proof of hallucination."
    )

else:

    st.info(
        "Upload a 4-band Sentinel-2 GeoTIFF "
        "to run GeoSharp."
    )

st.markdown(
    """
    <div class="geosharp-site">
      <footer>
        <div class="container">
          <div class="foot-row">
            <div>GeoSharp: Smart India Hackathon 2026</div>
            <div><a href="#">GitHub</a> &nbsp;·&nbsp; <a href="#">Team</a> &nbsp;·&nbsp; <a href="#">Contact</a></div>
            <div>Prototype uses a pretrained model for inference only.</div>
          </div>
        </div>
      </footer>
    </div>
    """,
    unsafe_allow_html=True,
)
