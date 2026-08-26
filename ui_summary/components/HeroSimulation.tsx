/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, FastForward, Activity, Layers, Sparkles, Zap, Minimize2, CheckCircle2 } from 'lucide-react';
import { Reveal } from './Reveal';

interface Particle2D {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  stagnation: number;
  bestX: number;
  bestY: number;
  bestVal: number;
  isOBL?: boolean;
}

export const HeroSimulation: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(true);
  const [mode, setMode] = useState<'SDAO' | 'SGD'>('SDAO');
  const [stepCount, setStepCount] = useState<number>(0);
  const [globalBestVal, setGlobalBestVal] = useState<number>(999.0);
  const [oblEvents, setOblEvents] = useState<number>(0);
  const [trappedCount, setTrappedCount] = useState<number>(0);
  const [diffusionCap, setDiffusionCap] = useState<string>('3.95');

  const particlesRef = useRef<Particle2D[]>([]);
  const globalBestRef = useRef<{ x: number; y: number; val: number }>({ x: 0, y: 0, val: 999.0 });

  // Double well potential function: left well is local optimum, right well is global optimum
  const evaluateWell = (x: number, y: number): number => {
    // x range approx [-4, 4], y range [-3, 3]
    // Local minimum at x = -2, y = 0 (val approx 15)
    // Global minimum at x = 2.2, y = 0 (val approx 0.05)
    const well1 = 20 * Math.exp(-0.8 * (Math.pow(x + 2.0, 2) + Math.pow(y, 2)));
    const well2 = 35 * Math.exp(-0.9 * (Math.pow(x - 2.2, 2) + Math.pow(y, 2)));
    const barrier = 12 * Math.exp(-1.5 * Math.pow(x, 2));
    const base = 0.5 * (Math.pow(x, 2) * 0.1 + Math.pow(y, 2) * 0.4);
    
    // Invert wells so they are minima
    return Math.max(0.01, 35 - well1 - well2 + barrier + base);
  };

  const initParticles = () => {
    const newParticles: Particle2D[] = [];
    let bestVal = 999;
    let bestX = 0;
    let bestY = 0;

    // Start all particles trapped in or near the left well (local minimum)
    for (let i = 0; i < 45; i++) {
      const x = -2.5 + Math.random() * 1.5;
      const y = -1.2 + Math.random() * 2.4;
      const val = evaluateWell(x, y);

      newParticles.push({
        id: i,
        x,
        y,
        vx: (Math.random() - 0.5) * 0.1,
        vy: (Math.random() - 0.5) * 0.1,
        stagnation: 0,
        bestX: x,
        bestY: y,
        bestVal: val
      });

      if (val < bestVal) {
        bestVal = val;
        bestX = x;
        bestY = y;
      }
    }

    particlesRef.current = newParticles;
    globalBestRef.current = { x: bestX, y: bestY, val: bestVal };
    setGlobalBestVal(bestVal);
    setStepCount(0);
    setOblEvents(0);
    setTrappedCount(45);
  };

  useEffect(() => {
    initParticles();
  }, [mode]);

  const stepSimulation = () => {
    const particles = particlesRef.current;
    if (particles.length === 0) return;

    let bestVal = globalBestRef.current.val;
    let bestX = globalBestRef.current.x;
    let bestY = globalBestRef.current.y;
    let oblTriggered = 0;
    let leftWellCount = 0;

    // Calculate centroid of particles for density repulsion
    let sumX = 0;
    let sumY = 0;
    particles.forEach(p => {
      sumX += p.x;
      sumY += p.y;
      if (p.x < 0) leftWellCount++;
    });
    const centerX = sumX / particles.length;
    const centerY = sumY / particles.length;

    const alpha = 0.08 * Math.exp(-0.005 * stepCount);
    const gamma = 0.65 * (1 + (leftWellCount / particles.length) * 0.5);
    const dVal = Math.max(0.2, 3.95 * Math.exp(-0.01 * stepCount) * (1 + 0.5 * (leftWellCount / 45)));
    setDiffusionCap(dVal.toFixed(2));

    particles.forEach(p => {
      p.isOBL = false;

      if (mode === 'SGD') {
        // Classical gradient descent: numerical gradient
        const eps = 0.05;
        const fx1 = evaluateWell(p.x + eps, p.y);
        const fx2 = evaluateWell(p.x - eps, p.y);
        const fy1 = evaluateWell(p.x, p.y + eps);
        const fy2 = evaluateWell(p.x, p.y - eps);
        const gx = (fx1 - fx2) / (2 * eps);
        const gy = (fy1 - fy2) / (2 * eps);

        // Move purely down gradient
        p.x -= 0.12 * gx;
        p.y -= 0.12 * gy;
      } else {
        // SDAO Engine: Fickian density repulsion + attraction + OBL
        // 1. Density Repulsion vector D_FL away from local cluster center
        const diffX = p.x - centerX;
        const diffY = p.y - centerY;
        const dist = Math.hypot(diffX, diffY) || 0.01;
        const repX = (diffX / dist) * (dVal * 0.15);
        const repY = (diffY / dist) * (dVal * 0.15);

        // 2. Attraction toward global best and personal best
        const attrGX = 0.3 * (bestX - p.x);
        const attrGY = 0.3 * (bestY - p.y);
        const attrPX = gamma * 0.2 * (p.bestX - p.x);
        const attrPY = gamma * 0.2 * (p.bestY - p.y);

        // 3. Stochastic Wiener noise
        const noiseX = (Math.random() - 0.5) * Math.sqrt(dVal) * 0.35;
        const noiseY = (Math.random() - 0.5) * Math.sqrt(dVal) * 0.35;

        p.x += repX + attrGX + attrPX + noiseX;
        p.y += repY + attrGY + attrPY + noiseY;

        // 4. Opposition-Based Learning (OBL) jump if stagnant in local trap
        if (p.x < 0 && p.stagnation > 8 && Math.random() < 0.28) {
          p.x = -p.x + (Math.random() - 0.5) * 0.8; // Catapult across origin to right well
          p.y = -p.y * 0.5;
          p.stagnation = 0;
          p.isOBL = true;
          oblTriggered++;
        }
      }

      // Keep in canvas view bounds [-4.2, 4.2] x [-2.8, 2.8]
      p.x = Math.max(-4.0, Math.min(4.0, p.x));
      p.y = Math.max(-2.6, Math.min(2.6, p.y));

      const val = evaluateWell(p.x, p.y);
      if (val < p.bestVal) {
        p.bestVal = val;
        p.bestX = p.x;
        p.bestY = p.y;
        p.stagnation = 0;
      } else {
        p.stagnation++;
      }

      if (val < bestVal) {
        bestVal = val;
        bestX = p.x;
        bestY = p.y;
      }
    });

    globalBestRef.current = { x: bestX, y: bestY, val: bestVal };
    setGlobalBestVal(bestVal);
    setStepCount(s => s + 1);
    setOblEvents(e => e + oblTriggered);
    setTrappedCount(leftWellCount);
  };

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isRunning && stepCount < 250) {
      timer = setTimeout(() => {
        stepSimulation();
      }, 60);
    } else if (stepCount >= 250) {
      setIsRunning(false);
    }
    return () => clearTimeout(timer);
  }, [isRunning, stepCount]);

  // Render Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Draw Double Well Heatmap Background
    const cols = 50;
    const rows = 35;
    const cellW = width / cols;
    const cellH = height / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const wx = -4.0 + (c / cols) * 8.0;
        const wy = -2.6 + (r / rows) * 5.2;
        const val = evaluateWell(wx, wy);

        // Normalize color map
        const norm = Math.min(1, Math.max(0, val / 30));

        // Dark terrain: deep ink base lifting toward warm ridge
        let red = Math.floor(14 + norm * 40);
        let green = Math.floor(14 + norm * 28);
        let blue = Math.floor(16 + norm * 12);

        if (wx < -0.8 && norm < 0.6) {
          // Left well (Local optimum trap - amber warning zone)
          red = 70 + Math.floor(norm * 60); green = 48 - Math.floor(norm * 20); blue = 24;
        } else if (wx > 0.8 && norm < 0.4) {
          // Right well (Global optimum basin - emerald target zone)
          red = 18; green = 70 - Math.floor(norm * 30); blue = 56;
        }

        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
      }
    }

    // Draw contours and labels for the two wells
    // Left well center approx x = -2, y = 0
    const leftX = ((-2.0 + 4.0) / 8.0) * width;
    const leftY = ((0 + 2.6) / 5.2) * height;
    ctx.strokeStyle = 'rgba(217, 153, 80, 0.55)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.arc(leftX, leftY, 60, 0, Math.PI * 2);
    ctx.stroke();

    // Right well center approx x = 2.2, y = 0
    const rightX = ((2.2 + 4.0) / 8.0) * width;
    const rightY = ((0 + 2.6) / 5.2) * height;
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.6)';
    ctx.beginPath();
    ctx.arc(rightX, rightY, 70, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Particles
    particlesRef.current.forEach(p => {
      const px = ((p.x + 4.0) / 8.0) * width;
      const py = ((p.y + 2.6) / 5.2) * height;

      // OBL jump flash effect
      if (p.isOBL) {
        ctx.strokeStyle = '#d8b978';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = mode === 'SDAO' ? (p.x > 0 ? '#34d399' : '#d8b978') : '#f87171';
      ctx.fill();
      ctx.strokeStyle = 'rgba(246,244,238,0.9)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // Draw Global Best Star
    if (globalBestRef.current.val < 900) {
      const gbx = ((globalBestRef.current.x + 4.0) / 8.0) * width;
      const gby = ((globalBestRef.current.y + 2.6) / 5.2) * height;
      ctx.beginPath();
      ctx.arc(gbx, gby, 9, 0, Math.PI * 2);
      ctx.fillStyle = '#34d399';
      ctx.fill();
      ctx.strokeStyle = 'rgba(246,244,238,0.95)';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

  }, [stepCount, mode]);

  return (
    <div className="mx-auto w-full max-w-6xl">
      <Reveal className="mb-8 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <span className="eyebrow">Core mechanism</span>
          <h2 className="display mt-3 text-3xl text-bone md:text-4xl">
            Escaping deceptive optima: SDAO vs. gradient descent
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-white/55">
            Forty-five candidate solutions start trapped in a deceptive local minimum (left well). Classical gradient
            descent stays stuck forever; SDAO&rsquo;s density repulsion and opposition-based learning catapult the swarm
            into the true global optimum (right well).
          </p>
        </div>

        <div className="flex w-full shrink-0 gap-1.5 rounded-full border border-white/10 bg-white/[0.03] p-1.5 sm:w-auto">
          <button
            onClick={() => { setMode('SDAO'); initParticles(); }}
            className={`flex-1 rounded-full px-4 py-2.5 text-[11px] font-bold uppercase tracking-wider transition-all sm:flex-none ${
              mode === 'SDAO' ? 'bg-gold text-ink' : 'text-white/55 hover:text-bone'
            }`}
          >
            SDAO (Fickian)
          </button>
          <button
            onClick={() => { setMode('SGD'); initParticles(); }}
            className={`flex-1 rounded-full px-4 py-2.5 text-[11px] font-bold uppercase tracking-wider transition-all sm:flex-none ${
              mode === 'SGD' ? 'bg-red-500 text-white' : 'text-white/55 hover:text-bone'
            }`}
          >
            Classical SGD
          </button>
        </div>
      </Reveal>

      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        <div className="flex flex-col overflow-hidden rounded-2xl border border-white/10 bg-white/[0.02] lg:col-span-7">
          <div className="relative aspect-[4/3] min-h-[260px] w-full sm:aspect-[16/11] sm:min-h-[340px]">
            <canvas ref={canvasRef} width={640} height={440} className="block h-full w-full" />
            <div className="pointer-events-none absolute left-2 right-2 top-2 flex items-center justify-between gap-2 sm:left-3.5 sm:top-3.5 sm:right-3.5">
              <div className="flex items-center gap-1.5 rounded-lg border border-amber-400/30 bg-amber-400/10 px-2 py-1 text-[10px] font-bold text-amber-200 backdrop-blur-md sm:px-3 sm:py-1.5 sm:text-xs">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-400" />
                Local trap (f &asymp; 15.0)
              </div>
              <div className="flex items-center gap-1.5 rounded-lg border border-emerald-400/30 bg-emerald-400/10 px-2 py-1 text-[10px] font-bold text-emerald-200 backdrop-blur-md sm:px-3 sm:py-1.5 sm:text-xs">
                <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />
                Global basin (f &asymp; 0.05)
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 px-3.5 py-3 text-xs sm:px-5 sm:py-3.5">
            <div className="flex items-center gap-3 font-mono text-[11px] text-white/50 sm:gap-5 sm:text-xs">
              <span>Step: <strong className="font-bold text-bone">{stepCount}/250</strong></span>
              <span>Best f(x): <strong className="font-bold text-emerald-300">{globalBestVal.toFixed(3)}</strong></span>
            </div>
            <div className="ml-auto flex items-center gap-2 sm:ml-0">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className="flex items-center gap-1.5 rounded-lg bg-bone px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider text-ink transition-transform hover:scale-[1.03] sm:text-xs"
              >
                {isRunning ? <Pause size={13} /> : <Play size={13} />}
                {isRunning ? 'Pause' : 'Resume'}
              </button>
              <button
                onClick={initParticles}
                title="Reset Simulation"
                className="flex items-center gap-1 rounded-lg border border-white/15 px-2.5 py-1.5 text-[11px] font-bold text-white/70 transition-colors hover:border-gold hover:text-gold sm:text-xs"
              >
                <RotateCcw size={13} /> Reset
              </button>
            </div>
          </div>
        </div>

        <div className="space-y-6 lg:col-span-5">
          <div className="space-y-4 rounded-2xl border border-white/10 bg-white/[0.02] p-6">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <span className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-gold">Swarm telemetry</span>
              <span className="font-mono text-xs text-white/40">N = 45</span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3.5">
                <div className="font-mono text-[11px] uppercase text-white/45">Trapped</div>
                <div className={`mt-1 display text-2xl font-bold ${trappedCount > 20 ? 'text-amber-300' : 'text-emerald-300'}`}>
                  {trappedCount} <span className="font-sans text-xs font-normal text-white/40">/ 45</span>
                </div>
              </div>
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3.5">
                <div className="font-mono text-[11px] uppercase text-white/45">Escaped</div>
                <div className={`mt-1 display text-2xl font-bold ${45 - trappedCount > 20 ? 'text-emerald-300' : 'text-white/50'}`}>
                  {45 - trappedCount} <span className="font-sans text-xs font-normal text-white/40">/ 45</span>
                </div>
              </div>
            </div>
            {mode === 'SDAO' && (
              <div className="grid grid-cols-2 gap-4 pt-1">
                <div className="flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] p-3">
                  <span className="text-xs text-white/55">Diffusion D(k)</span>
                  <span className="font-mono font-bold text-gold">{diffusionCap}</span>
                </div>
                <div className="flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] p-3">
                  <span className="text-xs text-white/55">OBL reflections</span>
                  <span className="font-mono font-bold text-emerald-300">{oblEvents}</span>
                </div>
              </div>
            )}
          </div>

          <div className="space-y-3 rounded-2xl border border-white/10 bg-white/[0.02] p-6">
            <h3 className="display flex items-center gap-2 text-xl text-bone">
              {mode === 'SDAO' ? (
                <><CheckCircle2 size={20} className="text-emerald-400" /><span>Why SDAO succeeds</span></>
              ) : (
                <><Activity size={20} className="text-red-400" /><span>Why gradient descent fails</span></>
              )}
            </h3>
            {mode === 'SDAO' ? (
              <div className="space-y-2.5 text-xs leading-relaxed text-white/55">
                <p><strong className="text-white/80">Fick&rsquo;s 2nd law active:</strong> Overcrowding in the left trap creates a repulsive force vector D_FL(x&#7522;) away from the cluster centroid, forcing particles over the barrier into the right well.</p>
                <p><strong className="text-white/80">Opposition-based learning (OBL):</strong> Particles that remain stagnant trigger an opposition jump across the origin (gold rings), landing directly in the global attraction basin.</p>
              </div>
            ) : (
              <div className="space-y-2.5 text-xs leading-relaxed text-white/55">
                <p><strong className="text-white/80">Zero gradient stagnation:</strong> At the bottom of the left well the local gradient is zero (&nabla;f &asymp; 0). Gradient descent relies on local slope, making escape impossible without external energy.</p>
                <p><strong className="text-white/80">Premature convergence:</strong> All 45 particles collapse permanently into the false minimum (f &asymp; 15.0), missing the true global solution (f &asymp; 0.05).</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
