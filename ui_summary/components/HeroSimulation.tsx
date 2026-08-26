/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, FastForward, Activity, Layers, Sparkles, Zap, Minimize2, CheckCircle2 } from 'lucide-react';

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

        let red = Math.floor(248 - norm * 60);
        let green = Math.floor(246 - norm * 80);
        let blue = Math.floor(240 - norm * 110);

        if (wx < -0.8 && norm < 0.6) {
          // Left well (Local optimum trap - yellowish amber warning zone)
          red = 254; green = 243 - Math.floor(norm * 100); blue = 199;
        } else if (wx > 0.8 && norm < 0.4) {
          // Right well (Global optimum basin - emerald target zone)
          red = 209; green = 250 - Math.floor(norm * 120); blue = 229;
        }

        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
      }
    }

    // Draw contours and labels for the two wells
    // Left well center approx x = -2, y = 0
    const leftX = ((-2.0 + 4.0) / 8.0) * width;
    const leftY = ((0 + 2.6) / 5.2) * height;
    ctx.strokeStyle = 'rgba(217, 119, 6, 0.4)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.arc(leftX, leftY, 60, 0, Math.PI * 2);
    ctx.stroke();

    // Right well center approx x = 2.2, y = 0
    const rightX = ((2.2 + 4.0) / 8.0) * width;
    const rightY = ((0 + 2.6) / 5.2) * height;
    ctx.strokeStyle = 'rgba(5, 150, 105, 0.5)';
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
        ctx.strokeStyle = '#C5A059';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = mode === 'SDAO' ? (p.x > 0 ? '#059669' : '#C5A059') : '#DC2626';
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // Draw Global Best Star
    if (globalBestRef.current.val < 900) {
      const gbx = ((globalBestRef.current.x + 4.0) / 8.0) * width;
      const gby = ((globalBestRef.current.y + 2.6) / 5.2) * height;
      ctx.beginPath();
      ctx.arc(gbx, gby, 9, 0, Math.PI * 2);
      ctx.fillStyle = '#10B981';
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

  }, [stepCount, mode]);

  return (
    <div className="w-full max-w-6xl mx-auto my-6 sm:my-12 p-4 sm:p-6 md:p-10 bg-white rounded-2xl sm:rounded-3xl border border-stone-200 shadow-xl">
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 sm:gap-6 mb-6 sm:mb-8 pb-5 sm:pb-6 border-b border-stone-200">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-stone-100 text-stone-700 text-xs font-bold tracking-widest uppercase rounded-full mb-3">
            <Sparkles size={14} className="text-nobel-gold" /> CORE MECHANISM SHOWCASE
          </div>
          <h2 className="font-serif text-2xl sm:text-3xl md:text-4xl text-stone-900">
            Escaping Deceptive Optima: SDAO vs. Gradient Descent
          </h2>
          <p className="text-stone-600 text-xs sm:text-sm mt-1 max-w-2xl leading-relaxed">
            Watch 45 candidate solutions initialized in a deceptive local minimum (left well). While classical gradient descent remains permanently trapped, SDAO's density repulsion and OBL catapult the swarm into the true global optimum (right well).
          </p>
        </div>

        {/* Mode Selector */}
        <div className="flex bg-stone-100 p-1.5 rounded-2xl border border-stone-200 w-full sm:w-auto shrink-0">
          <button
            onClick={() => { setMode('SDAO'); initParticles(); }}
            className={`flex-1 sm:flex-none px-3.5 sm:px-6 py-2.5 sm:py-3 rounded-xl text-[11px] sm:text-xs font-bold tracking-wider uppercase transition-all flex items-center justify-center gap-1.5 sm:gap-2 cursor-pointer ${
              mode === 'SDAO' ? 'bg-stone-900 text-white shadow-md' : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            <Zap size={14} className={mode === 'SDAO' ? 'text-nobel-gold' : ''} /> SDAO (Fickian)
          </button>
          <button
            onClick={() => { setMode('SGD'); initParticles(); }}
            className={`flex-1 sm:flex-none px-3.5 sm:px-6 py-2.5 sm:py-3 rounded-xl text-[11px] sm:text-xs font-bold tracking-wider uppercase transition-all flex items-center justify-center gap-1.5 sm:gap-2 cursor-pointer ${
              mode === 'SGD' ? 'bg-red-600 text-white shadow-md' : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            Classical SGD
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 sm:gap-8 items-start">
        {/* Canvas Visualizer Card */}
        <div className="lg:col-span-7 flex flex-col rounded-2xl overflow-hidden border border-stone-300 shadow-sm bg-[#F9F8F4]">
          {/* Viewport */}
          <div className="relative w-full aspect-[4/3] sm:aspect-[16/11] min-h-[260px] sm:min-h-[340px]">
            <canvas
              ref={canvasRef}
              width={640}
              height={440}
              className="w-full h-full block"
            />

            {/* Top Badges */}
            <div className="absolute top-2 left-2 right-2 sm:top-3.5 sm:left-3.5 sm:right-3.5 flex justify-between items-center gap-2 pointer-events-none">
              <div className="bg-amber-100/90 backdrop-blur-md border border-amber-300 text-amber-900 px-2 py-1 sm:px-3 sm:py-1.5 rounded-lg text-[10px] sm:text-xs font-bold flex items-center gap-1.5 shadow-xs">
                <span className="w-2 h-2 sm:w-2.5 sm:h-2.5 rounded-full bg-amber-500 inline-block animate-pulse shrink-0"></span>
                <span>Local Trap <span className="hidden xs:inline">(f ≈ 15.0)</span></span>
              </div>

              <div className="bg-emerald-100/90 backdrop-blur-md border border-emerald-300 text-emerald-900 px-2 py-1 sm:px-3 sm:py-1.5 rounded-lg text-[10px] sm:text-xs font-bold flex items-center gap-1.5 shadow-xs">
                <span className="w-2 h-2 sm:w-2.5 sm:h-2.5 rounded-full bg-emerald-600 inline-block shrink-0"></span>
                <span>Global Basin <span className="hidden xs:inline">(f ≈ 0.05)</span></span>
              </div>
            </div>
          </div>

          {/* Controls & Metrics Toolbar */}
          <div className="bg-white px-3.5 py-3 sm:px-5 sm:py-3.5 border-t border-stone-200 flex flex-wrap justify-between items-center gap-3 text-xs">
            <div className="flex items-center gap-3 sm:gap-5">
              <span className="font-mono text-stone-500 text-[11px] sm:text-xs">
                Step: <strong className="text-stone-900 font-bold">{stepCount}/250</strong>
              </span>
              <span className="font-mono text-stone-500 text-[11px] sm:text-xs">
                Best f(x): <strong className="text-emerald-700 font-bold">{globalBestVal.toFixed(3)}</strong>
              </span>
            </div>

            <div className="flex items-center gap-2 ml-auto sm:ml-0">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className="px-3 py-1.5 bg-stone-900 hover:bg-stone-800 text-white rounded-lg font-bold text-[11px] sm:text-xs uppercase tracking-wider flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                {isRunning ? <Pause size={13} /> : <Play size={13} />}
                {isRunning ? 'Pause' : 'Resume'}
              </button>
              <button
                onClick={initParticles}
                className="px-2.5 py-1.5 bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-300 rounded-lg font-bold text-[11px] sm:text-xs transition-colors flex items-center gap-1 cursor-pointer"
                title="Reset Simulation"
              >
                <RotateCcw size={13} /> Reset
              </button>
            </div>
          </div>
        </div>

        {/* Real-time Diagnostics & Explanations */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-stone-900 text-white p-6 rounded-2xl border border-stone-800 space-y-4">
            <div className="flex justify-between items-center border-b border-stone-800 pb-3">
              <span className="text-xs font-bold tracking-widest text-nobel-gold uppercase">SWARM TELEMETRY</span>
              <span className="text-xs font-mono text-stone-400">N = 45 particles</span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-stone-800 p-3.5 rounded-xl border border-stone-700">
                <div className="text-[11px] text-stone-400 uppercase font-mono">Trapped in Local Well</div>
                <div className={`text-2xl font-serif font-bold mt-1 ${trappedCount > 20 ? 'text-amber-400' : 'text-emerald-400'}`}>
                  {trappedCount} <span className="text-xs font-sans font-normal text-stone-400">/ 45</span>
                </div>
              </div>
              <div className="bg-stone-800 p-3.5 rounded-xl border border-stone-700">
                <div className="text-[11px] text-stone-400 uppercase font-mono">Escaped to Global Well</div>
                <div className={`text-2xl font-serif font-bold mt-1 ${45 - trappedCount > 20 ? 'text-emerald-400' : 'text-stone-400'}`}>
                  {45 - trappedCount} <span className="text-xs font-sans font-normal text-stone-400">/ 45</span>
                </div>
              </div>
            </div>

            {mode === 'SDAO' && (
              <div className="grid grid-cols-2 gap-4 pt-1">
                <div className="bg-stone-800/80 p-3 rounded-lg border border-stone-700/60 flex justify-between items-center">
                  <span className="text-xs text-stone-300">Diffusion D(k):</span>
                  <span className="font-mono font-bold text-nobel-gold">{diffusionCap}</span>
                </div>
                <div className="bg-stone-800/80 p-3 rounded-lg border border-stone-700/60 flex justify-between items-center">
                  <span className="text-xs text-stone-300">OBL Reflections:</span>
                  <span className="font-mono font-bold text-emerald-400">{oblEvents}</span>
                </div>
              </div>
            )}
          </div>

          <div className="p-6 rounded-2xl border transition-all bg-[#F9F8F4] border-stone-300 space-y-3">
            <h3 className="font-serif text-xl text-stone-900 flex items-center gap-2">
              {mode === 'SDAO' ? (
                <>
                  <CheckCircle2 size={20} className="text-emerald-600" />
                  <span>Why SDAO Succeeds</span>
                </>
              ) : (
                <>
                  <Activity size={20} className="text-red-600" />
                  <span>Why Gradient Descent Fails</span>
                </>
              )}
            </h3>

            {mode === 'SDAO' ? (
              <div className="space-y-2.5 text-xs text-stone-700 leading-relaxed">
                <p>
                  <strong>Fick's 2nd Law Active:</strong> Overcrowding in the left trap creates a repulsive force vector D_FL(xᵢ) away from the cluster centroid, forcing particles over the barrier into the right well.
                </p>
                <p>
                  <strong>Opposition-Based Learning (OBL):</strong> Particles that remain stagnant in the left well trigger an opposition jump across the origin (gold flashing rings), directly landing in the global attraction basin.
                </p>
              </div>
            ) : (
              <div className="space-y-2.5 text-xs text-stone-700 leading-relaxed">
                <p>
                  <strong>Zero Gradient Stagnation:</strong> In the bottom of the left well, the local gradient is zero (∇f ≈ 0). Gradient descent relies entirely on local slope, making escape mathematically impossible without external energy.
                </p>
                <p>
                  <strong>Premature Convergence:</strong> All 45 particles permanently collapse into the false minimum (f ≈ 15.0), missing the true global solution (f ≈ 0.05).
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
