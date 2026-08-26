/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, FastForward, Sliders, Activity, Sparkles, AlertCircle, ShieldAlert, Cpu } from 'lucide-react';
import { Reveal } from './Reveal';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { BENCHMARK_FUNCTIONS, evaluateFunction2D } from '../data/sdaoData';
import { Particle, IterationMetric, AlgorithmId } from '../types';

export const SDAOSimulator: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  // Simulator configuration
  const [selectedFuncId, setSelectedFuncId] = useState<string>('rastrigin');
  const [selectedAlgo, setSelectedAlgo] = useState<AlgorithmId | 'SGD'>('SDAO');
  const [popSize, setPopSize] = useState<number>(50);
  const [noiseVar, setNoiseVar] = useState<number>(0);
  const [alpha0, setAlpha0] = useState<number>(0.08);
  const [gamma0, setGamma0] = useState<number>(0.646);
  const [d0, setD0] = useState<number>(3.95);

  // Simulation execution state
  const [isRunning, setIsRunning] = useState<boolean>(true);
  const [iteration, setIteration] = useState<number>(0);
  const [globalBestPos, setGlobalBestPos] = useState<[number, number]>([0, 0]);
  const [globalBestVal, setGlobalBestVal] = useState<number>(9999);
  const [history, setHistory] = useState<IterationMetric[]>([]);
  const [oblTotalCount, setOblTotalCount] = useState<number>(0);
  const [boundContractionCount, setBoundContractionCount] = useState<number>(0);

  // Search bounds state (for bound contraction)
  const currentBoundsRef = useRef<[number, number]>([-5.12, 5.12]);
  const particlesRef = useRef<Particle[]>([]);

  const currentFunc = BENCHMARK_FUNCTIONS.find(f => f.id === selectedFuncId) || BENCHMARK_FUNCTIONS[0];

  // Initialize swarm
  const initSimulation = () => {
    const [minB, maxB] = currentFunc.bounds;
    currentBoundsRef.current = [minB, maxB];
    const newParticles: Particle[] = [];
    let initialBestVal = Infinity;
    let initialBestPos: [number, number] = [0, 0];

    for (let i = 0; i < popSize; i++) {
      const rx = minB + Math.random() * (maxB - minB);
      const ry = minB + Math.random() * (maxB - minB);
      const val = evaluateFunction2D(selectedFuncId, rx, ry, noiseVar);
      
      newParticles.push({
        id: i,
        position: [rx, ry],
        personalBest: [rx, ry],
        personalBestValue: val,
        stagnationCount: 0,
        densityGradient: [0, 0],
        densityValue: 0
      });

      if (val < initialBestVal) {
        initialBestVal = val;
        initialBestPos = [rx, ry];
      }
    }

    particlesRef.current = newParticles;
    setIteration(0);
    setGlobalBestVal(initialBestVal);
    setGlobalBestPos(initialBestPos);
    setHistory([{
      iteration: 0,
      bestValue: Number(initialBestVal.toFixed(2)),
      avgValue: Number(initialBestVal.toFixed(2)),
      diversity: Number(((maxB - minB) * 0.3).toFixed(2)),
      alpha: alpha0,
      gamma: gamma0,
      diffusion: d0,
      oblCount: 0
    }]);
    setOblTotalCount(0);
    setBoundContractionCount(0);
  };

  useEffect(() => {
    initSimulation();
  }, [selectedFuncId, selectedAlgo, popSize, noiseVar]);

  // Step simulation 1 iteration
  const stepIteration = () => {
    const particles = particlesRef.current;
    if (particles.length === 0) return;

    const [minB, maxB] = currentBoundsRef.current;
    const nextIter = iteration + 1;
    const maxIter = 300;
    const lambda = Math.log(10) / maxIter;

    // 1. Calculate Adaptive Coefficients
    const alpha = alpha0 * Math.exp(-lambda * nextIter);
    const avgSC = particles.reduce((acc, p) => acc + p.stagnationCount, 0) / particles.length;
    const gamma = Math.min(0.95, gamma0 * (1 + avgSC / maxIter));
    
    // Time decay & density diffusion
    const dTime = d0 * Math.exp(-0.02 * nextIter);
    const dDensity = dTime * 1.5;
    const currentD = Math.max(0.05, dTime + 0.3 * (dDensity - dTime));

    let newGlobalBestVal = globalBestVal;
    let newGlobalBestPos: [number, number] = [...globalBestPos];
    let stepOblCount = 0;
    let sumVal = 0;
    let sumX = 0, sumY = 0;

    // 2. Update each particle
    particles.forEach((p, idx) => {
      let [x, y] = p.position;

      if (selectedAlgo === 'SGD') {
        // Classical Gradient Descent approx
        const eps = 0.05;
        const fx1 = evaluateFunction2D(selectedFuncId, x + eps, y, noiseVar);
        const fx2 = evaluateFunction2D(selectedFuncId, x - eps, y, noiseVar);
        const fy1 = evaluateFunction2D(selectedFuncId, x, y + eps, noiseVar);
        const fy2 = evaluateFunction2D(selectedFuncId, x, y - eps, noiseVar);
        const gx = (fx1 - fx2) / (2 * eps);
        const gy = (fy1 - fy2) / (2 * eps);

        x -= alpha * 10 * gx;
        y -= alpha * 10 * gy;
      } else if (selectedAlgo === 'TLPSO' || selectedAlgo === 'AMSO') {
        // Swarm PSO approximation: pull toward pbest and gbest with inertia
        const r1 = Math.random();
        const r2 = Math.random();
        const vx = 0.5 * (p.personalBest[0] - x) * r1 + 0.7 * (globalBestPos[0] - x) * r2;
        const vy = 0.5 * (p.personalBest[1] - y) * r1 + 0.7 * (globalBestPos[1] - y) * r2;
        x += vx;
        y += vy;
      } else {
        // SDAO: Density Diffusion + Attraction + Noise
        // A. Density repulsion vector g_i
        let nX = 0, nY = 0, nCount = 0;
        const radius = (maxB - minB) * 0.25;
        particles.forEach((other, j) => {
          if (idx !== j) {
            const dist = Math.hypot(other.position[0] - x, other.position[1] - y);
            if (dist < radius) {
              nX += other.position[0];
              nY += other.position[1];
              nCount++;
            }
          }
        });

        let repX = 0, repY = 0;
        if (nCount >= 2) {
          const cx = nX / nCount;
          const cy = nY / nCount;
          const diffX = x - cx;
          const diffY = y - cy;
          const len = Math.hypot(diffX, diffY) || 1;
          repX = (diffX / len) * (currentD * 0.8);
          repY = (diffY / len) * (currentD * 0.8);
        }
        p.densityGradient = [repX, repY];

        // B. Global & Personal Attraction
        const delta = 0.35; // global attraction
        const attrGX = delta * (globalBestPos[0] - x);
        const attrGY = delta * (globalBestPos[1] - y);
        const attrPX = gamma * 0.4 * (p.personalBest[0] - x);
        const attrPY = gamma * 0.4 * (p.personalBest[1] - y);

        // C. Brownian noise sqrt(2D)*eta
        const noiseX = (Math.random() - 0.5) * Math.sqrt(currentD) * 1.5;
        const noiseY = (Math.random() - 0.5) * Math.sqrt(currentD) * 1.5;

        x = x + repX + attrGX + attrPX + noiseX;
        y = y + repY + attrGY + attrPY + noiseY;

        // D. Opposition-Based Learning (OBL) trigger
        const probOBL = 1 - Math.exp(-0.075 * p.stagnationCount);
        if (p.stagnationCount > 8 && Math.random() < probOBL) {
          x = minB + maxB - x;
          y = minB + maxB - y;
          p.stagnationCount = 0;
          stepOblCount++;
        }
      }

      // Enforce bounds
      x = Math.max(minB, Math.min(maxB, x));
      y = Math.max(minB, Math.min(maxB, y));
      p.position = [x, y];

      // Evaluate new fitness
      const val = evaluateFunction2D(selectedFuncId, x, y, noiseVar);
      sumVal += val;
      sumX += x;
      sumY += y;

      if (val < p.personalBestValue) {
        p.personalBestValue = val;
        p.personalBest = [x, y];
        p.stagnationCount = 0;
      } else {
        p.stagnationCount++;
      }

      if (val < newGlobalBestVal) {
        newGlobalBestVal = val;
        newGlobalBestPos = [x, y];
      }
    });

    // 3. Periodic Bound Contraction (every 10 steps for SDAO)
    if (selectedAlgo === 'SDAO' && nextIter % 10 === 0 && nextIter < 150) {
      const shrink = 0.85;
      const center = newGlobalBestPos;
      const span = (maxB - minB) * shrink * 0.5;
      const nbMin = Math.max(currentFunc.bounds[0], center[0] - span);
      const nbMax = Math.min(currentFunc.bounds[1], center[0] + span);
      currentBoundsRef.current = [nbMin, nbMax];
      setBoundContractionCount(c => c + 1);
    }

    // Calculate diversity (std dev of positions)
    const avgX = sumX / particles.length;
    const avgY = sumY / particles.length;
    const varSum = particles.reduce((acc, p) => acc + Math.pow(p.position[0] - avgX, 2) + Math.pow(p.position[1] - avgY, 2), 0);
    const diversity = Math.sqrt(varSum / particles.length);

    setIteration(nextIter);
    setGlobalBestVal(newGlobalBestVal);
    setGlobalBestPos(newGlobalBestPos);
    setOblTotalCount(c => c + stepOblCount);

    const safeBest = Math.max(0.0001, isNaN(newGlobalBestVal) || !isFinite(newGlobalBestVal) ? 0.0001 : newGlobalBestVal);
    const calculatedAvg = sumVal / particles.length;
    const safeAvg = Math.max(0.0001, isNaN(calculatedAvg) || !isFinite(calculatedAvg) ? 0.0001 : calculatedAvg);

    const newMetric: IterationMetric = {
      iteration: nextIter,
      bestValue: Number(safeBest.toFixed(4)),
      avgValue: Number(safeAvg.toFixed(4)),
      diversity: Number(diversity.toFixed(3)),
      alpha: Number(alpha.toFixed(4)),
      gamma: Number(gamma.toFixed(3)),
      diffusion: Number(currentD.toFixed(3)),
      oblCount: stepOblCount
    };

    setHistory(prev => [...prev.slice(-80), newMetric]);
  };

  // Animation Loop
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isRunning && iteration < 300) {
      timer = setTimeout(() => {
        stepIteration();
      }, 70);
    } else if (iteration >= 300) {
      setIsRunning(false);
    }
    return () => clearTimeout(timer);
  }, [isRunning, iteration]);

  // Render canvas landscape & swarm
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const [minOrig, maxOrig] = currentFunc.bounds;
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Render landscape heatmap (sampled on grid)
    const gridSize = 35;
    const cellW = width / gridSize;
    const cellH = height / gridSize;

    for (let iy = 0; iy < gridSize; iy++) {
      for (let ix = 0; ix < gridSize; ix++) {
        const wx = minOrig + (ix / gridSize) * (maxOrig - minOrig);
        const wy = minOrig + (iy / gridSize) * (maxOrig - minOrig);
        const val = evaluateFunction2D(selectedFuncId, wx, wy, 0);

        // Color map from low (green/gold) to high (stone/cream)
        const maxVal = selectedFuncId === 'schwefel' ? 800 : selectedFuncId === 'rosenbrock' ? 500 : 50;
        const norm = Math.max(0, Math.min(1, val / maxVal));

        // Dark terrain: ink base lifting toward warm ridge; emerald basin
        let r = Math.floor(12 + norm * 55);
        let g = Math.floor(12 + norm * 35);
        let b = Math.floor(15 + norm * 15);

        if (norm < 0.1) {
          r = 16 + Math.floor(norm * 60);
          g = 90 - Math.floor(norm * 120);
          b = 70;
        }

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(ix * cellW, iy * cellH, cellW + 1, cellH + 1);
      }
    }

    // Draw active search bound box if contracted
    const [curMin, curMax] = currentBoundsRef.current;
    if (curMin > minOrig + 0.01 || curMax < maxOrig - 0.01) {
      const bx1 = ((curMin - minOrig) / (maxOrig - minOrig)) * width;
      const by1 = ((curMin - minOrig) / (maxOrig - minOrig)) * height;
      const bw = ((curMax - curMin) / (maxOrig - minOrig)) * width;
      const bh = ((curMax - curMin) / (maxOrig - minOrig)) * height;

      ctx.strokeStyle = '#b8924a';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(bx1, by1, bw, bh);
      ctx.setLineDash([]);
    }

    // Draw Particles
    particlesRef.current.forEach(p => {
      const px = ((p.position[0] - minOrig) / (maxOrig - minOrig)) * width;
      const py = ((p.position[1] - minOrig) / (maxOrig - minOrig)) * height;

      // Draw density repulsion vector if active
      if (selectedAlgo === 'SDAO' && (Math.abs(p.densityGradient[0]) > 0.1 || Math.abs(p.densityGradient[1]) > 0.1)) {
        ctx.strokeStyle = 'rgba(216, 185, 120, 0.75)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + p.densityGradient[0] * 12, py + p.densityGradient[1] * 12);
        ctx.stroke();
      }

      // Particle dot
      ctx.beginPath();
      ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = selectedAlgo === 'SDAO' ? '#d8b978' : '#94a3b8';
      ctx.fill();
      ctx.strokeStyle = 'rgba(246,244,238,0.85)';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Draw Global Best Star
    const gbx = ((globalBestPos[0] - minOrig) / (maxOrig - minOrig)) * width;
    const gby = ((globalBestPos[1] - minOrig) / (maxOrig - minOrig)) * height;
    ctx.beginPath();
    ctx.arc(gbx, gby, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#34d399';
    ctx.fill();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2.5;
    ctx.stroke();

  }, [iteration, selectedFuncId, selectedAlgo]);

  return (
    <Reveal className="w-full my-16">
      <div className="mx-auto mb-16 max-w-3xl">
        <span className="eyebrow">Interactive optimizer</span>
        <h2 className="display mt-4 text-4xl text-bone md:text-5xl">Tune it, run it, watch the swarm converge</h2>
        <p className="mt-5 text-lg font-light leading-relaxed text-white/55">Test SDAO against classical gradient descent and state-of-the-art swarms across benchmark landscapes with customizable noise and adaptive parameter controls.</p>
      </div>

      <div className="bg-white/[0.03] rounded-2xl border border-white/10 shadow-xl overflow-hidden p-6 md:p-8">
        {/* Top Control Bar */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4 pb-6 mb-6 border-b border-white/10">
          <div>
            <label className="block text-xs font-bold text-white/45 uppercase tracking-wider mb-1.5">Benchmark Landscape</label>
            <select
              value={selectedFuncId}
              onChange={(e) => setSelectedFuncId(e.target.value)}
              className="w-full bg-white/[0.03] border border-white/15 rounded-xl px-3.5 py-2.5 text-sm font-bold text-bone focus:outline-none focus:ring-2 focus:ring-gold cursor-pointer"
            >
              {BENCHMARK_FUNCTIONS.map(f => (
                <option key={f.id} value={f.id}>{f.name} ({f.modality})</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-white/45 uppercase tracking-wider mb-1.5">Algorithm</label>
            <select
              value={selectedAlgo}
              onChange={(e) => setSelectedAlgo(e.target.value as any)}
              className="w-full bg-white/[0.03] border border-white/15 rounded-xl px-3.5 py-2.5 text-sm font-bold text-bone focus:outline-none focus:ring-2 focus:ring-gold cursor-pointer"
            >
              <option value="SDAO">⚡ SDAO (Proposed Adaptive Engine)</option>
              <option value="SHADEwithILS">SHADE with ILS</option>
              <option value="AMSO">AMSO (Adaptive Multi-Swarm)</option>
              <option value="TLPSO">TLPSO (Two-Level PSO)</option>
              <option value="SGD">Classical Gradient Descent (SGD)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-white/45 uppercase tracking-wider mb-1.5">
              Stochastic Noise σ²: <span className="text-gold font-mono">{noiseVar > 0 ? `Active (σ=${noiseVar})` : 'None'}</span>
            </label>
            <div className="flex gap-1.5">
              {[0, 0.5, 2.0].map(v => (
                <button
                  key={v}
                  onClick={() => setNoiseVar(v)}
                  className={`flex-1 py-2 rounded-lg text-xs font-bold uppercase transition-all cursor-pointer ${
                    noiseVar === v ? 'bg-white/[0.04] text-white' : 'bg-white/[0.04] text-white/60 hover:bg-white/[0.08]'
                  }`}
                >
                  {v === 0 ? 'Clean' : v === 0.5 ? 'Mild σ' : 'High σ'}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-end gap-2">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className="flex-1 py-2.5 px-4 bg-white/[0.04] hover:bg-white/[0.05] text-white rounded-xl text-xs font-bold uppercase tracking-wider transition-colors flex items-center justify-center gap-2  cursor-pointer"
            >
              {isRunning ? <Pause size={16} /> : <Play size={16} />}
              {isRunning ? 'Pause' : 'Run'}
            </button>
            <button
              onClick={initSimulation}
              className="py-2.5 px-3 bg-white/[0.04] hover:bg-white/[0.08] text-bone border border-white/15 rounded-xl text-xs font-bold transition-colors flex items-center justify-center cursor-pointer"
              title="Reset Swarm"
            >
              <RotateCcw size={16} />
            </button>
            <button
              onClick={stepIteration}
              disabled={isRunning}
              className="py-2.5 px-3 bg-white/[0.04] hover:bg-white/[0.08] disabled:opacity-50 text-bone border border-white/15 rounded-xl text-xs font-bold transition-colors flex items-center justify-center cursor-pointer"
              title="Step 1 Iteration"
            >
              <FastForward size={16} />
            </button>
          </div>
        </div>

        {/* Main Workspace: Landscape Canvas & Convergence Graph */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Canvas Landscape */}
          <div className="lg:col-span-6 space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="display text-lg font-bold text-bone flex items-center gap-2">
                <span>2D Landscape & Swarm Trajectories</span>
                <span className="text-[11px] font-sans font-bold px-2 py-0.5 rounded bg-white/[0.04] text-white/60">
                  {currentFunc.searchSpace}
                </span>
              </h3>
            </div>

            <div className="relative rounded-xl overflow-hidden border border-white/10 bg-ink aspect-square flex items-center justify-center">
              <canvas
                ref={canvasRef}
                width={450}
                height={450}
                className="w-full h-full block cursor-crosshair"
              />
              <div className="absolute bottom-2 left-2 right-2 sm:bottom-3 sm:left-3 sm:right-3 bg-ink/85 backdrop-blur-md px-2.5 py-1.5 sm:px-3.5 sm:py-2 rounded-lg border border-white/10  flex flex-wrap justify-between items-center text-[10px] sm:text-xs font-medium text-white/70 gap-1.5">
                <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                  <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-[#34d399] inline-block"></span> Global Best</span>
                  <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-[#d8b978] inline-block"></span> Particle</span>
                  {selectedAlgo === 'SDAO' && <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-gold inline-block"></span> Repulsion</span>}
                </div>
                <div className="font-mono font-bold text-white/45 ml-auto">Iter: {iteration}/300</div>
              </div>
            </div>
          </div>

          {/* Real-time Graph & Diagnostic Cards */}
          <div className="lg:col-span-6 space-y-6">
            <div className="flex justify-between items-center">
              <h3 className="display text-lg font-bold text-bone">
                Live Convergence Curve & Adaptive Diagnostics
              </h3>
              <span className="text-xs font-mono text-emerald-700 font-bold bg-emerald-50 px-2.5 py-1 rounded border border-emerald-200">
                Best Error: {globalBestVal.toExponential(3)}
              </span>
            </div>

            {/* Recharts Chart */}
            <div className="h-64 w-full bg-white/[0.03] p-4 rounded-xl border border-white/10 ">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="iteration" stroke="rgba(255,255,255,0.4)" fontSize={11} />
                  <YAxis scale="log" domain={[0.0001, 'auto']} allowDataOverflow={false} stroke="rgba(255,255,255,0.4)" fontSize={11} width={55} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0a0a0a', color: '#f6f4ee', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.12)', fontSize: '12px' }}
                    labelFormatter={(label) => `Iteration: #${label}`}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Line type="monotone" name="Best Value f(x)" dataKey="bestValue" stroke="#b8924a" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                  <Line type="monotone" name="Swarm Average" dataKey="avgValue" stroke="rgba(255,255,255,0.45)" strokeWidth={1.5} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Diagnostic Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-white/[0.03] p-3.5 rounded-xl border border-white/10 text-center">
                <div className="text-[11px] text-white/45 uppercase font-bold">Step Size α(k)</div>
                <div className="text-lg font-mono font-bold text-bone mt-1">
                  {history.length > 0 ? history[history.length - 1].alpha : alpha0}
                </div>
              </div>
              <div className="bg-white/[0.03] p-3.5 rounded-xl border border-white/10 text-center">
                <div className="text-[11px] text-white/45 uppercase font-bold">Memory γ(k)</div>
                <div className="text-lg font-mono font-bold text-gold mt-1">
                  {history.length > 0 ? history[history.length - 1].gamma : gamma0}
                </div>
              </div>
              <div className="bg-white/[0.03] p-3.5 rounded-xl border border-white/10 text-center">
                <div className="text-[11px] text-white/45 uppercase font-bold">Diffusion D(k)</div>
                <div className="text-lg font-mono font-bold text-bone mt-1">
                  {history.length > 0 ? history[history.length - 1].diffusion : d0}
                </div>
              </div>
              <div className="bg-white/[0.03] p-3.5 rounded-xl border border-white/10 text-center">
                <div className="text-[11px] text-white/45 uppercase font-bold">OBL Jumps</div>
                <div className="text-lg font-mono font-bold text-emerald-600 mt-1">
                  {oblTotalCount}
                </div>
              </div>
            </div>

            {selectedAlgo === 'SDAO' && (
              <div className="p-4 bg-gold/10 rounded-xl border border-gold/30 text-xs text-bone leading-relaxed flex items-center justify-between">
                <div>
                  <strong>Bound Contraction Events:</strong> Search bounds contracted {boundContractionCount} times around x_Gbest (Every 10 iterations).
                </div>
                <span className="font-mono font-bold px-2 py-1 bg-white/[0.03] rounded border border-gold/40">
                  {currentBoundsRef.current[0].toFixed(1)} to {currentBoundsRef.current[1].toFixed(1)}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </Reveal>
  );
};
