/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Sliders, Zap, Minimize2, TrendingDown, RefreshCw, Layers, ShieldAlert, Cpu } from 'lucide-react';
import { Reveal } from './Reveal';

export const AdaptiveMechanics: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'alpha' | 'gamma' | 'diffusion' | 'obl' | 'contraction'>('alpha');
  
  // Interactive Controls state
  const [alpha0, setAlpha0] = useState<number>(0.08);
  const [maxIter, setMaxIter] = useState<number>(300);
  const [stagnationCount, setStagnationCount] = useState<number>(15);
  const [gamma0, setGamma0] = useState<number>(0.646);
  const [d0, setD0] = useState<number>(3.95);
  const [rhoGlobal, setRhoGlobal] = useState<number>(1.5);
  const [dim, setDim] = useState<number>(50);

  // Calculate curves for display
  const lambda = Math.log(10) / maxIter;
  
  // Generate curve points for Learning Rate
  const alphaCurve = Array.from({ length: 20 }, (_, i) => {
    const k = Math.round((i / 19) * maxIter);
    const val = alpha0 * Math.exp(-lambda * k);
    return { k, val: Number(val.toFixed(4)) };
  });

  // Generate OBL probability curve
  const oblCurve = Array.from({ length: 20 }, (_, i) => {
    const sc = Math.round((i / 19) * 50);
    const p = 1 - Math.exp(-0.075 * sc);
    return { sc, p: Number((p * 100).toFixed(1)) };
  });

  // Calculate current dynamic values
  const currentAlpha = (alpha0 * Math.exp(-lambda * (maxIter * 0.5))).toFixed(4);
  const currentGamma = (gamma0 * (1 + stagnationCount / maxIter)).toFixed(3);
  const currentP = ((1 - Math.exp(-0.075 * stagnationCount)) * 100).toFixed(1);
  
  // Diffusion calculations
  const dTime = d0 * Math.exp(-0.02 * (maxIter * 0.3));
  const dDensity = dTime * (1 + 0.5 * rhoGlobal);
  const dRaw = dTime + 0.5 * (dDensity - dTime);
  const dCap = (0.5 * (2 * 0.2 - 0.05)) / (2 * dim) * 100; // Scaled for viz
  const finalD = Math.min(dRaw, Math.max(0.1, dCap)).toFixed(3);

  return (
    <Reveal className="w-full my-16">
      <div className="mx-auto mb-16 max-w-3xl">
        <span className="eyebrow">Adaptive engine</span>
        <h2 className="display mt-4 text-4xl text-bone md:text-5xl">Five mechanisms, self-tuning across the horizon</h2>
        <p className="mt-5 text-lg font-light leading-relaxed text-white/55">How SDAO dynamically modulates learning rates, memory coefficients, diffusion intensity, opposition reflection, and bound contraction over the optimization horizon.</p>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap justify-center gap-2 mb-8 border-b border-white/10 pb-4">
        {[
          { id: 'alpha', label: '1. Learning Rate α(k)', icon: <TrendingDown size={16} /> },
          { id: 'gamma', label: '2. Memory Coeff γ(k)', icon: <RefreshCw size={16} /> },
          { id: 'diffusion', label: '3. Diffusion D(k) & Cap', icon: <Layers size={16} /> },
          { id: 'obl', label: '4. Opposition-Based Learning', icon: <Zap size={16} /> },
          { id: 'contraction', label: '5. Periodic Bound Contraction', icon: <Minimize2 size={16} /> },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`px-5 py-3 rounded-xl text-sm font-bold tracking-wide transition-all flex items-center gap-2.5 cursor-pointer ${
              activeTab === tab.id
                ? 'bg-bone text-ink'
                : 'border border-white/15 text-white/55 hover:border-gold/50 hover:text-bone'
            }`}
          >
            <span className={activeTab === tab.id ? 'text-gold' : 'text-white/35'}>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-8 md:p-12">
        {activeTab === 'alpha' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-6 space-y-6">
              <h3 className="display text-3xl text-bone">
                Dynamic Learning Rate Adaptation
              </h3>
              <p className="text-white/60 leading-relaxed">
                The step size parameter α(k) regulates the magnitude of deterministic descent toward minima in the objective function landscape. To balance initial broad exploration with fine-grained late-stage exploitation, SDAO applies an exponential decay:
              </p>
              <div className="bg-white/[0.04] p-5 rounded-xl border border-white/10 font-mono text-center">
                <div className="text-xl md:text-2xl font-bold text-bone mb-2">
                  α(k) = α₀ · exp(-λk), \quad λ = ln(10) / N
                </div>
                <div className="text-xs text-white/45 font-sans mt-2">
                  Derivation (Appendix B): Ensures α(N) = 0.1 α₀ (exactly 10% of initial step size at final iteration N).
                </div>
              </div>

              <div className="space-y-4 pt-4 border-t border-white/10">
                <div>
                  <div className="flex justify-between text-xs font-bold text-white/70 uppercase mb-1">
                    <span>Initial Learning Rate (α₀):</span>
                    <span className="text-gold font-mono">{alpha0}</span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="0.2"
                    step="0.01"
                    value={alpha0}
                    onChange={(e) => setAlpha0(parseFloat(e.target.value))}
                    className="w-full accent-gold cursor-pointer"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs font-bold text-white/70 uppercase mb-1">
                    <span>Max Iterations (N):</span>
                    <span className="text-gold font-mono">{maxIter}</span>
                  </div>
                  <input
                    type="range"
                    min="100"
                    max="1000"
                    step="50"
                    value={maxIter}
                    onChange={(e) => setMaxIter(parseInt(e.target.value))}
                    className="w-full accent-gold cursor-pointer"
                  />
                </div>
              </div>
            </div>

            <div className="lg:col-span-6 bg-white/[0.04] p-8 rounded-2xl border border-white/10 text-white flex flex-col justify-between">
              <div className="text-xs font-bold tracking-widest text-gold uppercase mb-4">DECAY PROFILE VISUALIZER</div>
              <div className="h-64 flex items-end justify-between gap-1 pb-6 border-b border-white/10 relative">
                {alphaCurve.map((pt, i) => {
                  const heightPct = Math.max(8, (pt.val / alpha0) * 100);
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center group relative">
                      <div className="opacity-0 group-hover:opacity-100 absolute -top-8 bg-white/[0.05] text-gold px-2 py-0.5 rounded text-[10px] font-mono whitespace-nowrap transition-opacity z-10 pointer-events-none">
                        k={pt.k}: α={pt.val}
                      </div>
                      <div
                        className="w-full bg-gradient-to-t from-gold/40 to-gold rounded-t transition-all group-hover:brightness-125"
                        style={{ height: `${heightPct}%` }}
                      ></div>
                    </div>
                  );
                })}
              </div>
              <div className="flex justify-between text-xs font-mono text-white/35 mt-3">
                <span>Start: Iteration 0 (α = {alpha0})</span>
                <span>Midpoint: α ≈ {currentAlpha}</span>
                <span>End: Iteration {maxIter} (α = {(alpha0 * 0.1).toFixed(4)})</span>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'gamma' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-7 space-y-6">
              <h3 className="display text-3xl text-bone">
                Dynamic Memory Coefficient (γ)
              </h3>
              <p className="text-white/60 leading-relaxed">
                The memory coefficient γ modulates the attraction intensity toward each particle's historically best-known position (x_best,i). To overcome slow progress or stagnation, SDAO dynamically scales γ with the average stagnation count (SC_avg):
              </p>
              <div className="bg-white/[0.04] p-5 rounded-xl border border-white/10 font-mono text-center">
                <div className="text-xl md:text-2xl font-bold text-bone mb-2">
                  γ_dynamic = γ₀ × (1 + SC_avg / N)
                </div>
              </div>
              <p className="text-white/60 text-sm leading-relaxed">
                When the swarm shows limited improvement (SC_avg increases), reinforcing personal memory pulls particles back toward confirmed high-quality basins, preventing random wandering.
              </p>

              <div className="pt-4 border-t border-white/10">
                <div className="flex justify-between text-xs font-bold text-white/70 uppercase mb-1">
                  <span>Average Swarm Stagnation Count (SC_avg):</span>
                  <span className="text-gold font-mono">{stagnationCount} iterations</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="150"
                  step="1"
                  value={stagnationCount}
                  onChange={(e) => setStagnationCount(parseInt(e.target.value))}
                  className="w-full accent-gold cursor-pointer"
                />
              </div>
            </div>

            <div className="lg:col-span-5 bg-white/[0.04] p-8 rounded-2xl border border-white/10 text-white space-y-6">
              <div className="text-xs font-bold tracking-widest text-gold uppercase">STAGNATION RESPONSE</div>
              <div className="bg-white/[0.05] p-6 rounded-xl border border-white/10 text-center space-y-2">
                <div className="text-xs text-white/35 font-sans uppercase">Base γ₀ = {gamma0}</div>
                <div className="text-4xl display font-bold text-gold">{currentGamma}</div>
                <div className="text-xs text-emerald-400 font-mono">
                  +{((parseFloat(currentGamma) / gamma0 - 1) * 100).toFixed(1)}% Memory Reinforcement
                </div>
              </div>
              <p className="text-white/55 text-xs leading-relaxed">
                Alternatively, γ can be modulated by population diversity: decreasing when spread σ_pop(k) is high to encourage exploration, and increasing as diversity collapses for fine exploitation.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'diffusion' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-7 space-y-6">
              <h3 className="display text-3xl text-bone">
                Dynamic Diffusion D(k) & Dimension Cap
              </h3>
              <p className="text-white/60 leading-relaxed">
                SDAO regulates Brownian noise intensity through a 3-stage adaptive schedule:
              </p>
              <div className="space-y-3 font-mono text-xs">
                <div className="p-3.5 bg-white/[0.03] rounded-xl border border-white/10 flex justify-between items-center">
                  <span className="text-white/45 font-sans font-bold">1. Time Decay:</span>
                  <span className="text-bone font-bold">D_time = D₀ · exp(-βk)</span>
                </div>
                <div className="p-3.5 bg-white/[0.03] rounded-xl border border-white/10 flex justify-between items-center">
                  <span className="text-white/45 font-sans font-bold">2. Density Boost:</span>
                  <span className="text-bone font-bold">D_density = D_time · (1 + 0.5 ρ_global)</span>
                </div>
                <div className="p-3.5 bg-white/[0.04] text-gold rounded-xl border border-white/10 flex justify-between items-center">
                  <span className="text-white/55 font-sans font-bold">3. Dimension Cap (Eq 15):</span>
                  <span className="font-bold">D_cap(k) := c(2α_c(k) - L_b(k)²) / (2n)</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 pt-4 border-t border-white/10">
                <div>
                  <div className="flex justify-between text-xs font-bold text-white/70 uppercase mb-1">
                    <span>Global Density (ρ_global):</span>
                    <span className="text-gold font-mono">{rhoGlobal}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="5.0"
                    step="0.1"
                    value={rhoGlobal}
                    onChange={(e) => setRhoGlobal(parseFloat(e.target.value))}
                    className="w-full accent-gold cursor-pointer"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs font-bold text-white/70 uppercase mb-1">
                    <span>Dimension (d = n):</span>
                    <span className="text-gold font-mono">{dim}D</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="500"
                    step="10"
                    value={dim}
                    onChange={(e) => setDim(parseInt(e.target.value))}
                    className="w-full accent-gold cursor-pointer"
                  />
                </div>
              </div>
            </div>

            <div className="lg:col-span-5 bg-white/[0.04] p-8 rounded-2xl border border-white/10 text-white space-y-6">
              <div className="text-xs font-bold tracking-widest text-gold uppercase">VARIANCE REGULATOR OUTPUT</div>
              <div className="bg-white/[0.05] p-6 rounded-xl border border-white/10 text-center space-y-2">
                <div className="text-xs text-white/35 font-sans uppercase">Effective Diffusion Coeff D(k)</div>
                <div className="text-4xl display font-bold text-gold">{finalD}</div>
                <div className="text-xs text-white/35 font-mono mt-2">
                  {dim >= 100 ? 'Clamped by Dimension Cap (High-d Stability)' : 'Driven by Density & Time Decay'}
                </div>
              </div>
              <p className="text-white/55 text-xs leading-relaxed">
                Notice how increasing dimensionality to d = 500 automatically tightens the diffusion cap, suppressing additive noise variance growth 2Dn·h so the swarm never diverges!
              </p>
            </div>
          </div>
        )}

        {activeTab === 'obl' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-6 space-y-6">
              <h3 className="display text-3xl text-bone">
                Opposition-Based Learning (OBL) Engine
              </h3>
              <p className="text-white/60 leading-relaxed">
                When a particle exhibits prolonged stagnation (SCᵢ climbs), SDAO applies Opposition-Based Learning in a probabilistic manner to catapult it across the search domain:
              </p>
              <div className="bg-white/[0.04] p-5 rounded-xl border border-white/10 font-mono text-center">
                <div className="text-xl md:text-2xl font-bold text-bone mb-2">
                  Pᵢ = 1 - exp(-λ · SCᵢ)
                </div>
                <div className="text-xs text-white/45 font-sans mt-2">
                  If random u ∈ [0, 1] &lt; Pᵢ, particle reflects to opposite position: x_opp = lⱼ + uⱼ - xᵢ.
                </div>
              </div>

              <div className="pt-4 border-t border-white/10">
                <div className="flex justify-between text-xs font-bold text-white/70 uppercase mb-1">
                  <span>Particle Stagnation Counter (SCᵢ):</span>
                  <span className="text-gold font-mono">{stagnationCount} iterations</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="60"
                  step="1"
                  value={stagnationCount}
                  onChange={(e) => setStagnationCount(parseInt(e.target.value))}
                  className="w-full accent-gold cursor-pointer"
                />
              </div>
            </div>

            <div className="lg:col-span-6 bg-white/[0.04] p-8 rounded-2xl border border-white/10 text-white flex flex-col justify-between">
              <div className="text-xs font-bold tracking-widest text-gold uppercase mb-4">OBL TRIGGER PROBABILITY CURVE</div>
              <div className="h-64 flex items-end justify-between gap-1 pb-6 border-b border-white/10 relative">
                {oblCurve.map((pt, i) => {
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center group relative">
                      <div className="opacity-0 group-hover:opacity-100 absolute -top-8 bg-white/[0.05] text-gold px-2 py-0.5 rounded text-[10px] font-mono whitespace-nowrap transition-opacity z-10 pointer-events-none">
                        SC={pt.sc}: P={pt.p}%
                      </div>
                      <div
                        className={`w-full rounded-t transition-all group-hover:brightness-125 ${
                          pt.sc <= stagnationCount ? 'bg-gold' : 'bg-white/10'
                        }`}
                        style={{ height: `${Math.max(6, pt.p)}%` }}
                      ></div>
                    </div>
                  );
                })}
              </div>
              <div className="flex justify-between items-center mt-3">
                <span className="text-xs font-mono text-white/35">Current Trigger Chance:</span>
                <span className="text-2xl display font-bold text-emerald-400">{currentP}%</span>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'contraction' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-7 space-y-6">
              <h3 className="display text-3xl text-bone">
                Periodic Bound Contraction (Eq. 12 & 16)
              </h3>
              <p className="text-white/60 leading-relaxed">
                Every contraction interval (m = 10 iterations), SDAO shrinks the global search bounds [lⱼ, uⱼ] around the best solution xⱼ* found so far. This concentrates particle density in the most promising hyper-volume:
              </p>
              <div className="bg-white/[0.04] p-5 rounded-xl border border-white/10 font-mono text-center space-y-2 text-sm md:text-base text-bone font-bold">
                <div>new_lowerⱼ = xⱼ* - γ(xⱼ* - lⱼ)</div>
                <div>new_upperⱼ = xⱼ* + γ(uⱼ - xⱼ*)</div>
              </div>
              <div className="p-4 bg-white/[0.03] rounded-xl border border-white/10 text-xs text-white/70 leading-relaxed">
                <strong>Geometric Contraction Theorem:</strong> The interval length reduces by factor γ ∈ (0, 1). Iterating m times yields a search diameter γᵐ(uⱼ - lⱼ) → 0 as m → ∞, accelerating late-stage precision while preserving invariance of the feasible set!
              </div>
            </div>

            <div className="lg:col-span-5 bg-white/[0.04] p-8 rounded-2xl border border-white/10 text-white space-y-6">
              <div className="text-xs font-bold tracking-widest text-gold uppercase">BOUND CONTRACTION SIMULATOR</div>
              <div className="space-y-4">
                {[
                  { iter: 'Iter 0 (Initial)', range: '[-100.0, +100.0]', width: '100%', color: 'bg-white/15' },
                  { iter: 'Iter 10 (1st Contraction)', range: '[-35.4, +64.6]', width: '65%', color: 'bg-white/20' },
                  { iter: 'Iter 20 (2nd Contraction)', range: '[ -6.2, +33.8]', width: '42%', color: 'bg-gold/80' },
                  { iter: 'Iter 30 (3rd Contraction)', range: '[ +5.4, +14.6]', width: '25%', color: 'bg-gold' },
                  { iter: 'Iter 40+ (Hyper-Focus)', range: '[ +9.8, +10.2]', width: '12%', color: 'bg-emerald-500' },
                ].map((step, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-xs font-mono text-white/55">
                      <span>{step.iter}</span>
                      <span className="font-bold text-gold">{step.range}</span>
                    </div>
                    <div className="w-full bg-white/[0.05] h-4 rounded-full overflow-hidden flex items-center p-0.5">
                      <div className={`${step.color} h-full rounded-full transition-all duration-500`} style={{ width: step.width }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </Reveal>
  );
};
