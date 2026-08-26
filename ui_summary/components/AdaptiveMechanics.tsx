/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Sliders, Zap, Minimize2, TrendingDown, RefreshCw, Layers, ShieldAlert, Cpu } from 'lucide-react';

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
    <div className="w-full my-16">
      <div className="text-center max-w-3xl mx-auto mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-stone-100 text-stone-700 text-xs font-bold tracking-widest uppercase rounded-full mb-4 border border-stone-200">
          <Sliders size={14} className="text-nobel-gold" /> SECTION 3: CORE METHODOLOGY
        </div>
        <h2 className="font-serif text-4xl md:text-5xl text-stone-900 mb-4">
          Adaptive Mechanisms & OBL Engine
        </h2>
        <p className="text-lg text-stone-600 font-light leading-relaxed">
          How SDAO dynamically modulates learning rates, memory coefficients, diffusion intensity, opposition reflection, and bound contraction over the optimization horizon.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap justify-center gap-2 mb-8 border-b border-stone-200 pb-4">
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
                ? 'bg-stone-900 text-white shadow-md scale-105'
                : 'bg-white hover:bg-stone-100 text-stone-600 border border-stone-200'
            }`}
          >
            <span className={activeTab === tab.id ? 'text-nobel-gold' : 'text-stone-400'}>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      <div className="bg-white rounded-2xl border border-stone-200 shadow-xl p-8 md:p-12">
        {activeTab === 'alpha' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-6 space-y-6">
              <h3 className="font-serif text-3xl text-stone-900">
                Dynamic Learning Rate Adaptation
              </h3>
              <p className="text-stone-600 leading-relaxed">
                The step size parameter α(k) regulates the magnitude of deterministic descent toward minima in the objective function landscape. To balance initial broad exploration with fine-grained late-stage exploitation, SDAO applies an exponential decay:
              </p>
              <div className="bg-[#F9F8F4] p-5 rounded-xl border border-stone-200 font-mono text-center">
                <div className="text-xl md:text-2xl font-bold text-stone-900 mb-2">
                  α(k) = α₀ · exp(-λk), \quad λ = ln(10) / k_max
                </div>
              </div>
              <p className="text-stone-600 text-sm leading-relaxed">
                Tune the initial learning rate α₀ and optimization horizon k_max below to see how the decay profile shifts.
              </p>
              <div className="space-y-4 pt-2">
                <div>
                  <div className="flex justify-between text-xs font-mono text-stone-600 mb-1.5">
                    <span>α₀ (Initial Learning Rate)</span>
                    <span className="text-stone-900 font-bold">{alpha0.toFixed(3)}</span>
                  </div>
                  <input type="range" min="0.01" max="0.2" step="0.005" value={alpha0}
                    onChange={(e) => setAlpha0(Number(e.target.value))}
                    className="w-full accent-stone-900 cursor-pointer" />
                </div>
                <div>
                  <div className="flex justify-between text-xs font-mono text-stone-600 mb-1.5">
                    <span>k_max (Max Iterations)</span>
                    <span className="text-stone-900 font-bold">{maxIter}</span>
                  </div>
                  <input type="range" min="100" max="500" step="10" value={maxIter}
                    onChange={(e) => setMaxIter(Number(e.target.value))}
                    className="w-full accent-stone-900 cursor-pointer" />
                </div>
              </div>
            </div>

            <div className="lg:col-span-6 bg-stone-50 p-8 rounded-2xl border border-stone-200 text-stone-900 flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-center mb-4">
                  <span className="text-xs font-bold uppercase tracking-widest text-stone-500">α(k) Decay Curve</span>
                  <span className="font-mono text-sm font-bold text-nobel-gold">At k = {Math.round(maxIter / 2)}: α = {currentAlpha}</span>
                </div>
                <div className="flex items-end gap-1.5 h-48 relative">
                  {alphaCurve.map((pt, i) => {
                    const maxVal = alphaCurve[0].val;
                    const h = (pt.val / maxVal) * 100;
                    return (
                      <div key={i} className="flex-1 relative group">
                        <div
                          className="w-full bg-gradient-to-t from-nobel-gold/40 to-nobel-gold rounded-t transition-all group-hover:brightness-125"
                          style={{ height: `${Math.max(2, h)}%`, position: 'absolute', bottom: 0 }}
                        />
                        <div className="opacity-0 group-hover:opacity-100 absolute -top-8 bg-stone-800 text-white px-2 py-0.5 rounded text-[10px] font-mono whitespace-nowrap transition-opacity z-10 pointer-events-none">
                          k={pt.k}, α={pt.val}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between text-[10px] font-mono text-stone-400 mt-2">
                  <span>k = 0</span>
                  <span>k = {maxIter}</span>
                </div>
              </div>
              <div className="mt-4 p-3.5 bg-white rounded-xl border border-stone-200 text-xs text-stone-700 leading-relaxed">
                <strong className="text-stone-900">Interpretation:</strong> Early iterations use large α for global exploration; as k → k_max, α decays exponentially, concentrating the swarm into a fine exploitation phase around the best basin.
              </div>
            </div>
          </div>
        )}

        {activeTab === 'gamma' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-6 space-y-6">
              <h3 className="font-serif text-3xl text-stone-900">
                Memory Coefficient & Stagnation Sensing
              </h3>
              <p className="text-stone-600 leading-relaxed">
                The memory coefficient γ(k) weights the pull toward each particle's personal best x_best,i. Unlike α(k), γ(k) <strong>increases</strong> with the swarm's stagnation count, intensifying local search when progress stalls.
              </p>
              <div className="bg-[#F9F8F4] p-5 rounded-xl border border-stone-200 font-mono text-center">
                <div className="text-xl md:text-2xl font-bold text-stone-900 mb-2">
                  γ(k) = γ₀ · (1 + SC_avg / k_max)
                </div>
              </div>
              <div className="space-y-4 pt-2">
                <div>
                  <div className="flex justify-between text-xs font-mono text-stone-600 mb-1.5">
                    <span>γ₀ (Base Memory Coeff)</span>
                    <span className="text-stone-900 font-bold">{gamma0.toFixed(3)}</span>
                  </div>
                  <input type="range" min="0.1" max="1.0" step="0.01" value={gamma0}
                    onChange={(e) => setGamma0(Number(e.target.value))}
                    className="w-full accent-stone-900 cursor-pointer" />
                </div>
                <div>
                  <div className="flex justify-between text-xs font-mono text-stone-600 mb-1.5">
                    <span>Average Swarm Stagnation Count (SC_avg):</span>
                    <span className="text-stone-900 font-bold">{stagnationCount}</span>
                  </div>
                  <input type="range" min="0" max="50" step="1" value={stagnationCount}
                    onChange={(e) => setStagnationCount(Number(e.target.value))}
                    className="w-full accent-stone-900 cursor-pointer" />
                </div>
              </div>
            </div>

            <div className="lg:col-span-6 bg-stone-50 p-8 rounded-2xl border border-stone-200 text-stone-900 space-y-6">
              <div className="bg-white p-6 rounded-xl border border-stone-200 text-center space-y-2">
                <div className="text-xs font-bold uppercase tracking-widest text-stone-500">Current γ(k)</div>
                <div className="text-4xl font-mono font-bold text-nobel-gold">{currentGamma}</div>
                <div className="text-xs text-stone-600">
                  Base γ₀={gamma0.toFixed(3)} amplified by stagnation factor (1 + {stagnationCount}/{maxIter})
                </div>
              </div>
              <div className="p-3.5 bg-white rounded-xl border border-stone-200 text-xs text-stone-700 leading-relaxed">
                <strong className="text-stone-900">Stagnation Adaptation:</strong> When the global best fails to improve for m iterations, γ(k) grows, reinforcing exploitation of promising personal memories — a self-correcting feedback loop.
              </div>
              <div className="p-3.5 bg-nobel-gold/10 rounded-xl border border-nobel-gold/30 text-xs text-stone-800 leading-relaxed">
                <strong className="text-nobel-gold">Bound:</strong> γ(k) is capped so δ(k) + γ(k) ≤ 1, preserving the contraction guarantee and mean-square stability.
              </div>
            </div>
          </div>
        )}

        {activeTab === 'diffusion' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-6 space-y-6">
              <h3 className="font-serif text-3xl text-stone-900">
                Diffusion Coefficient D(k) & Dimension-Aware Cap
              </h3>
              <p className="text-stone-600 leading-relaxed">
                The diffusion coefficient controls the magnitude of stochastic noise √(2D)·η. SDAO computes a raw adapted value D_raw(k) from time-decay and density, then clamps it against a dimension-aware ceiling D_cap(k) to prevent variance explosion in high dimensions.
              </p>
              <div className="bg-[#F9F8F4] p-5 rounded-xl border border-stone-200 font-mono text-center space-y-3">
                <div className="text-xs text-stone-400 font-sans tracking-widest uppercase">Dimension-Aware Cap (Eq. 15)</div>
                <div className="text-base md:text-lg font-bold text-stone-900 leading-relaxed">
                  D(k) = min( D_raw(k), c·(2α_c - L_b²) / (2n) )
                </div>
              </div>
              <div className="space-y-4 pt-2">
                <div>
                  <div className="flex justify-between text-xs font-mono text-stone-600 mb-1.5">
                    <span>D₀ (Initial Diffusion)</span>
                    <span className="text-stone-900 font-bold">{d0.toFixed(2)}</span>
                  </div>
                  <input type="range" min="0.5" max="8.0" step="0.05" value={d0}
                    onChange={(e) => setD0(Number(e.target.value))}
                    className="w-full accent-stone-900 cursor-pointer" />
                </div>
                <div>
                  <div className="flex justify-between text-xs font-mono text-stone-600 mb-1.5">
                    <span>ρ_global (Global Density)</span>
                    <span className="text-stone-900 font-bold">{rhoGlobal.toFixed(2)}</span>
                  </div>
                  <input type="range" min="0" max="3.0" step="0.05" value={rhoGlobal}
                    onChange={(e) => setRhoGlobal(Number(e.target.value))}
                    className="w-full accent-stone-900 cursor-pointer" />
                </div>
                <div>
                  <div className="flex justify-between text-xs font-mono text-stone-600 mb-1.5">
                    <span>n (Dimension)</span>
                    <span className="text-stone-900 font-bold">{dim}</span>
                  </div>
                  <input type="range" min="10" max="500" step="10" value={dim}
                    onChange={(e) => setDim(Number(e.target.value))}
                    className="w-full accent-stone-900 cursor-pointer" />
                </div>
              </div>
            </div>

            <div className="lg:col-span-6 bg-stone-50 p-8 rounded-2xl border border-stone-200 text-stone-900 space-y-6">
              <div className="bg-white p-6 rounded-xl border border-stone-200 text-center space-y-2">
                <div className="text-xs font-bold uppercase tracking-widest text-stone-500">Final D(k) after Cap</div>
                <div className="text-4xl font-mono font-bold text-nobel-gold">{finalD}</div>
                <div className="text-xs text-stone-600">
                  min( D_raw={dRaw.toFixed(3)}, D_cap={dCap.toFixed(3)} )
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3.5 bg-white rounded-xl border border-stone-200 flex justify-between items-center">
                  <span className="text-xs text-stone-600">D_time (decay)</span>
                  <span className="font-mono font-bold text-stone-900">{dTime.toFixed(3)}</span>
                </div>
                <div className="p-3.5 bg-white rounded-xl border border-stone-200 flex justify-between items-center">
                  <span className="text-xs text-stone-600">D_density boost</span>
                  <span className="font-mono font-bold text-stone-900">{dDensity.toFixed(3)}</span>
                </div>
              </div>
              <div className="p-3.5 bg-nobel-gold/10 rounded-xl border border-nobel-gold/30 text-xs text-stone-800 leading-relaxed">
                <strong className="text-nobel-gold">High-d Safety:</strong> As n → 500, the cap D_cap shrinks as O(1/n), ensuring E[||X_k||²] stays bounded and convergence order ½ holds even at d = 500.
              </div>
            </div>
          </div>
        )}

        {activeTab === 'obl' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-6 space-y-6">
              <h3 className="font-serif text-3xl text-stone-900">
                Opposition-Based Learning (OBL) Engine
              </h3>
              <p className="text-stone-600 leading-relaxed">
                When particles stagnate in a local trap, SDAO probabilistically reflects them to the <strong>opposite</strong> side of the search space — a powerful escape mechanism that leaps across basins without gradient information.
              </p>
              <div className="bg-[#F9F8F4] p-5 rounded-xl border border-stone-200 font-mono text-center space-y-3">
                <div className="text-xs text-stone-400 font-sans tracking-widest uppercase">Opposition Reflection</div>
                <div className="text-lg md:text-xl font-bold text-stone-900 leading-relaxed">
                  x̂ᵢ = (a + b) - xᵢ, \quad P(OBL) = 1 - exp(-0.075·SC)
                </div>
              </div>
              <div className="space-y-4 pt-2">
                <div>
                  <div className="flex justify-between text-xs font-mono text-stone-600 mb-1.5">
                    <span>Stagnation Count (SC)</span>
                    <span className="text-stone-900 font-bold">{stagnationCount}</span>
                  </div>
                  <input type="range" min="0" max="50" step="1" value={stagnationCount}
                    onChange={(e) => setStagnationCount(Number(e.target.value))}
                    className="w-full accent-stone-900 cursor-pointer" />
                </div>
              </div>
            </div>

            <div className="lg:col-span-6 bg-stone-50 p-8 rounded-2xl border border-stone-200 text-stone-900 space-y-6">
              <div className="bg-white p-6 rounded-xl border border-stone-200 text-center space-y-2">
                <div className="text-xs font-bold uppercase tracking-widest text-stone-500">OBL Trigger Probability</div>
                <div className="text-4xl font-mono font-bold text-nobel-gold">{currentP}%</div>
                <div className="text-xs text-stone-600">
                  1 - exp(-0.075 × {stagnationCount})
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs font-mono text-stone-600 mb-1.5">
                  <span>P(OBL) vs Stagnation</span>
                  <span className="text-stone-900 font-bold">{currentP}%</span>
                </div>
                <div className="flex items-end gap-1.5 h-32 relative">
                  {oblCurve.map((pt, i) => {
                    const h = pt.p;
                    return (
                      <div key={i} className="flex-1 relative group">
                        <div
                          className="w-full bg-gradient-to-t from-emerald-500/40 to-emerald-500 rounded-t transition-all group-hover:brightness-125"
                          style={{ height: `${Math.max(2, h)}%`, position: 'absolute', bottom: 0 }}
                        />
                        <div className="opacity-0 group-hover:opacity-100 absolute -top-8 bg-stone-800 text-white px-2 py-0.5 rounded text-[10px] font-mono whitespace-nowrap transition-opacity z-10 pointer-events-none">
                          SC={pt.sc}, P={pt.p}%
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between text-[10px] font-mono text-stone-400 mt-2">
                  <span>SC = 0</span>
                  <span>SC = 50</span>
                </div>
              </div>
              <div className="p-3.5 bg-white rounded-xl border border-stone-200 text-xs text-stone-700 leading-relaxed">
                <strong className="text-stone-900">Ablation Proof:</strong> Removing OBL increases mean error from <strong>689</strong> to <strong>7,485</strong> at d=50 (an order of magnitude), confirming its critical role in escaping deceptive traps under noise.
              </div>
            </div>
          </div>
        )}

        {activeTab === 'contraction' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-6 space-y-6">
              <h3 className="font-serif text-3xl text-stone-900">
                Periodic Bound Contraction
              </h3>
              <p className="text-stone-600 leading-relaxed">
                Every m iterations, SDAO contracts the search box [a, b] around the current global best, shrinking the exploration domain to intensify convergence while preserving the diffusion escape mechanism inside the reduced bounds.
              </p>
              <div className="bg-[#F9F8F4] p-5 rounded-xl border border-stone-200 font-mono text-center space-y-2 text-sm md:text-base text-stone-900 font-bold">
                <div>Iter 0 (Initial): [a₀, b₀]</div>
                <div>Iter m (1st contraction): [a₁, b₁] ⊂ [a₀, b₀]</div>
                <div>Iter 2m: [a₂, b₂] ⊂ [a₁, b₁]</div>
              </div>
              <p className="text-stone-600 text-sm leading-relaxed">
                The contraction factor and interval m are tuned so the bounds never collapse faster than the diffusion decay, maintaining a healthy exploration-exploitation balance.
              </p>
            </div>

            <div className="lg:col-span-6 bg-stone-50 p-8 rounded-2xl border border-stone-200 text-stone-900 space-y-6">
              <div className="bg-white p-6 rounded-xl border border-stone-200 space-y-4">
                <div className="text-xs font-bold uppercase tracking-widest text-stone-500">Contraction Visualization</div>
                {[
                  { iter: 'Iter 0 (Initial)', range: '[-100.0, +100.0]', width: '100%', color: 'bg-stone-600' },
                  { iter: 'Iter 10 (1st Contraction)', range: '[-35.4, +64.6]', width: '65%', color: 'bg-stone-500' },
                  { iter: 'Iter 20 (2nd Contraction)', range: '[-12.1, +41.8]', width: '40%', color: 'bg-nobel-gold' },
                  { iter: 'Iter 30 (3rd Contraction)', range: '[-4.2, +18.3]', width: '22%', color: 'bg-emerald-600' },
                ].map((c, i) => (
                  <div key={i} className="space-y-1.5">
                    <div className="flex justify-between text-[11px] font-mono text-stone-600">
                      <span>{c.iter}</span>
                      <span>{c.range}</span>
                    </div>
                    <div className="w-full bg-stone-200 h-4 rounded-full overflow-hidden flex items-center p-0.5">
                      <div className={`h-full ${c.color} rounded-full transition-all duration-500`} style={{ width: c.width }}></div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="p-3.5 bg-white rounded-xl border border-stone-200 text-xs text-stone-700 leading-relaxed">
                <strong className="text-stone-900">Ablation Impact:</strong> Disabling contraction increases mean error from <strong>689</strong> to <strong>1,497</strong>, showing it accelerates late-stage convergence without sacrificing early exploration.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
