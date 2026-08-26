/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ShieldAlert, CheckCircle2, Sliders, BarChart2, Zap, Minimize2, Layers, RefreshCcw } from 'lucide-react';
import { Reveal } from './Reveal';
import { ABLATION_STUDY_DATA } from '../data/sdaoData';

export const AblationStudy: React.FC = () => {
  const [enableOBL, setEnableOBL] = useState<boolean>(true);
  const [enableContraction, setEnableContraction] = useState<boolean>(true);
  const [enableDiffusion, setEnableDiffusion] = useState<boolean>(true);
  const [selectedDegreeM, setSelectedDegreeM] = useState<number>(10);

  // Estimate dynamic error based on toggles
  let estimatedError = 689.79;
  let estimatedStd = 420.94;
  let statusColor = 'text-emerald-700 bg-emerald-50 border-emerald-200';
  let statusText = 'Optimal Performance (Full SDAO Engine)';

  if (!enableOBL && !enableContraction) {
    estimatedError = 12450.80;
    estimatedStd = 8900.50;
    statusColor = 'text-red-700 bg-red-50 border-red-200';
    statusText = 'Critical Degradation: Severe Premature Stagnation';
  } else if (!enableOBL) {
    estimatedError = 7485.99;
    estimatedStd = 6051.20;
    statusColor = 'text-red-700 bg-red-50 border-red-200';
    statusText = 'Severe Loss of Diversity: Cannot Escape Deceptive Optima';
  } else if (!enableContraction) {
    estimatedError = 1497.52;
    estimatedStd = 191.38;
    statusColor = 'text-amber-700 bg-amber-50 border-amber-200';
    statusText = 'Suboptimal Refinement: Slower Late-Stage Convergence';
  }

  if (!enableDiffusion) {
    estimatedError = estimatedError * 2.8;
  }

  return (
    <Reveal className="w-full my-16">
      <div className="mx-auto mb-16 max-w-3xl">
        <span className="eyebrow">Ablation & sensitivity</span>
        <h2 className="display mt-4 text-4xl text-bone md:text-5xl">Each component earns its place</h2>
        <p className="mt-5 text-lg font-light leading-relaxed text-white/55">Dissecting the functional contribution of Opposition-Based Learning (OBL), Periodic Bound Contraction, and k-d tree neighborhood degree m under noisy conditions (d=50).</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Interactive Engine Switcher */}
        <div className="lg:col-span-5 bg-white/[0.03] p-8 rounded-2xl border border-white/10 shadow-xl space-y-6 flex flex-col justify-between">
          <div>
            <h3 className="display text-2xl text-bone mb-2">
              Interactive Component Sandbox
            </h3>
            <p className="text-white/60 text-xs leading-relaxed mb-6">
              Toggle core SDAO modules to simulate what happens when adaptive mechanisms are disabled (reproducing Table 13 ablation experiments).
            </p>

            <div className="space-y-4">
              <div 
                onClick={() => setEnableOBL(!enableOBL)}
                className={`p-4 rounded-xl border flex items-center justify-between cursor-pointer transition-all ${
                  enableOBL ? 'bg-white/[0.04] text-white border-gold/60' : 'bg-white/[0.03] text-white/60 border-white/10 hover:border-white/15'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Zap size={20} className={enableOBL ? 'text-gold' : 'text-white/35'} />
                  <div>
                    <div className="font-bold text-sm">Opposition-Based Learning (OBL)</div>
                    <div className="text-[11px] opacity-80">Catapults stagnant particles across search bounds</div>
                  </div>
                </div>
                <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${enableOBL ? 'bg-gold text-bone' : 'bg-white/15 text-white/60'}`}>
                  {enableOBL ? '✓' : ''}
                </div>
              </div>

              <div 
                onClick={() => setEnableContraction(!enableContraction)}
                className={`p-4 rounded-xl border flex items-center justify-between cursor-pointer transition-all ${
                  enableContraction ? 'bg-white/[0.04] text-white border-gold/60' : 'bg-white/[0.03] text-white/60 border-white/10 hover:border-white/15'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Minimize2 size={20} className={enableContraction ? 'text-gold' : 'text-white/35'} />
                  <div>
                    <div className="font-bold text-sm">Periodic Bound Contraction</div>
                    <div className="text-[11px] opacity-80">Shrinks search box by factor γ every 10 steps</div>
                  </div>
                </div>
                <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${enableContraction ? 'bg-gold text-bone' : 'bg-white/15 text-white/60'}`}>
                  {enableContraction ? '✓' : ''}
                </div>
              </div>

              <div 
                onClick={() => setEnableDiffusion(!enableDiffusion)}
                className={`p-4 rounded-xl border flex items-center justify-between cursor-pointer transition-all ${
                  enableDiffusion ? 'bg-white/[0.04] text-white border-gold/60' : 'bg-white/[0.03] text-white/60 border-white/10 hover:border-white/15'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Layers size={20} className={enableDiffusion ? 'text-gold' : 'text-white/35'} />
                  <div>
                    <div className="font-bold text-sm">Density Diffusion (D_FL)</div>
                    <div className="text-[11px] opacity-80">Fick's 2nd Law repulsion from overcrowded clusters</div>
                  </div>
                </div>
                <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${enableDiffusion ? 'bg-gold text-bone' : 'bg-white/15 text-white/60'}`}>
                  {enableDiffusion ? '✓' : ''}
                </div>
              </div>
            </div>
          </div>

          <div className={`p-4 rounded-xl border ${statusColor} space-y-1`}>
            <div className="text-xs font-bold uppercase tracking-wider">Simulated Outcome at d=50</div>
            <div className="text-xl font-mono font-bold">Error: {Math.round(estimatedError).toLocaleString()} ± {Math.round(estimatedStd).toLocaleString()}</div>
            <div className="text-xs">{statusText}</div>
          </div>
        </div>

        {/* Table 13 Summary Cards & Figure 3 Sensitivity */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white/[0.04] text-white p-8 rounded-2xl border border-white/10 shadow-xl space-y-6">
            <div className="flex justify-between items-center">
              <div>
                <span className="text-xs font-bold tracking-widest text-gold uppercase">TABLE 13 EMPIRICAL DATA</span>
                <h3 className="display text-2xl text-white mt-1">Stochastic Benchmark (d=50, 30 runs)</h3>
              </div>
              <BarChart2 size={24} className="text-gold" />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {ABLATION_STUDY_DATA.map((item, idx) => (
                <div key={idx} className="bg-white/[0.05] p-4 rounded-xl border border-white/10 flex flex-col justify-between">
                  <div>
                    <div className="text-xs font-bold text-gold uppercase mb-1">{item.variant}</div>
                    <div className="text-2xl font-mono font-bold text-white my-2">{item.meanError.toLocaleString()}</div>
                    <div className="text-[11px] text-white/35 font-mono">± {item.stdDev.toLocaleString()} σ</div>
                  </div>
                  <div className="text-[11px] text-white/35 mt-3 pt-2 border-t border-white/10/60">{item.desc}</div>
                </div>
              ))}
            </div>

            <div className="p-4 bg-white/[0.05]/80 rounded-xl border border-white/10 text-xs text-white/55 leading-relaxed">
              <strong className="text-gold">Key Takeaway from Paper:</strong> Disabling OBL leads to an order of magnitude degradation (error skyrocketing from 689 to 7,485), proving its critical role in escaping deceptive local traps under Gaussian noise.
            </div>
          </div>

          {/* Figure 3 Sensitivity Analysis */}
          <div className="bg-white/[0.03] p-8 rounded-2xl border border-white/10 shadow-xl space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <h4 className="display text-xl text-bone">Neighborhood Radius Sensitivity (Figure 3)</h4>
                <p className="text-white/60 text-xs mt-0.5">Target degree m ∈ [5, 10, 20, 40] across n ∈ [10, 50, 100]</p>
              </div>
              <span className="font-mono text-sm font-bold bg-white/[0.04] text-bone px-3 py-1 rounded-lg border border-white/10">
                m = {selectedDegreeM}
              </span>
            </div>

            <div className="flex gap-2">
              {[5, 10, 20, 40].map(m => (
                <button
                  key={m}
                  onClick={() => setSelectedDegreeM(m)}
                  className={`flex-1 py-2 rounded-lg text-xs font-bold uppercase transition-all cursor-pointer ${
                    selectedDegreeM === m ? 'bg-white/[0.04] text-white ' : 'bg-white/[0.04] text-white/60 hover:bg-white/[0.06]'
                  }`}
                >
                  Degree m={m}
                </button>
              ))}
            </div>

            <div className="p-3.5 bg-white/[0.04] rounded-xl border border-white/10 text-xs text-white/70 leading-relaxed">
              {selectedDegreeM === 5 && <span><strong>m = 5 (Too Small):</strong> Very small m increases stochastic variance in the density-based direction because local centroids become overly noisy and erratic.</span>}
              {(selectedDegreeM === 10 || selectedDegreeM === 20) && <span><strong>m = {selectedDegreeM} (Optimal Plateau):</strong> Results exhibit a broad, highly stable performance plateau for m ∈ [10, 20]. This keeps neighborhood query costs O(N log N) low while providing accurate repulsion centroids!</span>}
              {selectedDegreeM === 40 && <span><strong>m = 40 (Too Large):</strong> Very large m oversmooths local structure and slows adaptation by averaging across unrelated search basins.</span>}
            </div>
          </div>
        </div>
      </div>
    </Reveal>
  );
};
