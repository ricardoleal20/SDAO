/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ShieldAlert, CheckCircle2, Sliders, BarChart2, Zap, Minimize2, Layers, RefreshCcw } from 'lucide-react';
import { ABLATION_STUDY_DATA } from '../data/sdaoData';

export const AblationStudy: React.FC = () => {
  const [enableOBL, setEnableOBL] = useState(true);
  const [enableContraction, setEnableContraction] = useState(true);
  const [enableDiffusion, setEnableDiffusion] = useState(true);
  const [selectedDegreeM, setSelectedDegreeM] = useState(10);

  // Estimate error based on toggles (illustrative model from ablation data)
  const baseline = 689.79;
  let estimatedError = baseline;
  let estimatedStd = 420.94;

  if (!enableOBL) {
    estimatedError = 7485.99;
    estimatedStd = 6051.20;
  } else if (!enableContraction) {
    estimatedError = 1497.52;
    estimatedStd = 191.38;
  }
  if (!enableDiffusion && enableOBL && enableContraction) {
    estimatedError *= 3.2;
    estimatedStd *= 2.1;
  }

  let statusColor = 'bg-emerald-50 border-emerald-300 text-emerald-900';
  let statusText = 'Full SDAO configuration — optimal performance achieved.';
  if (estimatedError > 1000 && estimatedError < 5000) {
    statusColor = 'bg-amber-50 border-amber-300 text-amber-900';
    statusText = 'Degraded: missing a key adaptive component.';
  } else if (estimatedError >= 5000) {
    statusColor = 'bg-red-50 border-red-300 text-red-900';
    statusText = 'Severe degradation: OBL removal causes order-of-magnitude error increase.';
  }

  return (
    <div className="w-full my-16">
      <div className="text-center max-w-3xl mx-auto mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-stone-100 text-stone-700 text-xs font-bold tracking-widest uppercase rounded-full mb-4 border border-stone-200">
          <Sliders size={14} className="text-nobel-gold" /> SECTION 4.6 & 4.7: COMPONENT IMPACT
        </div>
        <h2 className="font-serif text-4xl md:text-5xl text-stone-900 mb-4">
          Ablation Study & Sensitivity Analysis
        </h2>
        <p className="text-lg text-stone-600 font-light leading-relaxed">
          Dissecting the functional contribution of Opposition-Based Learning (OBL), Periodic Bound Contraction, and k-d tree neighborhood degree m under noisy conditions (d=50).
        </p>
      </div>

      <div className="bg-white rounded-2xl border border-stone-200 shadow-xl p-8 md:p-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Interactive Ablation Toggles */}
          <div className="lg:col-span-5 space-y-6">
            <div>
              <h3 className="font-serif text-2xl text-stone-900 mb-2">Component Toggles</h3>
              <p className="text-stone-600 text-xs leading-relaxed mb-6">
                Toggle each SDAO component on/off to estimate its isolated impact on optimization error at d=50 (stochastic benchmark). Values are modeled from Table 13 ablation data.
              </p>
            </div>

            {/* OBL Toggle */}
            <div className={`p-4 rounded-xl border-2 transition-all ${enableOBL ? 'bg-emerald-50 border-emerald-300' : 'bg-stone-50 border-stone-200 opacity-70'}`}>
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <Zap size={20} className={enableOBL ? 'text-emerald-600' : 'text-stone-400'} />
                  <div>
                    <div className="font-bold text-stone-900 text-sm">Opposition-Based Learning</div>
                    <div className="text-[11px] text-stone-500">Reflects stagnant particles across the search space</div>
                  </div>
                </div>
                <button
                  onClick={() => setEnableOBL(!enableOBL)}
                  className={`px-4 py-2 rounded-full text-xs font-bold uppercase tracking-wider transition-all cursor-pointer ${
                    enableOBL ? 'bg-emerald-600 text-white border-emerald-700 shadow-md' : 'bg-stone-200 text-stone-600 border-stone-300'
                  }`}
                >
                  {enableOBL ? 'ON' : 'OFF'}
                </button>
              </div>
              <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${enableOBL ? 'bg-emerald-600 text-white' : 'bg-stone-300 text-stone-600'}`}>
                {enableOBL ? '✓' : ''}
              </div>
            </div>

            {/* Contraction Toggle */}
            <div className={`p-4 rounded-xl border-2 transition-all ${enableContraction ? 'bg-emerald-50 border-emerald-300' : 'bg-stone-50 border-stone-200 opacity-70'}`}>
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <Minimize2 size={20} className={enableContraction ? 'text-emerald-600' : 'text-stone-400'} />
                  <div>
                    <div className="font-bold text-stone-900 text-sm">Periodic Bound Contraction</div>
                    <div className="text-[11px] text-stone-500">Shrinks search box around global best every m iterations</div>
                  </div>
                </div>
                <button
                  onClick={() => setEnableContraction(!enableContraction)}
                  className={`px-4 py-2 rounded-full text-xs font-bold uppercase tracking-wider transition-all cursor-pointer ${
                    enableContraction ? 'bg-emerald-600 text-white border-emerald-700 shadow-md' : 'bg-stone-200 text-stone-600 border-stone-300'
                  }`}
                >
                  {enableContraction ? 'ON' : 'OFF'}
                </button>
              </div>
              <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${enableContraction ? 'bg-emerald-600 text-white' : 'bg-stone-300 text-stone-600'}`}>
                {enableContraction ? '✓' : ''}
              </div>
            </div>

            {/* Diffusion Toggle */}
            <div className={`p-4 rounded-xl border-2 transition-all ${enableDiffusion ? 'bg-emerald-50 border-emerald-300' : 'bg-stone-50 border-stone-200 opacity-70'}`}>
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <Layers size={20} className={enableDiffusion ? 'text-emerald-600' : 'text-stone-400'} />
                  <div>
                    <div className="font-bold text-stone-900 text-sm">Density Diffusion (D_FL)</div>
                    <div className="text-[11px] text-stone-500">Fick's 2nd Law repulsion from overcrowded clusters</div>
                  </div>
                </div>
                <button
                  onClick={() => setEnableDiffusion(!enableDiffusion)}
                  className={`px-4 py-2 rounded-full text-xs font-bold uppercase tracking-wider transition-all cursor-pointer ${
                    enableDiffusion ? 'bg-emerald-600 text-white border-emerald-700 shadow-md' : 'bg-stone-200 text-stone-600 border-stone-300'
                  }`}
                >
                  {enableDiffusion ? 'ON' : 'OFF'}
                </button>
              </div>
              <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${enableDiffusion ? 'bg-emerald-600 text-white' : 'bg-stone-300 text-stone-600'}`}>
                {enableDiffusion ? '✓' : ''}
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
            <div className="bg-stone-900 text-white p-8 rounded-2xl border border-stone-800 shadow-xl space-y-6">
              <div className="flex justify-between items-center">
                <div>
                  <span className="text-xs font-bold tracking-widest text-nobel-gold uppercase">TABLE 13 EMPIRICAL DATA</span>
                  <h3 className="font-serif text-2xl text-white mt-1">Stochastic Benchmark (d=50, 30 runs)</h3>
                </div>
                <BarChart2 size={24} className="text-nobel-gold" />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {ABLATION_STUDY_DATA.map((item, idx) => (
                  <div key={idx} className="bg-stone-800 p-4 rounded-xl border border-stone-700 flex flex-col justify-between">
                    <div>
                      <div className="text-xs font-bold text-nobel-gold uppercase mb-1">{item.variant}</div>
                      <div className="text-2xl font-mono font-bold text-white my-2">{item.meanError.toLocaleString()}</div>
                      <div className="text-[11px] text-stone-400 font-mono">± {item.stdDev.toLocaleString()} σ</div>
                    </div>
                    <div className="text-[11px] text-stone-400 mt-3 pt-2 border-t border-stone-700/60">{item.desc}</div>
                  </div>
                ))}
              </div>

              <div className="p-4 bg-stone-800/80 rounded-xl border border-stone-700 text-xs text-stone-300 leading-relaxed">
                <strong className="text-nobel-gold">Key Takeaway from Paper:</strong> Disabling OBL leads to an order of magnitude degradation (error skyrocketing from 689 to 7,485), proving its critical role in escaping deceptive local traps under Gaussian noise.
              </div>
            </div>

            {/* Figure 3 Sensitivity Analysis */}
            <div className="bg-white p-8 rounded-2xl border border-stone-200 shadow-xl space-y-4">
              <div className="flex justify-between items-center">
                <div>
                  <h4 className="font-serif text-xl text-stone-900">Neighborhood Radius Sensitivity (Figure 3)</h4>
                  <p className="text-stone-600 text-xs mt-0.5">Target degree m ∈ [5, 10, 20, 40] across n ∈ [10, 50, 100]</p>
                </div>
                <span className="font-mono text-sm font-bold bg-stone-100 text-stone-800 px-3 py-1 rounded-lg border border-stone-200">
                  m = {selectedDegreeM}
                </span>
              </div>

              <div className="flex gap-2">
                {[5, 10, 20, 40].map(m => (
                  <button
                    key={m}
                    onClick={() => setSelectedDegreeM(m)}
                    className={`flex-1 py-2 rounded-lg text-xs font-bold uppercase transition-all cursor-pointer ${
                      selectedDegreeM === m ? 'bg-stone-900 text-white shadow-xs' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                  >
                    Degree m={m}
                  </button>
                ))}
              </div>

              <div className="p-3.5 bg-[#F9F8F4] rounded-xl border border-stone-200 text-xs text-stone-700 leading-relaxed">
                {selectedDegreeM === 5 && <span><strong>m = 5 (Too Small):</strong> Very small m increases stochastic variance in the density-based direction because local centroids become overly noisy and erratic.</span>}
                {(selectedDegreeM === 10 || selectedDegreeM === 20) && <span><strong>m = {selectedDegreeM} (Optimal Plateau):</strong> Results exhibit a broad, highly stable performance plateau for m ∈ [10, 20]. This keeps neighborhood query costs O(N log N) low while providing accurate repulsion centroids!</span>}
                {selectedDegreeM === 40 && <span><strong>m = 40 (Too Large):</strong> Very large m oversmooths local structure and slows adaptation by averaging across unrelated search basins.</span>}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
