/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Table, BarChart3, ShieldCheck, Cpu, Globe2, Sparkles, Award, ArrowUpRight, TrendingUp, CheckCircle2 } from 'lucide-react';
import { Reveal } from './Reveal';
import { 
  EMPIRICAL_RESULTS_D50, 
  SOCO11_RESULTS_D500, 
  ANOVA_SIGNIFICANCE_DATA, 
  WILCOXON_WINS_DATA,
  REAL_WORLD_PROBLEMS,
  BENCHMARK_FUNCTIONS
} from '../data/sdaoData';

export const BenchmarkResults: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'empirical' | 'soco11' | 'stats' | 'realworld' | 'functions'>('empirical');
  const [selectedDim, setSelectedDim] = useState<'d50' | 'd100' | 'd10' | 'd25'>('d50');

  return (
    <Reveal className="w-full my-16">
      <div className="mx-auto mb-16 max-w-3xl">
        <span className="eyebrow">Empirical validation</span>
        <h2 className="display mt-4 text-4xl text-bone md:text-5xl">Proven across thirty runs, four suites, to d=500</h2>
        <p className="mt-5 text-lg font-light leading-relaxed text-white/55">Comprehensive evaluations across 30 independent runs, 300 function evaluations (FEs), 4 benchmark categories (Standard, Stochastic, CEC 2017, Real-World), and dimensions scaling up to d = 500.</p>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap justify-center gap-2 mb-8 border-b border-white/10 pb-4">
        {[
          { id: 'empirical', label: '1. Empirical Scalability (d=10 to 100)', icon: <BarChart3 size={16} /> },
          { id: 'soco11', label: '2. SOCO11 Large-Scale (d=500)', icon: <Cpu size={16} /> },
          { id: 'stats', label: '3. Statistical Significance (ANOVA/Wilcoxon)', icon: <ShieldCheck size={16} /> },
          { id: 'realworld', label: '4. Real-World Optimization Tasks', icon: <Globe2 size={16} /> },
          { id: 'functions', label: '5. Benchmark Landscapes Catalogue', icon: <Table size={16} /> },
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

      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6 md:p-10">
        {activeTab === 'empirical' && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 pb-4 border-b border-white/10">
              <div>
                <h3 className="display text-2xl text-bone">
                  Dimensional Scalability Analysis (Tables 6–9)
                </h3>
                <p className="text-white/60 text-xs mt-1">
                  Average absolute error and standard deviation (μ ± σ) across representative algorithms. Notice how SDAO dominates as dimensionality scales beyond d ≥ 25!
                </p>
              </div>

              <div className="flex bg-white/[0.04] p-1 rounded-lg border border-white/10">
                {(['d10', 'd25', 'd50', 'd100'] as const).map(d => (
                  <button
                    key={d}
                    onClick={() => setSelectedDim(d)}
                    className={`px-3 py-1.5 rounded text-xs font-bold uppercase transition-all cursor-pointer ${
                      selectedDim === d ? 'bg-white/[0.04] text-white ' : 'text-white/60 hover:text-bone'
                    }`}
                  >
                    {d === 'd10' ? 'd = 10' : d === 'd25' ? 'd = 25' : d === 'd50' ? 'd = 50' : 'd = 100'}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-white/10 bg-white/[0.03]/50 text-[11px] font-bold text-white/45 uppercase tracking-wider">
                    <th className="py-3 px-4">Algorithm</th>
                    <th className={`py-3 px-4 transition-colors ${selectedDim === 'd10' ? 'bg-gold/25 text-bone font-black' : ''}`}>Standard (d=10)</th>
                    <th className={`py-3 px-4 transition-colors ${selectedDim === 'd25' ? 'bg-gold/25 text-bone font-black' : ''}`}>Stochastic (d=25)</th>
                    <th className={`py-3 px-4 transition-colors ${selectedDim === 'd50' ? 'bg-gold/25 text-bone font-black' : ''}`}>Real-World (d=50)</th>
                    <th className={`py-3 px-4 transition-colors ${selectedDim === 'd100' ? 'bg-gold/25 text-bone font-black' : ''}`}>CEC 2017 (d=100)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10 text-xs md:text-sm font-mono">
                  {EMPIRICAL_RESULTS_D50.map((row, idx) => (
                    <tr 
                      key={idx} 
                      className={`hover:bg-white/[0.03]/80 transition-colors ${row.isBest ? 'bg-gold/10 font-bold' : ''}`}
                    >
                      <td className="py-3.5 px-4 font-sans font-bold flex items-center gap-2">
                        {row.isBest && <Sparkles size={16} className="text-gold shrink-0" />}
                        <span className={row.isBest ? 'text-bone' : 'text-white/70'}>{row.algorithm}</span>
                      </td>
                      <td className={`py-3.5 px-4 transition-colors ${selectedDim === 'd10' ? 'bg-gold/20 font-bold text-bone' : 'text-white/60'}`}>{row.d10}</td>
                      <td className={`py-3.5 px-4 transition-colors ${selectedDim === 'd25' ? 'bg-gold/20 font-bold text-bone' : 'text-white/60'}`}>{row.d25}</td>
                      <td className={`py-3.5 px-4 transition-colors ${selectedDim === 'd50' ? 'bg-gold/20 font-bold text-bone' : 'text-white/60'}`}>{row.d50}</td>
                      <td className={`py-3.5 px-4 transition-colors ${selectedDim === 'd100' ? 'bg-gold/20 font-bold text-bone' : 'text-white/60'}`}>{row.d100}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="p-4 bg-white/[0.04] rounded-xl border border-white/10 text-xs text-white/70 leading-relaxed">
              <strong>Stochastic Resilience Highlight:</strong> While baseline methods like SHADE-ILS or AMSO achieve low errors on smooth d = 10 landscapes, they suffer orders of magnitude variance inflation under additive Gaussian noise at d = 50 and d = 100. SDAO's density repulsion and OBL preserve stable convergence regardless of noise!
            </div>
          </div>
        )}

        {activeTab === 'soco11' && (
          <div className="space-y-6">
            <div className="pb-4 border-b border-white/10">
              <h3 className="display text-2xl text-bone">
                SOCO11 Large-Scale Continuous Suite (d = 500, Table 15)
              </h3>
              <p className="text-white/60 text-xs mt-1">
                Evaluated with 25,000 function evaluations per algorithm. SOCO11 is notorious for extreme multimodality and ill-conditioning in high dimensions. Notice SDAO's dominant victories on shifted and hybrid compositions!
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {SOCO11_RESULTS_D500.map((res, idx) => (
                <div 
                  key={idx} 
                  className={`p-5 rounded-xl border transition-all ${
                    res.isSDAO 
                      ? 'bg-gold/15 border-gold shadow-md' 
                      : 'bg-white/[0.03] border-white/10 hover:border-white/15'
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-white/45">{res.function}</span>
                    <span className={`text-xs px-2 py-0.5 rounded font-bold uppercase ${
                      res.isSDAO ? 'bg-white/[0.04] text-gold' : 'bg-white/[0.06] text-white/70'
                    }`}>
                      {res.bestAlgo}
                    </span>
                  </div>
                  <div className="text-base font-mono font-bold text-bone mt-2">{res.bestValue}</div>
                  {res.isSDAO && (
                    <div className="text-[11px] text-emerald-800 font-sans mt-2 flex items-center gap-1 font-bold">
                      <CheckCircle2 size={13} className="text-emerald-700" /> SDAO Superior Accuracy & Low Variance
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="p-4 bg-white/[0.04] text-white/55 rounded-xl border border-white/10 text-xs leading-relaxed">
              <strong className="text-gold">Composite & Hybrid Dominance:</strong> SDAO attains the lowest mean objective value on four challenging functions: <strong>Bohachevsky (shifted), Hybrid 14, Hybrid 18, and Hybrid 19</strong>. This proves its capability to handle non-separable landscapes combining multimodality with structural irregularities!
            </div>
          </div>
        )}

        {activeTab === 'stats' && (
          <div className="space-y-8">
            <div>
              <h3 className="display text-2xl text-bone mb-2">
                Statistical Protocol & Significance Suite
              </h3>
              <p className="text-white/60 text-xs leading-relaxed">
                To guarantee statistical rigor, all experiments underwent 30 independent runs per function. A one-way ANOVA test combined with Tukey's HSD and non-parametric Wilcoxon signed-rank test confirmed that SDAO's gains are statistically significant (p &lt; 0.05).
              </p>
            </div>

            {/* ANOVA Table */}
            <div className="space-y-3">
              <h4 className="display text-lg font-bold text-bone flex items-center gap-2">
                <span>One-Way ANOVA P-Values across Categories (Table 10)</span>
                <span className="text-[10px] bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded uppercase font-sans">Statistically Significant</span>
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse border border-white/10">
                  <thead>
                    <tr className="bg-white/[0.04] text-[11px] font-bold text-white/60 uppercase">
                      <th className="py-2.5 px-4">Dimension</th>
                      <th className="py-2.5 px-4">Standard Benchmarks</th>
                      <th className="py-2.5 px-4">Stochastic Benchmarks</th>
                      <th className="py-2.5 px-4">Real-World Tasks</th>
                      <th className="py-2.5 px-4">CEC 2017 Suite</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/10 text-xs font-mono">
                    {ANOVA_SIGNIFICANCE_DATA.map((row, idx) => (
                      <tr key={idx} className="hover:bg-white/[0.03]">
                        <td className="py-3 px-4 font-sans font-bold text-bone">{row.dim}</td>
                        <td className="py-3 px-4 text-emerald-700 font-bold">{row.standard}</td>
                        <td className="py-3 px-4 text-emerald-700 font-bold">{row.stochastic}</td>
                        <td className="py-3 px-4 text-emerald-700 font-bold">{row.realWorld}</td>
                        <td className="py-3 px-4 text-emerald-700 font-bold">{row.cec}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Wilcoxon Table */}
            <div className="space-y-3">
              <h4 className="display text-lg font-bold text-bone flex items-center gap-2">
                <span>Wilcoxon Signed-Rank Test: SDAO Significant Wins at d=50 (Table 12)</span>
                <span className="text-[10px] bg-white/[0.04] text-gold px-2 py-0.5 rounded uppercase font-sans">38/38 CEC Dominance</span>
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse border border-white/10">
                  <thead>
                    <tr className="bg-white/[0.04] text-[11px] font-bold text-white/60 uppercase">
                      <th className="py-2.5 px-4">Compared Algorithm</th>
                      <th className="py-2.5 px-4">Standard (14 functions)</th>
                      <th className="py-2.5 px-4">Stochastic (12 functions)</th>
                      <th className="py-2.5 px-4">Real-World (6 tasks)</th>
                      <th className="py-2.5 px-4">CEC 2017 (38 functions)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/10 text-xs font-mono">
                    {WILCOXON_WINS_DATA.map((row, idx) => (
                      <tr key={idx} className="hover:bg-white/[0.03]">
                        <td className="py-3 px-4 font-sans font-bold text-bone">{row.algorithm}</td>
                        <td className="py-3 px-4 text-bone font-bold">{row.standard}</td>
                        <td className="py-3 px-4 text-emerald-700 font-bold">{row.stochastic}</td>
                        <td className="py-3 px-4 text-bone">{row.realWorld}</td>
                        <td className="py-3 px-4 text-emerald-700 font-bold bg-emerald-50/50">{row.cec}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'realworld' && (
          <div className="space-y-6">
            <div className="pb-4 border-b border-white/10">
              <h3 className="display text-2xl text-bone">
                Real-World Decision-Making Challenges (Table 3)
              </h3>
              <p className="text-white/60 text-xs mt-1">
                Unlike synthetic mathematical functions, real-world problems feature uncertain conditions, competing objectives, and complex dependencies. SDAO was benchmarked across 6 rigorous industry applications.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {REAL_WORLD_PROBLEMS.map((prob, idx) => (
                <div key={idx} className="bg-white/[0.03] p-6 rounded-xl border border-white/10 flex flex-col justify-between hover:border-gold transition-all">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-[10px] font-bold uppercase tracking-widest bg-white/[0.06] text-white/70 px-2 py-0.5 rounded">
                        {prob.category}
                      </span>
                      <span className="text-xs text-white/35 font-mono">{prob.reference}</span>
                    </div>
                    <h4 className="display text-xl text-bone mb-2">{prob.title}</h4>
                    <p className="text-xs text-white/60 mb-4"><strong>Application:</strong> {prob.application}</p>
                  </div>
                  <div className="p-3 bg-white/[0.03] rounded-lg border border-white/10 text-xs text-bone">
                    <span className="text-gold font-bold">Objective:</span> {prob.objective}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'functions' && (
          <div className="space-y-6">
            <div className="pb-4 border-b border-white/10">
              <h3 className="display text-2xl text-bone">
                Catalogue of Benchmark Functions (Table 1 & Table 2)
              </h3>
              <p className="text-white/60 text-xs mt-1">
                A diverse mathematical testbed varying in modality, smoothness, and deceptiveness to rigorously stress-test exploration vs. exploitation.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {BENCHMARK_FUNCTIONS.map((fn, idx) => (
                <div key={idx} className="bg-white/[0.03] p-5 rounded-xl border border-white/10">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="display text-lg font-bold text-bone">{fn.name}</h4>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                      fn.category === 'stochastic' ? 'bg-amber-100 text-amber-800' : 'bg-white/[0.06] text-white/70'
                    }`}>
                      {fn.modality}
                    </span>
                  </div>
                  <div className="text-xs text-white/45 mb-3 font-mono">{fn.searchSpace}</div>
                  <p className="text-xs text-white/70 mb-3">{fn.characteristics}</p>
                  {fn.equationDisplay && (
                    <div className="bg-white/[0.03] p-2.5 rounded border border-white/10 font-mono text-[11px] text-bone text-center">
                      {fn.equationDisplay}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Reveal>
  );
};
