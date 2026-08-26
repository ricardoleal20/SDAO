/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Table, BarChart3, ShieldCheck, Cpu, Globe2, Sparkles, Award, ArrowUpRight, TrendingUp, CheckCircle2 } from 'lucide-react';
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
    <div className="w-full my-16">
      <div className="text-center max-w-3xl mx-auto mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-stone-100 text-stone-700 text-xs font-bold tracking-widest uppercase rounded-full mb-4 border border-stone-200">
          <Award size={14} className="text-nobel-gold" /> SECTION 4: EMPIRICAL BENCHMARKING
        </div>
        <h2 className="font-serif text-4xl md:text-5xl text-stone-900 mb-4">
          Experimental Validation & Results
        </h2>
        <p className="text-lg text-stone-600 font-light leading-relaxed">
          Comprehensive evaluations across 30 independent runs, 300 function evaluations (FEs), 4 benchmark categories (Standard, Stochastic, CEC 2017, Real-World), and dimensions scaling up to d = 500.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap justify-center gap-2 mb-8 border-b border-stone-200 pb-4">
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
                ? 'bg-stone-900 text-white shadow-md scale-105'
                : 'bg-white hover:bg-stone-100 text-stone-600 border border-stone-200'
            }`}
          >
            <span className={activeTab === tab.id ? 'text-nobel-gold' : 'text-stone-400'}>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      <div className="bg-white rounded-2xl border border-stone-200 shadow-xl p-6 md:p-10">
        {activeTab === 'empirical' && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 pb-4 border-b border-stone-200">
              <div>
                <h3 className="font-serif text-2xl text-stone-900">
                  Dimensional Scalability Analysis (Tables 6–9)
                </h3>
                <p className="text-stone-600 text-xs mt-1">
                  Average absolute error and standard deviation (μ ± σ) across representative algorithms. Notice how SDAO dominates as dimensionality scales beyond d ≥ 25!
                </p>
              </div>

              <div className="flex bg-stone-100 p-1 rounded-lg border border-stone-200">
                {(['d10', 'd25', 'd50', 'd100'] as const).map(d => (
                  <button
                    key={d}
                    onClick={() => setSelectedDim(d)}
                    className={`px-3 py-1.5 rounded text-xs font-bold uppercase transition-all cursor-pointer ${
                      selectedDim === d ? 'bg-stone-900 text-white shadow-xs' : 'text-stone-600 hover:text-stone-900'
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
                  <tr className="border-b border-stone-200 bg-stone-50/50 text-[11px] font-bold text-stone-500 uppercase tracking-wider">
                    <th className="py-3 px-4">Algorithm</th>
                    <th className={`py-3 px-4 transition-colors ${selectedDim === 'd10' ? 'bg-nobel-gold/25 text-stone-900 font-black' : ''}`}>Standard (d=10)</th>
                    <th className={`py-3 px-4 transition-colors ${selectedDim === 'd25' ? 'bg-nobel-gold/25 text-stone-900 font-black' : ''}`}>Stochastic (d=25)</th>
                    <th className={`py-3 px-4 transition-colors ${selectedDim === 'd50' ? 'bg-nobel-gold/25 text-stone-900 font-black' : ''}`}>Real-World (d=50)</th>
                    <th className={`py-3 px-4 transition-colors ${selectedDim === 'd100' ? 'bg-nobel-gold/25 text-stone-900 font-black' : ''}`}>CEC 2017 (d=100)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100 text-xs md:text-sm font-mono">
                  {EMPIRICAL_RESULTS_D50.map((row, idx) => {
                    const dimVal = selectedDim === 'd10' ? row.d10 : selectedDim === 'd25' ? row.d25 : selectedDim === 'd50' ? row.d50 : row.d100 || 'N/A';
                    return (
                      <tr
                        key={idx}
                        className={`hover:bg-stone-50 transition-colors ${row.isBest ? 'bg-emerald-50/40 font-bold' : ''}`}
                      >
                        <td className="py-3 px-4 font-sans font-bold text-stone-900">
                          {row.isBest && <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-2"></span>}
                          {row.algorithm}
                        </td>
                        <td className={`py-3 px-4 ${selectedDim === 'd10' ? 'bg-nobel-gold/10 font-bold text-stone-900' : 'text-stone-600'}`}>{row.d10}</td>
                        <td className={`py-3 px-4 ${selectedDim === 'd25' ? 'bg-nobel-gold/10 font-bold text-stone-900' : 'text-stone-600'}`}>{row.d25}</td>
                        <td className={`py-3 px-4 ${selectedDim === 'd50' ? 'bg-nobel-gold/10 font-bold text-stone-900' : 'text-stone-600'}`}>{row.d50}</td>
                        <td className={`py-3 px-4 ${selectedDim === 'd100' ? 'bg-nobel-gold/10 font-bold text-stone-900' : 'text-stone-600'}`}>{row.d100 || 'N/A'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="p-4 bg-[#F9F8F4] rounded-xl border border-stone-200 text-xs text-stone-700 leading-relaxed">
              <strong className="text-nobel-gold">Key Insight:</strong> While most metaheuristics diverge exponentially (e.g., AMSO reaches ~10⁸¹ at d=100), SDAO maintains bounded, competitive error — a direct consequence of its dimension-aware diffusion cap.
              <br />Current focus column: <strong>{selectedDim === 'd10' ? 'Standard functions' : selectedDim === 'd25' ? 'Stochastic functions' : selectedDim === 'd50' ? 'Real-world tasks' : 'CEC 2017'}</strong>.
            </div>
          </div>
        )}

        {activeTab === 'soco11' && (
          <div className="space-y-6">
            <div className="pb-4 border-b border-stone-200">
              <h3 className="font-serif text-2xl text-stone-900">
                SOCO11 Large-Scale Benchmark (d=500, Table 15)
              </h3>
              <p className="text-stone-600 text-xs mt-1">
                Evaluated with 25,000 function evaluations per algorithm. SOCO11 is notorious for extreme multimodality and ill-conditioning in high dimensions. Notice SDAO's dominant victories on shifted and hybrid compositions!
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {SOCO11_RESULTS_D500.map((res, idx) => (
                <div
                  key={idx}
                  className={`p-5 rounded-xl border transition-all ${
                    res.isSDAO
                      ? 'bg-emerald-50 border-emerald-300 shadow-md'
                      : 'bg-stone-50 border-stone-200 hover:border-stone-300'
                  }`}
                >
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-serif text-lg font-bold text-stone-900">{res.function}</h4>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                      res.isSDAO ? 'bg-emerald-600 text-white' : 'bg-stone-200 text-stone-700'
                    }`}>
                      {res.bestAlgo}
                    </span>
                  </div>
                  <div className="text-xs font-mono text-stone-600">{res.bestValue}</div>
                  {res.isSDAO && (
                    <div className="text-[10px] font-bold text-emerald-700 mt-1 flex items-center gap-1">
                      <CheckCircle2 size={12} /> SDAO Wins This Function
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="p-4 bg-[#F9F8F4] rounded-xl border border-stone-200 text-xs text-stone-700 leading-relaxed">
              <strong className="text-nobel-gold">At d=500:</strong> SDAO wins on Bohachevsky (shifted) and 3 of the most challenging hybrid composition functions (Hybrid 14, 18, 19), demonstrating its robustness where traditional swarm intelligence typically collapses.
            </div>
          </div>
        )}

        {activeTab === 'stats' && (
          <div className="space-y-8">
            <div className="pb-4 border-b border-stone-200">
              <h3 className="font-serif text-2xl text-stone-900">
                Statistical Significance: ANOVA & Wilcoxon (Tables 10 & 12)
              </h3>
              <p className="text-stone-600 text-xs mt-1">
                Rigorous non-parametric testing confirms SDAO's superiority is statistically significant, not due to chance.
              </p>
            </div>

            <div>
              <h4 className="font-serif text-xl text-stone-900 mb-3">One-Way ANOVA p-values (Table 10)</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-stone-200 bg-stone-50 text-[11px] font-bold text-stone-500 uppercase tracking-wider">
                      <th className="py-2.5 px-4">Dimension</th>
                      <th className="py-2.5 px-4">Standard (14 funcs)</th>
                      <th className="py-2.5 px-4">Stochastic (12 funcs)</th>
                      <th className="py-2.5 px-4">Real-World (6 tasks)</th>
                      <th className="py-2.5 px-4">CEC 2017 (38 funcs)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-stone-100 font-mono">
                    {ANOVA_SIGNIFICANCE_DATA.map((row, idx) => (
                      <tr key={idx} className="hover:bg-stone-50">
                        <td className="py-3 px-4 font-sans font-bold text-stone-900">{row.dim}</td>
                        <td className="py-3 px-4 text-stone-700">{row.standard}</td>
                        <td className="py-3 px-4 text-emerald-700 font-bold">{row.stochastic}</td>
                        <td className="py-3 px-4 text-emerald-700 font-bold">{row.realWorld}</td>
                        <td className="py-3 px-4 text-stone-700">{row.cec}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] text-stone-500 mt-2">
                p-values below 0.05 indicate statistically significant differences. SDAO shows extreme significance (p &lt; 10⁻⁴⁰) on stochastic and real-world suites.
              </p>
            </div>

            <div>
              <h4 className="font-serif text-xl text-stone-900 mb-3">Wilcoxon Rank-Sum Wins for SDAO (Table 12 at d=50)</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-stone-200 bg-stone-50 text-[11px] font-bold text-stone-500 uppercase tracking-wider">
                      <th className="py-2.5 px-4">Compared Algorithm</th>
                      <th className="py-2.5 px-4">Standard (14 functions)</th>
                      <th className="py-2.5 px-4">Stochastic (12 functions)</th>
                      <th className="py-2.5 px-4">Real-World (6 tasks)</th>
                      <th className="py-2.5 px-4">CEC 2017 (38 functions)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-stone-100 text-xs font-mono">
                    {WILCOXON_WINS_DATA.map((row, idx) => (
                      <tr key={idx} className="hover:bg-stone-50">
                        <td className="py-3 px-4 font-sans font-bold text-stone-900">{row.algorithm}</td>
                        <td className="py-3 px-4 text-stone-800 font-bold">{row.standard}</td>
                        <td className="py-3 px-4 text-emerald-700 font-bold">{row.stochastic}</td>
                        <td className="py-3 px-4 text-stone-800">{row.realWorld}</td>
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
            <div className="pb-4 border-b border-stone-200">
              <h3 className="font-serif text-2xl text-stone-900">
                Real-World Decision-Making Challenges (Table 3)
              </h3>
              <p className="text-stone-600 text-xs mt-1">
                Unlike synthetic mathematical functions, real-world problems feature uncertain conditions, competing objectives, and complex dependencies. SDAO was benchmarked across 6 rigorous industry applications.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {REAL_WORLD_PROBLEMS.map((prob, idx) => (
                <div key={idx} className="bg-stone-50 p-6 rounded-xl border border-stone-200 flex flex-col justify-between hover:border-nobel-gold transition-all">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-[10px] font-bold uppercase tracking-widest bg-stone-200 text-stone-700 px-2 py-0.5 rounded">
                        {prob.category}
                      </span>
                      <span className="text-xs text-stone-400 font-mono">{prob.reference}</span>
                    </div>
                    <h4 className="font-serif text-xl text-stone-900 mb-2">{prob.title}</h4>
                    <p className="text-xs text-stone-600 mb-4"><strong>Application:</strong> {prob.application}</p>
                  </div>
                  <div className="p-3 bg-white rounded-lg border border-stone-200 text-xs text-stone-800">
                    <span className="text-nobel-gold font-bold">Objective:</span> {prob.objective}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'functions' && (
          <div className="space-y-6">
            <div className="pb-4 border-b border-stone-200">
              <h3 className="font-serif text-2xl text-stone-900">
                Catalogue of Benchmark Functions (Table 1 & Table 2)
              </h3>
              <p className="text-stone-600 text-xs mt-1">
                A diverse mathematical testbed varying in modality, smoothness, and deceptiveness to rigorously stress-test exploration vs. exploitation.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {BENCHMARK_FUNCTIONS.map((fn, idx) => (
                <div key={idx} className="bg-stone-50 p-5 rounded-xl border border-stone-200">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-serif text-lg font-bold text-stone-900">{fn.name}</h4>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                      fn.category === 'stochastic' ? 'bg-amber-100 text-amber-800' : 'bg-stone-200 text-stone-700'
                    }`}>
                      {fn.modality}
                    </span>
                  </div>
                  <div className="text-xs text-stone-500 mb-3 font-mono">{fn.searchSpace}</div>
                  <p className="text-xs text-stone-700 mb-3">{fn.characteristics}</p>
                  {fn.equationDisplay && (
                    <div className="bg-white p-2.5 rounded border border-stone-200 font-mono text-[11px] text-stone-800 text-center">
                      {fn.equationDisplay}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
