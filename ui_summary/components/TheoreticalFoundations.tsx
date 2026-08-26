/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { BookOpen, Cpu, ShieldCheck, Compass, GitBranch, Layers, CheckCircle2, ChevronRight } from 'lucide-react';

export const TheoreticalFoundations: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'fick' | 'sde' | 'density' | 'update' | 'stability'>('fick');

  return (
    <div className="w-full my-16">
      <div className="text-center max-w-3xl mx-auto mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-stone-100 text-stone-700 text-xs font-bold tracking-widest uppercase rounded-full mb-4 border border-stone-200">
          <BookOpen size={14} className="text-nobel-gold" /> SECTION 2 & ALGORITHM 1
        </div>
        <h2 className="font-serif text-4xl md:text-5xl text-stone-900 mb-4">
          Theoretical Foundations of SDAO
        </h2>
        <p className="text-lg text-stone-600 font-light leading-relaxed">
          Grounding metaheuristic optimization in classical physics: from Fick's Second Law of Diffusion to Itô Stochastic Differential Equations and Mean-Square Contractivity.
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex flex-wrap justify-center gap-2 mb-8 border-b border-stone-200 pb-4">
        {[
          { id: 'fick', label: "1. Fick's 2nd Law & Diffusion", icon: <Compass size={16} /> },
          { id: 'sde', label: '2. Stochastic Dynamics & Fokker-Planck', icon: <GitBranch size={16} /> },
          { id: 'density', label: '3. Density-Based Repulsion (D_FL)', icon: <Layers size={16} /> },
          { id: 'update', label: '4. Composite SDAO Update Rule', icon: <Cpu size={16} /> },
          { id: 'stability', label: '5. Mean-Square Stability in High-d', icon: <ShieldCheck size={16} /> },
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

      {/* Tab Content Cards */}
      <div className="bg-white rounded-2xl border border-stone-200 shadow-lg p-8 md:p-12 transition-all duration-300">
        {activeTab === 'fick' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-7 space-y-6">
              <h3 className="font-serif text-3xl text-stone-900">
                Reinterpreting Fick's Second Law for Search Spaces
              </h3>
              <p className="text-stone-600 text-lg leading-relaxed">
                In physical chemical systems, Fick's Second Law governs how particles spontaneously disperse from areas of high concentration to areas of lower concentration over time.
              </p>
              <div className="bg-[#F9F8F4] p-6 rounded-xl border border-stone-200 font-mono text-center my-6">
                <div className="text-xs text-stone-400 font-sans tracking-widest uppercase mb-2">Continuous Diffusion Equation</div>
                <div className="text-2xl md:text-3xl font-bold text-stone-900">
                  ∂C(r, t) / ∂t = D ∇²C(r, t)
                </div>
              </div>
              <p className="text-stone-600 leading-relaxed">
                SDAO maps this physical analogy directly to population-based optimization:
              </p>
              <ul className="space-y-3 text-sm text-stone-700">
                <li className="flex items-start gap-3">
                  <CheckCircle2 size={18} className="text-emerald-600 shrink-0 mt-0.5" />
                  <span><strong>Concentration C(r, t):</strong> Represents the local density of candidate solutions at position r and iteration t. Overcrowded regions correspond to local optima basins.</span>
                </li>
                <li className="flex items-start gap-3">
                  <CheckCircle2 size={18} className="text-emerald-600 shrink-0 mt-0.5" />
                  <span><strong>Diffusion Coefficient D:</strong> Controls the dispersion rate of candidate solutions across the n-dimensional domain.</span>
                </li>
                <li className="flex items-start gap-3">
                  <CheckCircle2 size={18} className="text-emerald-600 shrink-0 mt-0.5" />
                  <span><strong>Laplacian ∇²:</strong> Models the spontaneous migration of particles away from densely clustered zones toward sparsely sampled, underexplored areas.</span>
                </li>
              </ul>
            </div>
            <div className="lg:col-span-5 bg-stone-900 text-white p-8 rounded-2xl border border-stone-800 space-y-4">
              <div className="text-xs font-bold tracking-widest text-nobel-gold uppercase">PHYSICAL ↔ ALGORITHMIC MAP</div>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between items-center border-b border-stone-800 pb-2">
                  <span className="text-stone-300">Particle position</span>
                  <span className="font-mono text-white">xᵢ ∈ ℝⁿ</span>
                </div>
                <div className="flex justify-between items-center border-b border-stone-800 pb-2">
                  <span className="text-stone-300">Concentration</span>
                  <span className="font-mono text-white">ρ(x) density</span>
                </div>
                <div className="flex justify-between items-center border-b border-stone-800 pb-2">
                  <span className="text-stone-300">Diffusion flux</span>
                  <span className="font-mono text-white">J = -D∇C</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-stone-300">Thermal noise</span>
                  <span className="font-mono text-white">√(2D)·ηᵢᵏ</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'sde' && (
          <div className="space-y-8">
            <div className="max-w-3xl">
              <h3 className="font-serif text-3xl text-stone-900 mb-4">
                Stochastic Differential Equations & Fokker-Planck
              </h3>
              <p className="text-stone-600 text-lg leading-relaxed">
                SDAO's dynamics are governed by an Itô SDE combining deterministic drift (attraction) with stochastic diffusion (Brownian motion), ensuring rigorous mathematical tractability.
              </p>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
              <div className="lg:col-span-7 bg-[#F9F8F4] p-8 rounded-2xl border border-stone-200">
                <div className="text-xs text-stone-400 font-sans tracking-widest uppercase mb-3">Itô Stochastic Differential Equation</div>
                <div className="text-xl md:text-2xl font-mono font-bold text-stone-900 text-center leading-relaxed">
                  dxᵢ = b(xᵢ)dt + σ(xᵢ)dWₜ
                </div>
                <div className="mt-6 grid grid-cols-2 gap-4 text-xs text-stone-600">
                  <div className="bg-white p-3 rounded-lg border border-stone-200">
                    <strong className="text-stone-900 block mb-1">Drift b(x):</strong>
                    Deterministic attraction toward global & personal bests.
                  </div>
                  <div className="bg-white p-3 rounded-lg border border-stone-200">
                    <strong className="text-stone-900 block mb-1">Diffusion σ(x):</strong>
                    Stochastic exploration via Brownian noise scaled by √(2D).
                  </div>
                </div>
              </div>
              <div className="lg:col-span-5 space-y-4">
                <div className="bg-stone-900 text-white p-6 rounded-2xl border border-stone-800">
                  <div className="text-xs font-bold tracking-widest text-nobel-gold uppercase mb-2">FOKKER-PLANCK EQUATION</div>
                  <p className="text-stone-300 text-sm leading-relaxed mb-3">
                    The probability density p(x,t) of the swarm evolves according to:
                  </p>
                  <div className="bg-stone-800 p-4 rounded-xl font-mono text-center text-sm text-white border border-stone-700">
                    ∂p/∂t = -∇·(bp) + ½∇²(σ²p)
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'density' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-7 space-y-6">
              <h3 className="font-serif text-3xl text-stone-900">
                Density-Based Repulsion Force (D_FL)
              </h3>
              <p className="text-stone-600 text-lg leading-relaxed">
                The signature innovation of SDAO: a Fickian repulsion field that pushes particles away from overcrowded clusters, preventing premature convergence in local optima basins.
              </p>
              <div className="bg-[#F9F8F4] p-6 rounded-xl border border-stone-200 font-mono text-center my-6">
                <div className="text-xs text-stone-400 font-sans tracking-widest uppercase mb-2">Density Repulsion Vector</div>
                <div className="text-xl md:text-2xl font-bold text-stone-900">
                  gᵢ = -∇ρ(xᵢ) · D(k)
                </div>
              </div>
              <ul className="space-y-3 text-sm text-stone-700">
                <li className="flex items-start gap-3">
                  <CheckCircle2 size={18} className="text-emerald-600 shrink-0 mt-0.5" />
                  <span><strong>Local density ρ(xᵢ):</strong> Estimated via k-d tree neighborhood queries within radius r* — O(N log N) cost.</span>
                </li>
                <li className="flex items-start gap-3">
                  <CheckCircle2 size={18} className="text-emerald-600 shrink-0 mt-0.5" />
                  <span><strong>Gradient -∇ρ:</strong> Points away from high-density regions, creating the "diffusion pressure" that escapes traps.</span>
                </li>
                <li className="flex items-start gap-3">
                  <CheckCircle2 size={18} className="text-emerald-600 shrink-0 mt-0.5" />
                  <span><strong>Adaptive D(k):</strong> Diffusion coefficient decays over iterations, smoothly transitioning from exploration to exploitation.</span>
                </li>
              </ul>
            </div>
            <div className="lg:col-span-5 bg-stone-900 text-white p-8 rounded-2xl border border-stone-800 space-y-4">
              <h4 className="font-serif text-2xl text-white">k-d Tree Neighborhood</h4>
              <p className="text-stone-300 text-sm leading-relaxed">
                The density at each particle is computed efficiently using a k-d tree spatial index, querying the m nearest neighbors:
              </p>
              <div className="bg-stone-800 p-4 rounded-xl font-mono text-center text-sm text-nobel-gold border border-stone-700">
                ρ(xᵢ) = m / |B(xᵢ, r*)|
              </div>
              <p className="text-stone-400 text-xs leading-relaxed">
                Where B(xᵢ, r*) is the ball of radius r* centered at xᵢ. The neighborhood degree m is a key hyperparameter analyzed in the sensitivity study.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'update' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-7 space-y-6">
              <h3 className="font-serif text-3xl text-stone-900">
                Composite SDAO Update Rule
              </h3>
              <p className="text-stone-600 text-lg leading-relaxed">
                Each iteration combines three forces: density repulsion (diversification), attraction (intensification), and stochastic noise (exploration), into a single update vector.
              </p>
              <div className="bg-[#F9F8F4] p-6 rounded-xl border border-stone-200 font-mono text-center my-6 space-y-3">
                <div className="text-xs text-stone-400 font-sans tracking-widest uppercase">SDAO Position Update (Eq. 11)</div>
                <div className="text-lg md:text-xl font-bold text-stone-900 leading-relaxed">
                  xᵢᵏ⁺¹ = xᵢᵏ + δ(k)·(x_Gbest - xᵢᵏ) + γ(k)·(x_best,i - xᵢᵏ) + gᵢᵏ + √(2D)·ηᵢᵏ
                </div>
              </div>
              <div className="space-y-3">
                <div className="p-3.5 bg-stone-50 rounded-xl border border-stone-200">
                  <strong className="text-stone-900 block mb-1">1. Diversification (D_FL):</strong>
                  Density repulsion prevents clustering and drives exploration into unmapped territories.
                </div>
                <div className="p-3.5 bg-stone-50 rounded-xl border border-stone-200">
                  <strong className="text-stone-900 block mb-1">2. Intensification (δ, γ):</strong>
                  Linear attractors pull the swarm toward the global best x_Gbest and personal bests x_best,i.
                </div>
                <div className="p-3.5 bg-stone-50 rounded-xl border border-stone-200">
                  <strong className="text-stone-900 block mb-1">3. Stochasticity (√(2D)ηᵢᵏ):</strong>
                  Additive Gaussian noise guarantees continued escape from deceptive local basins.
                </div>
              </div>
            </div>
            <div className="lg:col-span-5 bg-[#F9F8F4] p-8 rounded-2xl border border-stone-200 space-y-4">
              <h4 className="font-serif text-2xl text-stone-900">Convergence Insights</h4>
              <p className="text-stone-600 text-sm leading-relaxed">
                From a rigorous theoretical standpoint, deterministic convergence is achieved when all particles collapse into an attraction manifold defined by x_Gbest and x_best,i:
              </p>
              <div className="bg-white p-4 rounded-xl border border-stone-200 font-mono text-center text-sm text-stone-900">
                lim(k→∞) ||xᵢᵏ - x_Gbest|| → 0
              </div>
              <p className="text-stone-600 text-sm leading-relaxed">
                Because δ, γ ∈ (0, 1) with δ + γ ≤ 1, the error recursion {'e_{k+1} = (1 - δ - γ)e_k'} guarantees monotonic geometric contraction without overshoot or oscillations.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'stability' && (
          <div className="space-y-8">
            <div className="max-w-3xl">
              <h3 className="font-serif text-3xl text-stone-900 mb-4">
                High-Dimensional Stability & Dimension-Aware Caps
              </h3>
              <p className="text-stone-600 text-lg leading-relaxed">
                As problem dimensionality d scales up to 100 or 500 (e.g., SOCO11), unconstrained stochastic variance 2Dn·h grows linearly with n, causing standard metaheuristics to diverge or plateau. SDAO enforces strict dissipativity margins and dimension-aware caps.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-stone-900 text-white p-8 rounded-2xl border border-stone-800 space-y-4">
                <div className="text-xs font-bold tracking-widest text-nobel-gold uppercase">RULE 1: DISSIPATIVITY MARGIN</div>
                <h4 className="font-serif text-2xl text-white">One-Sided Lipschitz Dissipativity</h4>
                <p className="text-stone-300 text-sm leading-relaxed">
                  For the Euler-Maruyama discretization to remain mean-square stable, the drift term b(x) must satisfy the one-sided Lipschitz dissipativity margin α_c:
                </p>
                <div className="bg-stone-800 p-4 rounded-xl font-mono text-center text-sm text-nobel-gold border border-stone-700">
                  {'α_c(k) := (δ(k) + γ(k)) - L_DFL > 0'}
                </div>
                <p className="text-stone-400 text-xs leading-relaxed">
                  By bounding δ(k) + γ(k) ≥ a₀ with a₀ ∈ [0.05, 0.3], the linear attraction terms uniformly overpower the Lipschitz bound L_DFL of the density repulsion.
                </p>
              </div>

              <div className="bg-stone-900 text-white p-8 rounded-2xl border border-stone-800 space-y-4">
                <div className="text-xs font-bold tracking-widest text-nobel-gold uppercase">RULE 2: DIMENSION-AWARE CAP</div>
                <h4 className="font-serif text-2xl text-white">Variance Regulation Equation (15)</h4>
                <p className="text-stone-300 text-sm leading-relaxed">
                  To prevent the additive noise variance from exploding in high dimensions (n), SDAO clamps the adapted diffusion coefficient D_raw(k) against the ceiling D_cap(k):
                </p>
                <div className="bg-stone-800 p-4 rounded-xl font-mono text-center text-sm text-nobel-gold border border-stone-700">
                  D(k) = min( D_raw(k), c(2α_c(k) - L_b(k)²) / (2n) )
                </div>
                <p className="text-stone-400 text-xs leading-relaxed">
                  This inverse scaling with 2n ensures that lim(k→∞) E[||X_k||²] remains uniformly bounded, preserving classical strong convergence of order 1/2 even when d = 500!
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
