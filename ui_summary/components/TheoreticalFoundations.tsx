/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { BookOpen, Cpu, ShieldCheck, Compass, GitBranch, Layers, CheckCircle2, ChevronRight } from 'lucide-react';
import { Reveal } from './Reveal';

export const TheoreticalFoundations: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'fick' | 'sde' | 'density' | 'update' | 'stability'>('fick');

  return (
    <Reveal className="w-full my-16">
      <div className="mx-auto mb-16 max-w-3xl">
        <span className="eyebrow">Theoretical foundations</span>
        <h2 className="display mt-4 text-4xl text-bone md:text-5xl">Grounded in classical physics, not analogy</h2>
        <p className="mt-5 text-lg font-light leading-relaxed text-white/55">Fick&rsquo;s Second Law of Diffusion, It&ocirc; Stochastic Differential Equations, and Mean-Square Contractivity form the mathematical spine of SDAO.</p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex flex-wrap justify-center gap-2 mb-8 border-b border-white/10 pb-4">
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
                ? 'bg-bone text-ink'
                : 'border border-white/15 text-white/55 hover:border-gold/50 hover:text-bone'
            }`}
          >
            <span className={activeTab === tab.id ? 'text-gold' : 'text-white/35'}>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content Cards */}
      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-8 md:p-12">
        {activeTab === 'fick' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-7 space-y-6">
              <h3 className="display text-3xl text-bone">
                Reinterpreting Fick's Second Law for Search Spaces
              </h3>
              <p className="text-white/60 text-lg leading-relaxed">
                In physical chemical systems, Fick's Second Law governs how particles spontaneously disperse from areas of high concentration to areas of lower concentration over time.
              </p>
              <div className="bg-white/[0.04] p-6 rounded-xl border border-white/10 font-mono text-center my-6">
                <div className="text-xs text-white/35 font-sans tracking-widest uppercase mb-2">Continuous Diffusion Equation</div>
                <div className="text-2xl md:text-3xl font-bold text-bone">
                  ∂C(r, t) / ∂t = D ∇²C(r, t)
                </div>
              </div>
              <p className="text-white/60 leading-relaxed">
                SDAO maps this physical analogy directly to population-based optimization:
              </p>
              <ul className="space-y-3 text-sm text-white/70">
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
            <div className="lg:col-span-5 bg-white/[0.04] text-white p-8 rounded-2xl border border-white/10 shadow-xl space-y-6 relative overflow-hidden">
              <div className="absolute -right-10 -bottom-10 w-40 h-40 bg-gold/20 rounded-full blur-2xl pointer-events-none"></div>
              <div className="text-xs font-bold tracking-widest text-gold uppercase">WHY IT MATTERS</div>
              <h4 className="display text-2xl text-white">Escaping Premature Convergence</h4>
              <p className="text-white/55 text-sm leading-relaxed">
                Classical evolutionary algorithms often collapse into a single cluster too early. When all particles gather in a local trap, gradient information vanishes (∇f ≈ 0).
              </p>
              <p className="text-white/55 text-sm leading-relaxed">
                By enforcing Fickian diffusion, SDAO generates an intrinsic thermodynamic repulsion exactly when density peaks, guaranteeing that the swarm never stagnates in false minima.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'sde' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-7 space-y-6">
              <h3 className="display text-3xl text-bone">
                Stochastic Dynamics & Fokker-Planck Connection
              </h3>
              <p className="text-white/60 text-lg leading-relaxed">
                To incorporate directed optimization behavior, Fick's equation is extended with a deterministic drift term μ(xᵢᵏ) acting as an external force field. This establishes a structural resemblance to the <strong>Fokker-Planck equation</strong> and the <strong>Itô Stochastic Differential Equation (SDE)</strong>:
              </p>
              <div className="bg-white/[0.04] p-6 rounded-xl border border-white/10 font-mono text-center my-6">
                <div className="text-xs text-white/35 font-sans tracking-widest uppercase mb-2">Itô Stochastic Differential Equation</div>
                <div className="text-2xl md:text-3xl font-bold text-bone">
                  dxᵢᵏ = μ(xᵢᵏ) dk + √(2D) dWᵏ
                </div>
              </div>
              <div className="space-y-4 text-white/70 text-sm">
                <div className="p-4 bg-white/[0.03] rounded-xl border border-white/10">
                  <strong className="text-bone block mb-1">1. The Drift Coefficient μ(xᵢᵏ):</strong>
                  Defined as the negative gradient of a scalar velocity potential v(xᵢᵏ) = αf(xᵢ), yielding μ(xᵢᵏ) = -α ∇f(xᵢᵏ). This biases particle motion toward decreasing objective potential.
                </div>
                <div className="p-4 bg-white/[0.03] rounded-xl border border-white/10">
                  <strong className="text-bone block mb-1">2. The Wiener Process dWᵏ (Brownian Motion):</strong>
                  Models continuous stochastic perturbations scaled by intensity √(2D). This injects controlled randomness, ensuring global search capability without sacrificing local convergence.
                </div>
              </div>
            </div>
            <div className="lg:col-span-5 bg-white/[0.04] text-white p-8 rounded-2xl border border-white/10 shadow-xl space-y-6">
              <div className="text-xs font-bold tracking-widest text-gold uppercase">NUMERICAL TRACTABILITY</div>
              <h4 className="display text-2xl text-white">Euler-Maruyama Discretization</h4>
              <p className="text-white/55 text-sm leading-relaxed">
                Continuous SDEs cannot be computed directly on digital hardware. SDAO applies the first-order Euler-Maruyama approximation with unit time step Δk = 1:
              </p>
              <div className="bg-white/[0.05] p-4 rounded-xl font-mono text-gold text-center text-sm border border-white/10">
                xᵢᵏ⁺¹ = xᵢᵏ - α ∇f(xᵢᵏ) + √(2D) ηᵢᵏ
              </div>
              <p className="text-white/35 text-xs italic">
                where ηᵢᵏ ~ N(0, I) is a standard normal random vector.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'density' && (
          <div className="space-y-8">
            <div className="max-w-3xl">
              <h3 className="display text-3xl text-bone mb-4">
                Replacing Gradient Descent with Density-Based Diffusion (D_FL)
              </h3>
              <p className="text-white/60 text-lg leading-relaxed">
                In rugged, noisy, or non-differentiable landscapes, gradients ∇f are misleading or computationally prohibitive to compute. SDAO completely replaces gradient descent with a gradient-free density repulsion term D_FL(xᵢ).
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                {
                  step: 'Step 1',
                  title: 'Neighborhood Identification',
                  formula: 'N_i = k-d tree(xᵢ, r*)',
                  desc: 'A spatial k-d tree identifies all neighbor particles within target degree radius r* = (m / [(N_p - 1)V_n(1)])^(1/n) in O(N log N) time.'
                },
                {
                  step: 'Step 2',
                  title: 'Center of Mass Computation',
                  formula: 'x̄ᵢ = (1 / |N_i|) Σ_{j ∈ N_i} xⱼ',
                  desc: 'Computes the local centroid x̄ᵢ of all neighboring particles surrounding candidate solution i.'
                },
                {
                  step: 'Step 3',
                  title: 'Density Gradient Estimation',
                  formula: 'gᵢ = xᵢ - x̄ᵢ  ⇒  ĝᵢ = gᵢ / ||gᵢ||',
                  desc: 'The local repulsion direction vector is the normalized difference between current position and the neighbor centroid.'
                },
                {
                  step: 'Step 4',
                  title: 'Diffusion Scaling',
                  formula: 'D_FL(xᵢ) = λ ĝᵢ',
                  desc: 'Scales the unit repulsion vector by adaptive diffusion strength parameter λ, pushing particles outward from clusters.'
                }
              ].map((item, idx) => (
                <div key={idx} className="bg-white/[0.03] p-6 rounded-xl border border-white/10 flex flex-col justify-between hover:border-gold transition-all">
                  <div>
                    <span className="text-xs font-bold text-gold tracking-widest uppercase">{item.step}</span>
                    <h4 className="display text-lg text-bone mt-1 mb-3">{item.title}</h4>
                    <div className="bg-white/[0.03] p-3 rounded-lg border border-white/10 font-mono text-xs text-bone text-center mb-4">
                      {item.formula}
                    </div>
                  </div>
                  <p className="text-xs text-white/60 leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'update' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-7 space-y-6">
              <h3 className="display text-3xl text-bone">
                The Complete SDAO Composite Update Rule
              </h3>
              <p className="text-white/60 text-lg leading-relaxed">
                By substituting the gradient descent term with D_FL(xᵢ) and integrating global (δ) and personal (γ) search attractors, we arrive at the master equation governing SDAO particle kinematics:
              </p>
              <div className="bg-white/[0.04] text-white p-6 md:p-8 rounded-2xl border border-white/10 shadow-xl font-mono text-center my-6">
                <div className="text-xs text-gold tracking-widest uppercase mb-3 font-sans">Equation (10) & (11) Master Kinematic Update</div>
                <div className="text-xl md:text-2xl font-bold leading-relaxed text-white">
                  xᵢᵏ⁺¹ = xᵢᵏ + D_FL(xᵢ) + δ(x_Gbest - xᵢᵏ) + γ(x_best,i - xᵢᵏ) + √(2D) ηᵢᵏ
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
                <div className="p-3.5 bg-white/[0.03] rounded-xl border border-white/10">
                  <strong className="text-bone block mb-1">1. Diversification (D_FL):</strong>
                  Density repulsion prevents clustering and drives exploration into unmapped territories.
                </div>
                <div className="p-3.5 bg-white/[0.03] rounded-xl border border-white/10">
                  <strong className="text-bone block mb-1">2. Intensification (δ, γ):</strong>
                  Linear attractors pull the swarm toward the global best x_Gbest and personal bests x_best,i.
                </div>
                <div className="p-3.5 bg-white/[0.03] rounded-xl border border-white/10">
                  <strong className="text-bone block mb-1">3. Stochasticity (√(2D)ηᵢᵏ):</strong>
                  Additive Gaussian noise guarantees continued escape from deceptive local basins.
                </div>
              </div>
            </div>
            <div className="lg:col-span-5 bg-white/[0.04] p-8 rounded-2xl border border-white/10 space-y-4">
              <h4 className="display text-2xl text-bone">Convergence Insights</h4>
              <p className="text-white/60 text-sm leading-relaxed">
                From a rigorous theoretical standpoint, deterministic convergence is achieved when all particles collapse into an attraction manifold defined by x_Gbest and x_best,i:
              </p>
              <div className="bg-white/[0.03] p-4 rounded-xl border border-white/10 font-mono text-center text-sm text-bone">
                lim(k→∞) ||xᵢᵏ - x_Gbest|| → 0
              </div>
              <p className="text-white/60 text-sm leading-relaxed">
                Because δ, γ ∈ (0, 1) with δ + γ ≤ 1, the error recursion {'e_{k+1} = (1 - δ - γ)e_k'} guarantees monotonic geometric contraction without overshoot or oscillations.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'stability' && (
          <div className="space-y-8">
            <div className="max-w-3xl">
              <h3 className="display text-3xl text-bone mb-4">
                High-Dimensional Stability & Dimension-Aware Caps
              </h3>
              <p className="text-white/60 text-lg leading-relaxed">
                As problem dimensionality d scales up to 100 or 500 (e.g., SOCO11), unconstrained stochastic variance 2Dn·h grows linearly with n, causing standard metaheuristics to diverge or plateau. SDAO enforces strict dissipativity margins and dimension-aware caps.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-white/[0.04] text-white p-8 rounded-2xl border border-white/10 space-y-4">
                <div className="text-xs font-bold tracking-widest text-gold uppercase">RULE 1: DISSIPATIVITY MARGIN</div>
                <h4 className="display text-2xl text-white">One-Sided Lipschitz Dissipativity</h4>
                <p className="text-white/55 text-sm leading-relaxed">
                  For the Euler-Maruyama discretization to remain mean-square stable, the drift term b(x) must satisfy the one-sided Lipschitz dissipativity margin α_c:
                </p>
                <div className="bg-white/[0.05] p-4 rounded-xl font-mono text-center text-sm text-gold border border-white/10">
                  {'α_c(k) := (δ(k) + γ(k)) - L_DFL > 0'}
                </div>
                <p className="text-white/35 text-xs leading-relaxed">
                  By bounding δ(k) + γ(k) ≥ a₀ with a₀ ∈ [0.05, 0.3], the linear attraction terms uniformly overpower the Lipschitz bound L_DFL of the density repulsion.
                </p>
              </div>

              <div className="bg-white/[0.04] text-white p-8 rounded-2xl border border-white/10 space-y-4">
                <div className="text-xs font-bold tracking-widest text-gold uppercase">RULE 2: DIMENSION-AWARE CAP</div>
                <h4 className="display text-2xl text-white">Variance Regulation Equation (15)</h4>
                <p className="text-white/55 text-sm leading-relaxed">
                  To prevent the additive noise variance from exploding in high dimensions (n), SDAO clamps the adapted diffusion coefficient D_raw(k) against the ceiling D_cap(k):
                </p>
                <div className="bg-white/[0.05] p-4 rounded-xl font-mono text-center text-sm text-gold border border-white/10">
                  D(k) = min( D_raw(k), c(2α_c(k) - L_b(k)²) / (2n) )
                </div>
                <p className="text-white/35 text-xs leading-relaxed">
                  This inverse scaling with 2n ensures that lim(k→∞) E[||X_k||²] remains uniformly bounded, preserving classical strong convergence of order 1/2 even when d = 500!
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </Reveal>
  );
};
