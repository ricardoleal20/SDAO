/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { BenchmarkFunction, RealWorldProblem, EmpiricalResultRow } from '../types';

export const BENCHMARK_FUNCTIONS: BenchmarkFunction[] = [
  {
    id: 'rastrigin',
    name: 'Rastrigin Function',
    category: 'standard',
    modality: 'Multimodal',
    searchSpace: 'Oscillatory (-5.12 to 5.12)',
    characteristics: 'Tests escape from local optima with dense cosine oscillations',
    globalOptimum: 0,
    bounds: [-5.12, 5.12],
    equationDisplay: 'f(x) = 10d + \\sum [x_i^2 - 10\\cos(2\\pi x_i)]',
    defaultDim: 50
  },
  {
    id: 'ackley',
    name: 'Ackley Function (Stochastic)',
    category: 'stochastic',
    modality: 'Multimodal',
    searchSpace: 'Flat outer region (-32.768 to 32.768)',
    characteristics: 'Evaluates balance between exploration/exploitation with deceptive flat outer regions + additive Gaussian noise',
    globalOptimum: 0,
    bounds: [-32.768, 32.768],
    equationDisplay: 'f(x) = -20\\exp(-0.2\\sqrt{\\frac{1}{d}\\sum x_i^2}) - \\exp(\\frac{1}{d}\\sum\\cos(2\\pi x_i)) + 20 + e + \\epsilon',
    defaultDim: 50
  },
  {
    id: 'rosenbrock',
    name: 'Rosenbrock Function',
    category: 'standard',
    modality: 'Unimodal',
    searchSpace: 'Narrow curved valley (-5 to 10)',
    characteristics: 'Tests navigation inside non-linear parabolic banana-shaped valleys',
    globalOptimum: 0,
    bounds: [-2.0, 2.0],
    equationDisplay: 'f(x) = \\sum [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]',
    defaultDim: 50
  },
  {
    id: 'schwefel',
    name: 'Schwefel Function',
    category: 'standard',
    modality: 'Multimodal',
    searchSpace: 'Deceptive distant optima (-500 to 500)',
    characteristics: 'Tests effectiveness in global search with second-best optima far from the global minimum',
    globalOptimum: 0,
    bounds: [-500, 500],
    equationDisplay: 'f(x) = 418.9829d - \\sum x_i\\sin(\\sqrt{|x_i|})',
    defaultDim: 50
  },
  {
    id: 'griewank',
    name: 'Griewank Function',
    category: 'standard',
    modality: 'Multimodal',
    searchSpace: 'Moderate complexity (-600 to 600)',
    characteristics: 'Balances local/global search efficiency with product cosine interference',
    globalOptimum: 0,
    bounds: [-10, 10],
    equationDisplay: 'f(x) = \\frac{1}{4000}\\sum x_i^2 - \\prod \\cos(\\frac{x_i}{\\sqrt{i}}) + 1',
    defaultDim: 50
  },
  {
    id: 'sphere',
    name: 'Sphere Function',
    category: 'standard',
    modality: 'Unimodal',
    searchSpace: 'Convex (-5.12 to 5.12)',
    characteristics: 'Measures baseline convergence speed and deterministic stability',
    globalOptimum: 0,
    bounds: [-5.12, 5.12],
    equationDisplay: 'f(x) = \\sum x_i^2',
    defaultDim: 50
  },
  {
    id: 'drop-wave',
    name: 'Drop-Wave Function',
    category: 'standard',
    modality: 'Multimodal',
    searchSpace: 'Sharp concentric valleys (-5.12 to 5.12)',
    characteristics: 'Evaluates handling of abrupt gradient changes around origin',
    globalOptimum: -1,
    bounds: [-5.12, 5.12],
    equationDisplay: 'f(x) = -\\frac{1 + \\cos(12\\sqrt{x^2+y^2})}{0.5(x^2+y^2) + 2}',
    defaultDim: 2
  },
  {
    id: 'weierstrass',
    name: 'Weierstrass Function (Stochastic)',
    category: 'stochastic',
    modality: 'Fractal-like',
    searchSpace: 'Oscillatory non-smooth (-0.5 to 0.5)',
    characteristics: 'Evaluates performance on continuous nowhere-differentiable fractal terrains under uncertainty',
    globalOptimum: 0,
    bounds: [-0.5, 0.5],
    equationDisplay: 'f(x) = \\sum_{i=1}^d \\sum_{k=0}^{k_{max}} [a^k \\cos(2\\pi b^k(x_i + 0.5))] + \\epsilon',
    defaultDim: 50
  },
  {
    id: 'shifted-rastrigin',
    name: 'Shifted & Rotated Rastrigin (CEC 2017)',
    category: 'cec',
    modality: 'Multimodal',
    searchSpace: 'Asymmetric rotated landscape (-100 to 100)',
    characteristics: 'Eliminates algorithmic bias due to origin symmetry and introduces coordinate coupling',
    globalOptimum: 0,
    bounds: [-5.12, 5.12],
    equationDisplay: 'f(x) = f_{Rastrigin}(M(x - o)) + f_{bias}',
    defaultDim: 50
  },
  {
    id: 'supply-chain',
    name: 'Supply Chain Network Design (Real-World)',
    category: 'real-world',
    modality: 'Complex',
    searchSpace: 'Constrained multi-echelon logistics (-10 to 10)',
    characteristics: 'Optimizes production, warehousing, transportation, and distribution costs under demand uncertainty',
    globalOptimum: 0,
    bounds: [-10, 10],
    equationDisplay: 'f(x) = \\sum C_{prod}(x) + \\sum C_{trans}(x) + \\sum C_{hold}(x) + \\text{penalty}',
    defaultDim: 50
  }
];

// Evaluate objective function in 2D (x, y) with optional noise variance
export function evaluateFunction2D(id: string, x: number, y: number, noiseVar: number = 0): number {
  let val = 0;
  const noise = noiseVar > 0 ? (Math.random() - 0.5) * Math.sqrt(noiseVar) * 2 : 0;

  switch (id) {
    case 'rastrigin':
    case 'shifted-rastrigin': {
      // Shifted by offset for cec simulation if desired
      const ox = id === 'shifted-rastrigin' ? x - 1.2 : x;
      const oy = id === 'shifted-rastrigin' ? y + 0.8 : y;
      val = 20 + (ox * ox - 10 * Math.cos(2 * Math.PI * ox)) + (oy * oy - 10 * Math.cos(2 * Math.PI * oy));
      break;
    }
    case 'ackley': {
      const sumSq = 0.5 * (x * x + y * y);
      const sumCos = 0.5 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y));
      val = -20 * Math.exp(-0.2 * Math.sqrt(sumSq)) - Math.exp(sumCos) + 20 + Math.E;
      break;
    }
    case 'rosenbrock': {
      val = 100 * Math.pow(y - x * x, 2) + Math.pow(1 - x, 2);
      break;
    }
    case 'schwefel': {
      const sx = x * 100;
      const sy = y * 100;
      val = 418.9829 * 2 - (sx * Math.sin(Math.sqrt(Math.abs(sx))) + sy * Math.sin(Math.sqrt(Math.abs(sy))));
      val = val / 100; // normalize scale for viz
      break;
    }
    case 'griewank': {
      const sumSq = (x * x + y * y) / 4000;
      const prodCos = Math.cos(x) * Math.cos(y / Math.sqrt(2));
      val = sumSq - prodCos + 1;
      break;
    }
    case 'drop-wave': {
      const rSq = x * x + y * y;
      val = - (1 + Math.cos(12 * Math.sqrt(rSq))) / (0.5 * rSq + 2) + 1; // +1 to make >= 0
      break;
    }
    case 'weierstrass': {
      let sum = 0;
      const a = 0.5, b = 3;
      for (let k = 0; k <= 10; k++) {
        sum += Math.pow(a, k) * (Math.cos(2 * Math.PI * Math.pow(b, k) * (x + 0.5)) + Math.cos(2 * Math.PI * Math.pow(b, k) * (y + 0.5)));
      }
      val = sum + 2;
      break;
    }
    case 'supply-chain': {
      // Rugged multi-basin cost landscape simulation
      val = Math.pow(x - 2, 2) + Math.pow(y + 1, 2) + 3 * Math.sin(3 * x) * Math.cos(3 * y) + 5;
      break;
    }
    case 'sphere':
    default: {
      val = x * x + y * y;
      break;
    }
  }

  return Math.max(0, val + noise);
}

export const REAL_WORLD_PROBLEMS: RealWorldProblem[] = [
  {
    title: 'Cache Optimization',
    reference: 'Psounis K. (2001)',
    category: 'Computational',
    application: 'Cloud computing & embedded memory architectures',
    objective: 'Minimize cache miss rates and memory latency bottlenecks'
  },
  {
    title: 'Production Scheduling',
    reference: 'Uzsoy R. et al. (2023)',
    category: 'Logistics',
    application: 'Manufacturing & flexible assembly systems',
    objective: 'Minimize completion delays and optimize machine resource allocation'
  },
  {
    title: 'Online Advertising Bidding',
    reference: 'Majima K. et al. (2024)',
    category: 'Financial',
    application: 'Digital marketing & e-commerce sponsored search',
    objective: 'Maximize ROI while balancing Cost-Per-Click (CPC) against strict budget constraints'
  },
  {
    title: 'Network Packet Routing',
    reference: 'Peterson L. & Davie B. (2021)',
    category: 'Network',
    application: 'Telecommunications & 5G dynamic mesh networks',
    objective: 'Minimize end-to-end latency, network congestion, and packet loss rates'
  },
  {
    title: 'Retail Inventory Optimization',
    reference: 'Chopra S. & Meindl P. (2023)',
    category: 'Logistics',
    application: 'Global supply chains & warehouse management',
    objective: 'Minimize stockout occurrences and reduce holding/storage inventory costs'
  },
  {
    title: 'Supply Chain Network Design',
    reference: 'Whittington R. et al. (2023)',
    category: 'Logistics',
    application: 'Industrial operations & multi-tier distribution',
    objective: 'Optimize combined production, transportation, and multi-facility distribution costs'
  }
];

export const ALGORITHM_COMPARISONS = [
  {
    name: 'SDAO (Proposed)',
    principle: 'Diffusion-based stochastic optimization',
    strategy: 'Adaptive density-based search + Brownian noise',
    adaptation: 'Yes (Dynamic alpha, gamma, D + OBL + Bound Contraction)'
  },
  {
    name: 'SHADE with ILS',
    principle: 'Differential Evolution (DE)',
    strategy: 'Population-based mutation + Iterated Local Search',
    adaptation: 'Yes (Historical success memory adaptation)'
  },
  {
    name: 'AMSO',
    principle: 'Multi-Swarm Particle Swarm Optimization',
    strategy: 'Multiple interacting sub-swarms',
    adaptation: 'Yes (Adaptive inter-swarm communication)'
  },
  {
    name: 'TLPSO',
    principle: 'Hierarchical Particle Swarm Optimization',
    strategy: 'Two-level swarm topology structure',
    adaptation: 'Yes (Dynamic hierarchy & topology)'
  },
  {
    name: 'Algebraic SGD',
    principle: 'Algebraic Stochastic Gradient Descent',
    strategy: 'Gradient-based stochastic updates',
    adaptation: 'No (Fixed learning rate schedule)'
  },
  {
    name: 'Stochastic Fractal Search (SFS)',
    principle: 'Fractal-Inspired Metaheuristic',
    strategy: 'Self-similarity growth and randomized Lévy flights',
    adaptation: 'Yes (Randomized step diffusion)'
  },
  {
    name: 'Path Relinking',
    principle: 'Trajectory-based optimization',
    strategy: 'Intensification via intermediate paths between elite solutions',
    adaptation: 'No (Fixed search trajectories)'
  },
  {
    name: 'Fishing Cat Optimizer (FCO)',
    principle: 'Nature-inspired metaheuristic (animal behavior)',
    strategy: 'Cooperative hunting and prey-capturing phases',
    adaptation: 'Yes (Adaptive exploration/exploitation trade-off)'
  },
  {
    name: 'Gyro Fireworks Algorithm (GFA)',
    principle: 'Fireworks-inspired metaheuristic',
    strategy: 'Explosion with gyro-based rotating sparks',
    adaptation: 'Yes (Enhanced diversity & convergence stability)'
  },
  {
    name: 'Starfish Optimization (SFOA)',
    principle: 'Bio-inspired metaheuristic (marine biology)',
    strategy: 'Regeneration-based operators + decentralized coordination',
    adaptation: 'Yes (Self-adaptive regeneration)'
  },
  {
    name: 'PaDE-Pet',
    principle: 'Differential Evolution (DE)',
    strategy: 'Population enhancement with parameter control',
    adaptation: 'Yes (Parameter adaptation + diversity maintenance)'
  }
];

// Table 8 (d=50) & Table 9 (d=100) Summary Data for Empirical comparison
export const EMPIRICAL_RESULTS_D50: EmpiricalResultRow[] = [
  { algorithm: 'SDAO (Proposed)', d10: '140 ± 26.3', d25: '508 ± 29.0', d50: '1.25k ± 106', d100: '8.01k ± 964', isBest: true },
  { algorithm: 'SHADEwithILS', d10: '1.02 ± 1.72', d25: '329 ± 119', d50: '1.08e+21 ± 5.03e+21', d100: 'inf ± 1.16e+63' },
  { algorithm: 'AMSO', d10: '8.04k ± 7.19k', d25: '2.07e+15 ± 3.94e+15', d50: '2.11e+37 ± 8.59e+37', d100: '8.13e+81 ± 2.42e+82' },
  { algorithm: 'TLPSO', d10: '5.61k ± 2.58k', d25: '5.20e+14 ± 1.17e+15', d50: '1.16e+35 ± 2.15e+35', d100: '1.04e+80 ± 2.31e+80' },
  { algorithm: 'AlgebraicSGD', d10: '83.8M ± 139M', d25: '2.31e+22 ± 9.69e+22', d50: '2.78e+47 ± 8.31e+47', d100: '1.78e+97 ± 8.98e+97' },
  { algorithm: 'StochasticFractalSearch', d10: '26.9k ± 22.5k', d25: '2.11e+16 ± 8.01e+16', d50: '1.41e+38 ± 4.87e+38', d100: '1.06e+82 ± 3.68e+82' },
  { algorithm: 'FCO', d10: '135 ± 34.0', d25: '412 ± 60.5', d50: '969 ± 89.9', d100: '4.00e+14 ± 2.01e+15' },
  { algorithm: 'GFA', d10: '105 ± 22.2', d25: '339 ± 46.3', d50: '691 ± 79.1', d100: '1.55e+18 ± 6.38e+18' },
  { algorithm: 'SFOA', d10: '130 ± 12.5', d25: '485 ± 23.9', d50: '1.15k ± 38.8', d100: '2.14e+17 ± 1.15e+18' },
  { algorithm: 'PaDE-pet', d10: '1.60 ± 0.44', d25: '96.9 ± 25.4', d50: '1.71e+15 ± 2.29e+15', d100: '6.11e+56 ± 2.92e+57' }
];

// Table 13 Ablation Study Data (d=50 Stochastic Benchmark)
export const ABLATION_STUDY_DATA = [
  { variant: 'SDAO (Full Version)', meanError: 689.79, stdDev: 420.94, desc: 'Includes Density Diffusion, OBL, and Periodic Bound Contraction', color: '#047857' },
  { variant: 'SDAO w/o Contraction', meanError: 1497.52, stdDev: 191.38, desc: 'Removes periodic contraction of search box around global best', color: '#D97706' },
  { variant: 'SDAO w/o OBL', meanError: 7485.99, stdDev: 6051.20, desc: 'Removes probabilistic Opposition-Based Learning under stagnation', color: '#DC2626' }
];

// Table 15 SOCO11 Benchmark Suite at d=500
export const SOCO11_RESULTS_D500 = [
  { function: 'Shifted Sphere', bestAlgo: 'GFA', bestValue: '8500.1444 ± 442.6224', isSDAO: false },
  { function: 'Schwefel Problem', bestAlgo: 'AlgebraicSGD', bestValue: '258.0294 ± 4.2281', isSDAO: false },
  { function: 'Shifted Rosenbrock', bestAlgo: 'AlgebraicSGD', bestValue: '8.75e+10 ± 1.62e+08', isSDAO: false },
  { function: 'Shifted Rastrigin', bestAlgo: 'SFOA', bestValue: '4476.3931 ± 125.1079', isSDAO: false },
  { function: 'Shifted Griewank', bestAlgo: 'GFA', bestValue: '97.8929 ± 2.5715', isSDAO: false },
  { function: 'Shifted Ackley', bestAlgo: 'AlgebraicSGD', bestValue: '118.4152 ± 0.0329', isSDAO: false },
  { function: 'Bohachevsky (shifted)', bestAlgo: 'SDAO', bestValue: '2.18e+04 ± 667.1546', isSDAO: true },
  { function: 'Hybrid 12', bestAlgo: 'GFA', bestValue: '6222.4475 ± 175.8124', isSDAO: false },
  { function: 'Hybrid 14', bestAlgo: 'SDAO', bestValue: '7022.1895 ± 117.9362', isSDAO: true },
  { function: 'Hybrid 16', bestAlgo: 'GFA', bestValue: '5160.7915 ± 137.6104', isSDAO: false },
  { function: 'Hybrid 18', bestAlgo: 'SDAO', bestValue: '5134.2125 ± 74.2017', isSDAO: true },
  { function: 'Hybrid 19', bestAlgo: 'SDAO', bestValue: '8.11e+11 ± 3.38e+12', isSDAO: true }
];

// ANOVA p-values across dimensions (Table 10)
export const ANOVA_SIGNIFICANCE_DATA = [
  { dim: 'd = 10', standard: '3.80 × 10⁻¹³', stochastic: '8.37 × 10⁻⁴⁷', realWorld: '1.79 × 10⁻²¹³', cec: '1.18 × 10⁻³' },
  { dim: 'd = 25', standard: '9.60 × 10⁻²', stochastic: '1.27 × 10⁻⁴⁴', realWorld: '2.32 × 10⁻²⁶⁴', cec: '1.11 × 10⁻¹' },
  { dim: 'd = 50', standard: '7.43 × 10⁻⁴', stochastic: '8.11 × 10⁻⁴⁴', realWorld: '5.11 × 10⁻¹²²', cec: '4.23 × 10⁻¹' },
  { dim: 'd = 100', standard: '3.33 × 10⁻¹', stochastic: '4.51 × 10⁻⁴⁰', realWorld: '3.64 × 10⁻²²²', cec: 'N/A' }
];

// Wilcoxon Statistically Significant Wins for SDAO (Table 12 at d=50)
export const WILCOXON_WINS_DATA = [
  { algorithm: 'PathRelinking', standard: '14 / 14', stochastic: '12 / 12', realWorld: '4 / 6', cec: '38 / 38' },
  { algorithm: 'StochasticFractalSearch (SFS)', standard: '14 / 14', stochastic: '12 / 12', realWorld: '6 / 6', cec: '38 / 38' },
  { algorithm: 'AlgebraicSGD', standard: '13 / 14', stochastic: '12 / 12', realWorld: '5 / 6', cec: '37 / 38' },
  { algorithm: 'TLPSO', standard: '14 / 14', stochastic: '11 / 12', realWorld: '4 / 6', cec: '35 / 38' },
  { algorithm: 'AMSO', standard: '14 / 14', stochastic: '10 / 12', realWorld: '6 / 6', cec: '35 / 38' },
  { algorithm: 'PaDE-pet', standard: '12 / 14', stochastic: '9 / 12', realWorld: '0 / 6', cec: '33 / 38' },
  { algorithm: 'SHADEwithILS', standard: '10 / 14', stochastic: '10 / 12', realWorld: '0 / 6', cec: '29 / 38' }
];
