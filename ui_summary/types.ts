/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';

export type BenchmarkCategory = 'standard' | 'stochastic' | 'real-world' | 'cec' | 'soco11';

export type AlgorithmId = 
  | 'SDAO' 
  | 'SHADEwithILS' 
  | 'AMSO' 
  | 'TLPSO' 
  | 'AlgebraicSGD' 
  | 'SFS' 
  | 'PathRelinking' 
  | 'FCO' 
  | 'GFA' 
  | 'SFOA' 
  | 'PaDE-Pet';

export interface BenchmarkFunction {
  id: string;
  name: string;
  category: BenchmarkCategory;
  modality: 'Unimodal' | 'Multimodal' | 'Fractal-like' | 'Hierarchical' | 'Complex';
  searchSpace: string;
  characteristics: string;
  globalOptimum: number;
  bounds: [number, number];
  equationDisplay?: string;
  defaultDim: number;
}

export interface Particle {
  id: number;
  position: [number, number]; // 2D projection or 2D evaluation for viz
  personalBest: [number, number];
  personalBestValue: number;
  stagnationCount: number;
  densityGradient: [number, number]; // Repulsion vector g_i
  densityValue: number;
}

export interface SimulationConfig {
  populationSize: number;
  maxIterations: number;
  learningRate: number;      // alpha_0
  memoryCoeff: number;       // gamma_0
  diffusionCoeff: number;    // D_0
  densityRadius: number;     // r^* / rho_0
  decayRate: number;         // lambda
  contractInterval: number;  // m iterations
  noiseVariance: number;     // sigma^2 for stochastic benchmarks
  dimension: number;         // d
  algorithm: AlgorithmId;
  benchmarkId: string;
  enableOBL: boolean;
  enableContraction: boolean;
  enableDiffusion: boolean;
}

export interface IterationMetric {
  iteration: number;
  bestValue: number;
  avgValue: number;
  diversity: number;
  alpha: number;
  gamma: number;
  diffusion: number;
  oblCount: number;
}

export interface EmpiricalResultRow {
  algorithm: string;
  d10: string;
  d25: string;
  d50: string;
  d100?: string;
  isBest?: boolean;
}

export interface RealWorldProblem {
  title: string;
  reference: string;
  category: string;
  application: string;
  objective: string;
}

export interface AuthorInfo {
  name: string;
  affiliation: string;
  email: string;
  location: string;
}
