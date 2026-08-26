/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { HeroSimulation } from './components/HeroSimulation';
import { TheoreticalFoundations } from './components/TheoreticalFoundations';
import { AdaptiveMechanics } from './components/AdaptiveMechanics';
import { BenchmarkResults } from './components/BenchmarkResults';
import { AblationStudy } from './components/AblationStudy';
import { SDAOSimulator } from './components/SDAOSimulator';
import {
  ArrowDown, Menu, X, BookOpen, Github, ExternalLink, Sparkles,
  Award, ShieldCheck, Activity, Copy, Check, Globe2, FileText, Cpu
} from 'lucide-react';

const AuthorCard: React.FC = () => {
  const [copied, setCopied] = useState(false);

  const bibtex = `@article{leal2026sdao,
  title={Stochastic diffusion adaptive optimization, a novel metaheuristic approach},
  author={Leal Lopez, Ricardo M.},
  journal={Discover Analytics},
  volume={4},
  number={6},
  year={2026},
  doi={10.1007/s44257-025-00054-1}
}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full bg-white rounded-2xl border border-stone-200 shadow-lg p-8 md:p-12 my-16">
      <div className="text-center max-w-3xl mx-auto mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-stone-100 text-stone-700 text-xs font-bold tracking-widest uppercase rounded-full mb-3 border border-stone-200">
          <Award size={14} className="text-nobel-gold" /> AUTHOR & ATTRIBUTION
        </div>
        <h2 className="font-serif text-3xl md:text-4xl text-stone-900 mb-2">
          Ricardo M. Leal Lopez
        </h2>
        <p className="text-sm font-bold text-nobel-gold tracking-widest uppercase">
          Corresponding Author • Independent Researcher
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-center border-t border-stone-200 pt-8">
        <div className="md:col-span-5 space-y-4">
          <h3 className="font-serif text-xl text-stone-900">Open Access Research</h3>
          <p className="text-stone-600 text-sm leading-relaxed">
            Published in <strong>Discover Analytics (2026) 4:6</strong> under a Creative Commons Attribution 4.0 International License. The algorithm consistently demonstrated competitive performance and significantly outperformed state-of-the-art methods in noisy and high-dimensional environments.
          </p>
          <div className="flex flex-wrap gap-3 pt-2">
            <a
              href="https://github.com/ricardoleal20/SDAO"
              target="_blank"
              rel="noopener noreferrer"
              className="px-5 py-2.5 bg-stone-900 text-white rounded-xl text-xs font-bold tracking-wider uppercase hover:bg-stone-800 transition-all flex items-center gap-2 shadow-sm cursor-pointer"
            >
              <Github size={16} /> GitHub Repository
            </a>
            <a
              href="https://doi.org/10.1007/s44257-025-00054-1"
              target="_blank"
              rel="noopener noreferrer"
              className="px-5 py-2.5 bg-white text-stone-800 border border-stone-300 rounded-xl text-xs font-bold tracking-wider uppercase hover:bg-stone-50 transition-all flex items-center gap-2 shadow-xs cursor-pointer"
            >
              <ExternalLink size={16} className="text-nobel-gold" /> Read Paper (DOI)
            </a>
          </div>
        </div>

        <div className="md:col-span-7 bg-stone-900 text-stone-300 p-6 rounded-xl border border-stone-800 relative group">
          <div className="flex justify-between items-center mb-3 pb-2 border-b border-stone-800">
            <span className="text-xs font-mono text-nobel-gold font-bold">BibTeX Citation</span>
            <button
              onClick={handleCopy}
              className="px-3 py-1 bg-stone-800 hover:bg-stone-700 text-white rounded text-xs font-mono transition-colors flex items-center gap-1.5 cursor-pointer"
            >
              {copied ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              {copied ? 'Copied!' : 'Copy BibTeX'}
            </button>
          </div>
          <pre className="text-xs font-mono text-stone-300 overflow-x-auto whitespace-pre leading-relaxed">
            {bibtex}
          </pre>
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    setMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      const headerOffset = 90;
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth"
      });
    }
  };

  return (
    <div className="min-h-screen bg-[#F9F8F4] text-stone-800 selection:bg-nobel-gold selection:text-white">

      {/* Navigation */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'bg-[#F9F8F4]/95 backdrop-blur-md shadow-sm py-3.5 border-b border-stone-200' : 'bg-transparent py-6'}`}>
        <div className="container mx-auto px-6 flex justify-between items-center">
          <div className="flex items-center gap-3.5 cursor-pointer" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
            <div className="flex flex-col">
              <span className="font-serif font-bold text-lg tracking-wide text-stone-900 leading-tight">
                SDAO <span className="font-sans text-xs text-nobel-gold font-semibold uppercase tracking-widest ml-1">2026</span>
              </span>
              <span className="text-[10px] text-stone-500 font-mono -mt-0.5">Stochastic Diffusion Adaptive Opt</span>
            </div>
          </div>

          <div className="hidden lg:flex items-center gap-7 text-xs font-bold tracking-widest text-stone-600 uppercase">
            <a href="#simulator" onClick={scrollToSection('simulator')} className="hover:text-nobel-gold transition-colors cursor-pointer flex items-center gap-1">
              <Activity size={14} className="text-nobel-gold" /> Live Simulator
            </a>
            <a href="#theory" onClick={scrollToSection('theory')} className="hover:text-nobel-gold transition-colors cursor-pointer">Theory & SDE</a>
            <a href="#mechanics" onClick={scrollToSection('mechanics')} className="hover:text-nobel-gold transition-colors cursor-pointer">Adaptive Engine</a>
            <a href="#results" onClick={scrollToSection('results')} className="hover:text-nobel-gold transition-colors cursor-pointer">Empirical Results</a>
            <a href="#ablation" onClick={scrollToSection('ablation')} className="hover:text-nobel-gold transition-colors cursor-pointer">Ablation Study</a>
            <a href="#author" onClick={scrollToSection('author')} className="hover:text-nobel-gold transition-colors cursor-pointer">Author</a>

            <a
              href="https://github.com/ricardoleal20/SDAO"
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 bg-stone-900 text-white rounded-xl hover:bg-stone-800 transition-all flex items-center gap-1.5 shadow-sm cursor-pointer border border-stone-800"
            >
              <Github size={14} /> GitHub Repo
            </a>
          </div>

          <button className="lg:hidden text-stone-900 p-2 focus:outline-none" onClick={() => setMenuOpen(!menuOpen)}>
            {menuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </nav>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="fixed inset-0 z-40 bg-[#F9F8F4] flex flex-col items-center justify-center gap-6 text-lg font-serif animate-fade-in p-6">
          <a href="#simulator" onClick={scrollToSection('simulator')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Live Simulator</a>
          <a href="#theory" onClick={scrollToSection('theory')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Theory & SDE</a>
          <a href="#mechanics" onClick={scrollToSection('mechanics')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Adaptive Engine</a>
          <a href="#results" onClick={scrollToSection('results')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Empirical Results</a>
          <a href="#ablation" onClick={scrollToSection('ablation')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Ablation Study</a>
          <a href="#author" onClick={scrollToSection('author')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Author & Citation</a>
          <a
            href="https://github.com/ricardoleal20/SDAO"
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => setMenuOpen(false)}
            className="px-6 py-3 bg-stone-900 text-white rounded-full shadow-lg cursor-pointer flex items-center gap-2 mt-4 text-sm font-sans font-bold uppercase tracking-wider"
          >
            <Github size={18} /> View GitHub Repo
          </a>
        </div>
      )}

      {/* Header / Hero Banner */}
      <header className="pt-32 pb-16 md:pt-40 md:pb-24 border-b border-stone-200 bg-gradient-to-b from-[#F9F8F4] via-[#F5F4F0] to-[#F9F8F4] relative overflow-hidden">
        {/* Subtle background glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-nobel-gold/10 rounded-full blur-[120px] pointer-events-none"></div>

        <div className="container mx-auto px-6 relative z-10 text-center">
          <div className="inline-flex items-center gap-2 mb-6 px-4 py-1.5 border border-nobel-gold text-nobel-gold text-xs tracking-[0.2em] uppercase font-bold rounded-full bg-white/70 shadow-xs">
            <Sparkles size={14} className="text-nobel-gold animate-spin" /> Discover Analytics (2026) 4:6 • Open Access Research
          </div>

          <h1 className="font-serif text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-medium leading-[1.05] mb-6 text-stone-900 max-w-6xl mx-auto">
            Stochastic Diffusion Adaptive Optimization
          </h1>

          <p className="font-serif italic text-2xl md:text-3xl text-stone-600 mb-8 max-w-4xl mx-auto">
            A novel metaheuristic grounded in diffusion dynamics, Fick's second law, and stochastic modeling.
          </p>

          <p className="max-w-3xl mx-auto text-base md:text-lg text-stone-700 font-light leading-relaxed mb-12">
            By replacing traditional gradient descent with a density-driven diffusion mechanism (D_FL), SDAO repels candidate solutions from overcrowded local optima basins and conquers noisy, deceptive, and high-dimensional search spaces up to d = 500.
          </p>

          <div className="flex flex-wrap justify-center gap-4">
            <a
              href="#simulator"
              onClick={scrollToSection('simulator')}
              className="px-6 py-3.5 bg-stone-900 hover:bg-stone-800 text-white rounded-xl text-xs font-bold tracking-wider uppercase transition-all flex items-center gap-2 shadow-md cursor-pointer scale-105"
            >
              <Activity size={16} className="text-nobel-gold" /> Launch Live Simulator
            </a>
            <a
              href="https://github.com/ricardoleal20/SDAO"
              target="_blank"
              rel="noopener noreferrer"
              className="px-6 py-3.5 bg-white hover:bg-stone-50 text-stone-900 border border-stone-300 rounded-xl text-xs font-bold tracking-wider uppercase transition-all flex items-center gap-2 shadow-xs cursor-pointer"
            >
              <Github size={16} /> GitHub Code & Protocol
            </a>
            <a
              href="https://doi.org/10.1007/s44257-025-00054-1"
              target="_blank"
              rel="noopener noreferrer"
              className="px-6 py-3.5 bg-white hover:bg-stone-50 text-stone-700 border border-stone-300 rounded-xl text-xs font-bold tracking-wider uppercase transition-all flex items-center gap-2 shadow-xs cursor-pointer"
            >
              <ExternalLink size={16} className="text-nobel-gold" /> View Paper (DOI)
            </a>
          </div>

          {/* Key Metrics Strip */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto mt-16 pt-10 border-t border-stone-300/60">
            <div className="bg-white/80 p-4 rounded-xl border border-stone-200/80 shadow-xs">
              <div className="text-xs text-stone-500 font-bold uppercase tracking-wider">Benchmark Horizon</div>
              <div className="text-2xl font-serif font-bold text-stone-900 mt-1">300 FEs</div>
              <div className="text-[11px] text-stone-400 font-mono mt-0.5">30 Independent runs</div>
            </div>
            <div className="bg-white/80 p-4 rounded-xl border border-stone-200/80 shadow-xs">
              <div className="text-xs text-stone-500 font-bold uppercase tracking-wider">High-d Scalability</div>
              <div className="text-2xl font-serif font-bold text-nobel-gold mt-1">d = 500</div>
              <div className="text-[11px] text-stone-400 font-mono mt-0.5">SOCO11 Large-Scale</div>
            </div>
            <div className="bg-white/80 p-4 rounded-xl border border-stone-200/80 shadow-xs">
              <div className="text-xs text-stone-500 font-bold uppercase tracking-wider">Statistical Power</div>
              <div className="text-2xl font-serif font-bold text-stone-900 mt-1">p &lt; 10⁻⁴⁰</div>
              <div className="text-[11px] text-stone-400 font-mono mt-0.5">One-Way ANOVA</div>
            </div>
            <div className="bg-white/80 p-4 rounded-xl border-stone-200/80 shadow-xs">
              <div className="text-xs text-stone-500 font-bold uppercase tracking-wider">Wilcoxon Wins</div>
              <div className="text-2xl font-serif font-bold text-emerald-700 mt-1">38 / 38</div>
              <div className="text-[11px] text-stone-400 font-mono mt-0.5">CEC 2017 Dominance</div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Assembly */}
      <main className="container mx-auto px-6">
        {/* Interactive Hero Landscape Comparison */}
        <section id="hero-sim">
          <HeroSimulation />
        </section>

        {/* Real-Time SDAO Optimizer Simulator */}
        <section id="simulator">
          <SDAOSimulator />
        </section>

        {/* Theoretical Foundations */}
        <section id="theory">
          <TheoreticalFoundations />
        </section>

        {/* Adaptive Methodology & OBL Mechanics */}
        <section id="mechanics">
          <AdaptiveMechanics />
        </section>

        {/* Empirical Benchmarking Results */}
        <section id="results">
          <BenchmarkResults />
        </section>

        {/* Ablation Study & Sensitivity */}
        <section id="ablation">
          <AblationStudy />
        </section>

        {/* Author Showcase & BibTeX Citation */}
        <section id="author">
          <AuthorCard />
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-stone-900 text-stone-400 py-16 border-t border-stone-800 mt-20">
        <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="text-center md:text-left space-y-2">
            <div className="flex items-center justify-center md:justify-start gap-2 text-white font-serif font-bold text-2xl">
              SDAO Metaheuristic
            </div>
            <p className="text-xs text-stone-400">
              Interactive web representation of "Stochastic diffusion adaptive optimization, a novel metaheuristic approach" by Ricardo M. Leal Lopez (2026).
            </p>
          </div>

          <div className="flex flex-wrap justify-center gap-6 text-xs font-bold text-stone-300 uppercase tracking-wider">
            <a href="https://github.com/ricardoleal20/SDAO" target="_blank" rel="noopener noreferrer" className="hover:text-nobel-gold transition-colors flex items-center gap-1.5">
              <Github size={14} /> GitHub Repository
            </a>
            <a href="https://doi.org/10.1007/s44257-025-00054-1" target="_blank" rel="noopener noreferrer" className="hover:text-nobel-gold transition-colors flex items-center gap-1.5">
              <ExternalLink size={14} /> Discover Analytics Paper
            </a>
          </div>
        </div>

        <div className="container mx-auto px-6 mt-12 pt-8 border-t border-stone-800 text-center text-xs text-stone-500">
          © 2026 The Author(s). Licensed under Creative Commons Attribution 4.0 International License (CC-BY 4.0).
        </div>
      </footer>
    </div>
  );
};

export default App;
